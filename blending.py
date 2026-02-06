# blending.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd
from pathlib import Path

from models_dc import fit_all as dc_fit_all, price_match as dc_price_match
from models import load_trained_targets, predict_proba as ml_predict

def _load_features():
    from config import FEATURES_PARQUET
    df = pd.read_parquet(FEATURES_PARQUET)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values(['League', 'Date'])


BLEND_WEIGHTS_JSON = Path("models/blend_weights.json")

# Map model targets -> output column name builder
OU_LINES = ["0_5","1_5","2_5","3_5","4_5"]
AH_LINES = ["-1_0","-0_5","0_0","+0_5","+1_0"]

# For each target, define a function that maps an ordered class-prob vector to output column names
def _cols_for_target(target: str, class_labels: List[str]) -> List[str]:
    if target == "y_1X2":
        return ["P_1X2_H","P_1X2_D","P_1X2_A"]
    if target == "y_BTTS":
        # assume class order ["N","Y"] or ["Y","N"]; we’ll align by label
        return ["P_BTTS_N","P_BTTS_Y"]
    if target.startswith("y_OU_"):
        l = target.split("_")[-1]
        return [f"P_OU_{l}_U", f"P_OU_{l}_O"]
    if target.startswith("y_AH_"):
        l = target.split("_",2)[2]
        return [f"P_AH_{l}_A", f"P_AH_{l}_P", f"P_AH_{l}_H"]
    if target == "y_GOAL_RANGE":
        return [f"P_GR_{k}" for k in ["0","1","2","3","4","5+"]]
    if target == "y_CS":
        cols = [f"P_CS_{a}_{b}" for a in range(6) for b in range(6)]
        cols.append("P_CS_Other")
        return cols
    if target == "y_HT":
        return ["P_HT_H","P_HT_D","P_HT_A"]
    if target == "y_HTFT":
        return [f"P_HTFT_{a}_{b}" for a in ["H","D","A"] for b in ["H","D","A"]]
    if target.startswith("y_HomeTG_"):
        l = target.split("_",2)[2]
        return [f"P_HomeTG_{l}_U", f"P_HomeTG_{l}_O"]
    if target.startswith("y_AwayTG_"):
        l = target.split("_",2)[2]
        return [f"P_AwayTG_{l}_U", f"P_AwayTG_{l}_O"]
    if target in ["y_HomeCardsY_BAND","y_AwayCardsY_BAND"]:
        side = "Home" if target.startswith("y_Home") else "Away"
        return [f"P_{side}CardsY_{b}" for b in ["0-2","3","4-5","6+"]]
    if target in ["y_HomeCorners_BAND","y_AwayCorners_BAND"]:
        side = "Home" if target.startswith("y_Home") else "Away"
        return [f"P_{side}Corners_{b}" for b in ["0-3","4-5","6-7","8-9","10+"]]
    return []  # targets without DC support will still get P_*; blend weight learning will skip if DC missing

# targets where DC can produce probabilities (others will skip blending and keep ML only)
def _dc_supported(target: str) -> bool:
    return (
        target in ["y_1X2","y_BTTS","y_GOAL_RANGE","y_CS"]
        or target.startswith("y_OU_")
        or target.startswith("y_AH_")
    )

def _align_probs_to_labels(probs: np.ndarray, model_labels: List[str], desired_labels: List[str]) -> np.ndarray:
    out = np.zeros((probs.shape[0], len(desired_labels)))
    for j, lab in enumerate(desired_labels):
        if lab in model_labels:
            out[:, j] = probs[:, model_labels.index(lab)]
    # normalize rows to 1 to be safe
    s = out.sum(axis=1, keepdims=True); s[s==0] = 1.0
    return out / s

def _logloss_multiclass(y_int: np.ndarray, P: np.ndarray) -> float:
    # y_int in [0..K-1], P shape (n,K)
    eps = 1e-12
    return -float(np.mean(np.log(np.clip(P[np.arange(len(y_int)), y_int], eps, 1.0))))

def _opt_alpha(y_int: np.ndarray, p_ml: np.ndarray, p_dc: np.ndarray) -> float:
    # scalar alpha in [0,1] minimizing multiclass logloss of alpha*ml + (1-alpha)*dc
    # coarse grid + local refine
    grid = np.linspace(0, 1, 21)
    best = (0.5, 1e9)
    for a in grid:
        P = a*p_ml + (1-a)*p_dc
        val = _logloss_multiclass(y_int, P)
        if val < best[1]:
            best = (a, val)
    # local refine around best grid point
    a = best[0]
    for step in [0.1, 0.05, 0.02, 0.01]:
        cand = np.clip(np.array([a-2*step, a-step, a, a+step, a+2*step]), 0, 1)
        vals = [ _logloss_multiclass(y_int, a_*p_ml + (1-a_)*p_dc) for a_ in cand ]
        a = cand[int(np.argmin(vals))]
    return float(a)

def learn_blend_weights() -> Dict[str, float]:
    """
    Returns dict: target_name -> alpha (0..1) where alpha weights ML vs DC.
    Saves to models/blend_weights.json
    """
    # Load historical features and trained ML models
    df = _load_features()
    models = load_trained_targets()
    # Compute ML predictions on rows where target is known (in-sample; acceptable for weight selection)
    weights: Dict[str, float] = {}

    # Pre-fit DC once per league from all historical (for speed & determinism)
    base = df.dropna(subset=["FTHG","FTAG","HomeTeam","AwayTeam","League","Date"]).copy()
    dc_params = dc_fit_all(base[["League","Date","HomeTeam","AwayTeam","FTHG","FTAG"]])

    for target, m in models.items():
        if target not in df.columns:
            continue
        sub = df.dropna(subset=[target]).copy()
        if sub.empty:
            continue

        # ML probs
        ml_dict = ml_predict({target: m}, sub)
        p_ml_full = ml_dict[target]  # shape (n, K_ml)
        ml_labels = list(m.classes_)
        # Desired label order (from the data's categorical order)
        desired_labels = list(sub[target].astype("category").cat.categories)
        p_ml = _align_probs_to_labels(p_ml_full, ml_labels, desired_labels)
        y_int = sub[target].astype("category").cat.codes.values

        # DC probs (skip if target not supported)
        if not _dc_supported(target):
            continue

        # Build DC probs per row for this target
        dc_rows = []
        for _, r in sub[["League","HomeTeam","AwayTeam"]].iterrows():
            lg, ht, at = r["League"], r["HomeTeam"], r["AwayTeam"]
            mp = {}
            if lg in dc_params:
                mp = dc_price_match(dc_params[lg], ht, at, max_goals=8)
            # map mp to desired_labels
            if target == "y_1X2":
                vec = [mp.get("DC_1X2_H",0.0), mp.get("DC_1X2_D",0.0), mp.get("DC_1X2_A",0.0)]
                labs = ["H","D","A"]
            elif target == "y_BTTS":
                vec = [mp.get("DC_BTTS_N",0.0), mp.get("DC_BTTS_Y",0.0)]
                labs = ["N","Y"]
            elif target == "y_GOAL_RANGE":
                labs = ["0","1","2","3","4","5+"]
                vec = [mp.get(f"DC_GR_{k}",0.0) for k in labs]
            elif target == "y_CS":
                labs = [f"{a}-{b}" for a in range(6) for b in range(6)] + ["Other"]
                vec = [mp.get(f"DC_CS_{a}_{b}",0.0) for a in range(6) for b in range(6)] + [mp.get("DC_CS_Other",0.0)]
            elif target.startswith("y_OU_"):
                l = target.split("_")[-1]; labs = ["U","O"]
                vec = [mp.get(f"DC_OU_{l}_U",0.0), mp.get(f"DC_OU_{l}_O",0.0)]
            elif target.startswith("y_AH_"):
                l = target.split("_",2)[2]; labs = ["A","P","H"]
                vec = [mp.get(f"DC_AH_{l}_A",0.0), mp.get(f"DC_AH_{l}_P",0.0), mp.get(f"DC_AH_{l}_H",0.0)]
            else:
                vec = None; labs = []
            if vec is None:
                dc_rows.append(None)
            else:
                # align to desired labels
                vec = np.array(vec, dtype=float)
                lab_map = {lab:i for i, lab in enumerate(labs)}
                aligned = np.zeros(len(desired_labels))
                for j, lab in enumerate(desired_labels):
                    if lab in lab_map:
                        aligned[j] = vec[lab_map[lab]]
                s = aligned.sum(); aligned = aligned / s if s>0 else aligned
                dc_rows.append(aligned)
        if any(v is None for v in dc_rows):
            continue
        p_dc = np.vstack(dc_rows)
        # optimize alpha
        alpha = _opt_alpha(y_int, p_ml, p_dc)
        weights[target] = alpha

    # save
    BLEND_WEIGHTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(BLEND_WEIGHTS_JSON, "w") as f:
        json.dump(weights, f, indent=2)
    return weights
