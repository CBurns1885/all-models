#!/usr/bin/env python3
"""
Overnight Automated Parameter Tuning Script
============================================
Two-phase tuning:
  Phase 1 - Greedy sequential search across 6 parameter groups (~43 trials)
  Phase 2 - Joint refinement around best values, testing interactions (~80-120 trials)
Checkpoint/resume, signal handling, CSV+JSON output.

Usage:
    python auto_tune.py              # full run (both phases)
    python auto_tune.py --resume     # resume from last checkpoint
    python auto_tune.py --dry-run    # print plan without running
    python auto_tune.py --phase1     # greedy phase only
    python auto_tune.py --phase2     # joint refinement only (uses existing best)
"""

import sys, io
# Force UTF-8 stdout so emoji in backtest/predict print statements don't crash on Windows cp1252
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import argparse
import csv
import json
import os
import signal
import sys
import time
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

LOG_CSV = OUTPUTS / "tuning_log.csv"
REPORT_JSON = OUTPUTS / "tuning_report.json"
BEST_JSON = OUTPUTS / "tuning_best_params.json"
CHECKPOINT = OUTPUTS / "tuning_checkpoint.json"

# ---------------------------------------------------------------------------
# Parameter search space (greedy sequential: one group at a time)
# ---------------------------------------------------------------------------
PARAM_GROUPS = [
    # Group 1: Calibration scalar (7 trials)
    {
        "name": "calibration",
        "params": {
            "calibration_scalar": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        },
    },
    # Group 2: Style boost magnitude (5 trials)
    {
        "name": "style_boost",
        "params": {
            "style_boost_magnitude": [0.03, 0.05, 0.08, 0.10, 0.12],
        },
    },
    # Group 3: Home advantage multipliers (8 trials)
    {
        "name": "home_advantage",
        "params": {
            "home_adv_1x2_h": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "home_adv_1x2_a": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
        "constraint": lambda p: p["home_adv_1x2_h"] == p["home_adv_1x2_a"],
        "extra_static": {"home_adv_1x2_d": None},  # derived: half of H
    },
    # Group 4: Poisson blend weights (6 trials)
    {
        "name": "poisson_blend",
        "params": {
            "_poisson_scale": [0.7, 0.85, 1.0, 1.15, 1.3, 1.5],
        },
        "transform": lambda p: {
            "poisson_blend_weights": {
                "0_5": round(0.3 * p["_poisson_scale"], 3),
                "1_5": round(0.4 * p["_poisson_scale"], 3),
                "2_5": round(0.5 * p["_poisson_scale"], 3),
                "3_5": round(0.4 * p["_poisson_scale"], 3),
                "4_5": round(0.3 * p["_poisson_scale"], 3),
            },
            "btts_poisson_weight": round(0.35 * p["_poisson_scale"], 3),
        },
    },
    # Group 5: Time half-life (7 trials)
    {
        "name": "time_halflife",
        "params": {
            "time_half_life": [120, 180, 270, 365, 500, 730, 1000],
        },
    },
    # Group 6: ML weight cap (5 trials)
    {
        "name": "ml_weight_cap",
        "params": {
            "ml_weight_cap": [0.70, 0.75, 0.80, 0.85, 0.90],
        },
    },
    # Group 7: DC temperature scaling (6 trials) — T>1 softens DC overconfidence
    {
        "name": "dc_temperature",
        "params": {
            "dc_temperature": [1.0, 1.1, 1.25, 1.5, 1.75, 2.0],
        },
    },
    # Group 8: 1X2-specific DC temperature — 1X2 tends to be overconfident, needs more softening
    # Uses dc_temperature as floor; independent of global temperature
    {
        "name": "dc_temperature_1x2",
        "params": {
            "dc_temperature_1x2": [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
        },
    },
    # Group 9: ML weight cap for 1X2 — DC is purpose-built for match results; reduce ML influence
    {
        "name": "ml_weight_cap_1x2",
        "params": {
            "ml_weight_cap_1x2": [0.40, 0.50, 0.60, 0.70, 0.80],
        },
    },
    # Group 10: Temperature scaling for binary markets (corners, cards, YC, HomeTG, AwayTG)
    # These have no DC signal; LightGBM outputs near-100% confidence but accuracy is 54-87%
    # T > 1 compresses probabilities toward 0.5, reducing overconfidence
    {
        "name": "temperature_binary",
        "params": {
            "temperature_binary": [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0],
        },
    },
    # HomeTG/AwayTG have DC support so are better calibrated — gentler compression than corners
    {
        "name": "temperature_hometg",
        "params": {
            "temperature_hometg": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        },
    },
]


def _expand_combos(group: dict):
    """Yield dicts of param combos for a group, respecting constraints."""
    names = list(group["params"].keys())
    values = list(group["params"].values())

    for combo in product(*values):
        p = dict(zip(names, combo))

        # Apply constraint filter
        constraint = group.get("constraint")
        if constraint and not constraint(p):
            continue

        # Apply transform (replaces raw keys with real override keys)
        transform = group.get("transform")
        if transform:
            p = transform(p)

        # Derive extras
        extra = group.get("extra_static", {})
        for k, v in extra.items():
            if v is None and "home_adv_1x2_h" in p:
                p[k] = round(p["home_adv_1x2_h"] / 2, 3)
            elif v is not None:
                p[k] = v

        yield p


def _trial_count():
    total = 0
    for g in PARAM_GROUPS:
        total += sum(1 for _ in _expand_combos(g))
    return total


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------
_TUNING_PARAMS_PATH = Path(__file__).parent / "outputs" / "tuning_best_params.json"
_MARKET_BACKTEST_CSV = Path(__file__).parent / "outputs" / "market_backtest_analysis.csv"
_PRED_CACHE_PATH = Path(__file__).parent / "outputs" / "tuning_preds_cache.parquet"
_NEUTRAL_OVERRIDES = {
    "calibration_scalar": 1.0,
    "style_boost_magnitude": 0.0,
    "home_adv_1x2_h": 0.0,
    "home_adv_1x2_a": 0.0,
    "home_adv_1x2_d": 0.0,
    "poisson_blend_weights": {"0_5": 0.0, "1_5": 0.0, "2_5": 0.0, "3_5": 0.0, "4_5": 0.0, "5_5": 0.0},
    "btts_poisson_weight": 0.0,
    "time_half_life": 180,
    "ml_weight_cap": 1.0,
    "dc_temperature": 1.0,
    "dc_temperature_1x2": 1.0,
    "ml_weight_cap_1x2": 1.0,
    "temperature_binary": 1.0,
    "temperature_hometg": 1.0,
}
_PRED_CACHE: dict = {}  # in-memory: {"df": DataFrame, "built_at": float}
# Keys whose groups require a full predict_week() re-run (cannot apply in-process)
_SUBPROCESS_KEYS = {"style_boost_magnitude", "time_half_life"}


def _ensure_pred_cache() -> pd.DataFrame:
    """Build or load the prediction cache.  Cache is built once with NEUTRAL overrides
    (no calibration_scalar, no home_adv, etc.) so each trial can apply its overrides
    to the raw P_/DC_ columns in-process.  Returns the cached DataFrame."""
    import subprocess

    if _PRED_CACHE.get("df") is not None:
        return _PRED_CACHE["df"]

    if _PRED_CACHE_PATH.exists():
        df = pd.read_parquet(_PRED_CACHE_PATH)
        if len(df) > 0:
            _PRED_CACHE["df"] = df
            print(f"  [CACHE] Loaded prediction cache: {len(df)} matches")
            return df

    # Build cache: write neutral overrides then run market_backtest.py
    print("  [CACHE] Building prediction cache (run once, ~5 min)...")
    _TUNING_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TUNING_PARAMS_PATH, "w") as f:
        json.dump(_NEUTRAL_OVERRIDES, f, indent=2)

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "market_backtest.py"),
         "--weeks", "20", "--min-confidence", "0.70"],
        text=True, encoding="utf-8", errors="replace", env=env
    )
    if result.returncode != 0:
        raise RuntimeError(f"Cache build failed (exit {result.returncode})")

    if not _PRED_CACHE_PATH.exists():
        raise RuntimeError("market_backtest.py ran but did not produce tuning_preds_cache.parquet")

    df = pd.read_parquet(_PRED_CACHE_PATH)
    _PRED_CACHE["df"] = df
    print(f"  [CACHE] Built prediction cache: {len(df)} matches")
    return df


def _apply_overrides_and_score(cache: pd.DataFrame, overrides: dict,
                                min_conf: float = 0.70) -> dict:
    """Apply TUNING_OVERRIDES to cached P_/DC_ predictions and score accuracy.

    Parameters applied in-process (no re-training or DC re-fitting):
      - calibration_scalar: scale P_ probs toward/from 0.5
      - home_adv_1x2_*: multiplicative adjustment to 1X2 P_ columns
      - dc_temperature: temperature-scale DC_ columns
      - poisson_blend_weights / btts_poisson_weight: re-blend P_ + DC_
      - ml_weight_cap: cap max P_ probability
      - style_boost_magnitude, time_half_life: require prediction re-run — skipped here
        (subprocess fallback used for those groups)
    """
    df = cache.copy()

    cs = float(overrides.get("calibration_scalar", 1.0))
    ha_h = float(overrides.get("home_adv_1x2_h", 0.0))
    ha_a = float(overrides.get("home_adv_1x2_a", 0.0))
    ha_d = float(overrides.get("home_adv_1x2_d", 0.0))
    dc_temp = float(overrides.get("dc_temperature", 1.0))
    ml_cap = float(overrides.get("ml_weight_cap", 1.0))
    pw = overrides.get("poisson_blend_weights", {})
    bw = float(overrides.get("btts_poisson_weight", 0.0))
    # Per-market 1X2 params (fall back to global if not set)
    dc_temp_1x2 = float(overrides.get("dc_temperature_1x2", dc_temp))
    ml_cap_1x2 = float(overrides.get("ml_weight_cap_1x2", ml_cap))

    def renorm(arr: np.ndarray) -> np.ndarray:
        s = arr.sum(axis=1, keepdims=True)
        s = np.where(s == 0, 1.0, s)
        return arr / s

    def apply_cs(cols):
        if cs == 1.0:
            return
        vals = df[cols].values.astype(float)
        vals = renorm(0.5 + (vals - 0.5) * cs)
        df[cols] = vals

    def apply_dc_temp(dc_cols):
        if dc_temp == 1.0:
            return
        vals = df[dc_cols].values.astype(float)
        log_v = np.log(np.clip(vals, 1e-10, None)) / dc_temp
        log_v -= log_v.max(axis=1, keepdims=True)
        df[dc_cols] = renorm(np.exp(log_v))

    # Apply calibration scalar to all P_ columns
    p_cols_all = [c for c in df.columns if c.startswith("P_")]
    if p_cols_all and cs != 1.0:
        vals = df[p_cols_all].values.astype(float)
        # Scale each group toward 0.5 (don't cross-normalise across markets)
        df[p_cols_all] = np.clip(0.5 + (vals - 0.5) * cs, 0.0, 1.0)

    # Apply home advantage to 1X2
    for col, ha in [("P_1X2_H", ha_h), ("P_1X2_A", ha_a), ("P_1X2_D", ha_d)]:
        if col in df.columns and ha != 0.0:
            if "H" in col:
                df[col] = np.minimum(df[col].values.astype(float) * (1 + ha), 0.95)
            elif "A" in col:
                df[col] = np.maximum(df[col].values.astype(float) * (1 - ha), 0.05)
            else:
                df[col] = np.clip(df[col].values.astype(float) * (1 - ha), 0.05, 0.45)

    # Apply ml_weight_cap
    if ml_cap < 1.0:
        for c in p_cols_all:
            if c in df.columns:
                df[c] = np.minimum(df[c].values.astype(float), ml_cap)

    # Re-blend 1X2 with per-market DC temperature + ML weight cap
    dc_cols_1x2 = [c for c in ["DC_1X2_H", "DC_1X2_D", "DC_1X2_A"] if c in df.columns]
    p_cols_1x2  = [c for c in ["P_1X2_H", "P_1X2_D", "P_1X2_A"]   if c in df.columns]
    if len(dc_cols_1x2) == 3 and len(p_cols_1x2) == 3:
        dc_vals = df[dc_cols_1x2].values.astype(float)
        if dc_temp_1x2 != 1.0:
            log_v = np.log(np.clip(dc_vals, 1e-10, None)) / dc_temp_1x2
            log_v -= log_v.max(axis=1, keepdims=True)
            dc_vals = renorm(np.exp(log_v))
        p_vals = df[p_cols_1x2].values.astype(float)
        blend_1x2 = renorm(ml_cap_1x2 * p_vals + (1 - ml_cap_1x2) * dc_vals)
        df["BLEND_1X2_H"] = blend_1x2[:, 0]
        df["BLEND_1X2_D"] = blend_1x2[:, 1]
        df["BLEND_1X2_A"] = blend_1x2[:, 2]

    # Re-blend OU markets
    for line_tag in ["0_5", "1_5", "2_5", "3_5", "4_5", "5_5"]:
        alpha = float(pw.get(line_tag, 0.0))
        p_o = f"P_OU_{line_tag}_O"; p_u = f"P_OU_{line_tag}_U"
        d_o = f"DC_OU_{line_tag}_O"; d_u = f"DC_OU_{line_tag}_U"
        b_o = f"BLEND_OU_{line_tag}_O"; b_u = f"BLEND_OU_{line_tag}_U"
        if p_o in df.columns and d_o in df.columns:
            df[b_o] = alpha * df[p_o].values.astype(float) + (1 - alpha) * df[d_o].values.astype(float)
            df[b_u] = alpha * df[p_u].values.astype(float) + (1 - alpha) * df[d_u].values.astype(float)

    # Re-blend BTTS
    if "P_BTTS_Y" in df.columns and "DC_BTTS_Y" in df.columns and bw != 0.0:
        df["BLEND_BTTS_Y"] = bw * df["P_BTTS_Y"].values.astype(float) + \
                             (1 - bw) * df["DC_BTTS_Y"].values.astype(float)
        df["BLEND_BTTS_N"] = 1 - df["BLEND_BTTS_Y"]

    # Apply temperature_binary to pure-ML binary markets (corners, YC, cards) — NOT HomeTG/AwayTG
    # HomeTG/AwayTG have DC support and use temperature_hometg (separate param below)
    temp_b = float(overrides.get("temperature_binary", 1.0))
    if temp_b != 1.0:
        binary_prefixes = ("P_TotalCorners_", "P_HomeCorners_", "P_AwayCorners_",
                           "P_TotalYC_", "P_BookingPts_", "P_HomeTeam_Card", "P_AwayTeam_Card")
        for col in df.columns:
            if any(col.startswith(pfx) for pfx in binary_prefixes):
                vals = df[col].values.astype(float)
                log_odds = np.log(np.clip(vals, 1e-9, 1 - 1e-9) / np.clip(1 - vals, 1e-9, 1 - 1e-9)) / temp_b
                df[col] = 1.0 / (1.0 + np.exp(-log_odds))

    # Apply temperature_hometg to DC-supported TG markets (HomeTG, AwayTG)
    temp_htg = float(overrides.get("temperature_hometg", 1.0))
    if temp_htg != 1.0:
        for col in df.columns:
            if col.startswith("P_HomeTG_") or col.startswith("P_AwayTG_"):
                vals = df[col].values.astype(float)
                log_odds = np.log(np.clip(vals, 1e-9, 1 - 1e-9) / np.clip(1 - vals, 1e-9, 1 - 1e-9)) / temp_htg
                df[col] = 1.0 / (1.0 + np.exp(-log_odds))

    # Score markets
    MARKET_PAIRS = {
        "1X2":          (["BLEND_1X2_H", "BLEND_1X2_D", "BLEND_1X2_A"], ["FTR"], {"H": "H", "D": "D", "A": "A"}),
        "BTTS":         (["P_BTTS_Y", "P_BTTS_N"],           ["BTTS_actual"], None),
        "OU_2_5":       (["P_OU_2_5_O", "P_OU_2_5_U"],       ["OU_2_5_actual"], None),
        "OU_1_5":       (["P_OU_1_5_O", "P_OU_1_5_U"],       ["OU_1_5_actual"], None),
        "OU_3_5":       (["P_OU_3_5_O", "P_OU_3_5_U"],       ["OU_3_5_actual"], None),
        "HomeTG_0_5":   (["P_HomeTG_0_5_O", "P_HomeTG_0_5_U"], ["y_HomeTG_0_5"], None),
        "AwayTG_0_5":   (["P_AwayTG_0_5_O", "P_AwayTG_0_5_U"], ["y_AwayTG_0_5"], None),
        "TotalCorners_O9_5": (["P_TotalCorners_O9_5_Y", "P_TotalCorners_O9_5_N"], ["y_TotalCorners_O9_5"], None),
        "TotalYC_O3_5": (["P_TotalYC_O3_5_Y", "P_TotalYC_O3_5_N"], ["y_TotalYC_O3_5"], None),
    }

    # Precompute actual values if not already in cache
    if "BTTS_actual" not in df.columns:
        if "FTHG" in df.columns and "FTAG" in df.columns:
            total = df["FTHG"].fillna(0).astype(int) + df["FTAG"].fillna(0).astype(int)
            df["BTTS_actual"] = np.where(
                (df["FTHG"].fillna(0).astype(int) > 0) & (df["FTAG"].fillna(0).astype(int) > 0), "Y", "N")
            for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                tag = str(line).replace(".", "_")
                df[f"OU_{tag}_actual"] = np.where(total > line, "O", "U")

    results = []
    for market, (prob_cols, actual_cols, label_map) in MARKET_PAIRS.items():
        pcols = [c for c in prob_cols if c in df.columns]
        acol = next((c for c in actual_cols if c in df.columns), None)
        if not pcols or not acol:
            continue
        probs = df[pcols].values.astype(float)
        pred_idx = probs.argmax(axis=1)
        actual_raw = df[acol].values

        # Normalise actual labels to match prob column suffixes
        labels = [col.split("_")[-1] for col in pcols]
        if label_map:
            actual = np.array([label_map.get(str(a), str(a)) for a in actual_raw])
        else:
            actual = np.array([str(a) for a in actual_raw])

        # One-hot encode for real Brier score (sensitive to probability values, not just argmax)
        one_hot = np.zeros_like(probs)
        for j, label in enumerate(labels):
            one_hot[:, j] = (actual == label).astype(float)

        brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
        pred_labels = np.array([labels[i] for i in pred_idx])
        accuracy = float((pred_labels == actual).mean())

        results.append({
            "market": market,
            "n": len(actual),
            "brier": brier,
            "accuracy": accuracy,
        })

    if not results:
        return {"accuracy": 0.0, "brier": 1.0, "weighted_brier": 1.0, "roi": -100.0}

    total_n = sum(r["n"] for r in results)
    weighted_brier = sum(r["brier"] * r["n"] for r in results) / total_n
    avg_acc = float(np.mean([r["accuracy"] for r in results])) * 100

    return {
        "accuracy": round(avg_acc, 2),
        "brier": round(weighted_brier, 4),
        "weighted_brier": round(weighted_brier, 4),
        "roi": 0.0,
        "markets": results,
    }


def run_backtest(overrides: dict) -> dict:
    """Score TUNING_OVERRIDES via market_backtest.py subprocess (~5 min/trial with --weeks 1).

    Writes overrides to tuning_best_params.json so predict.py picks them up,
    then calls market_backtest.py --weeks 1 and reads the per-market accuracy CSV.
    """
    import subprocess

    start = time.time()
    try:
        _TUNING_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_TUNING_PARAMS_PATH, "w") as f:
            json.dump(overrides, f, indent=2)

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "market_backtest.py"),
             "--weeks", "20", "--min-confidence", "0.70"],
            text=True, encoding="utf-8", errors="replace", env=env
        )
        if result.returncode != 0:
            raise RuntimeError(f"subprocess exit {result.returncode}")

        if not _MARKET_BACKTEST_CSV.exists():
            raise RuntimeError("no analysis CSV produced")

        df_res = pd.read_csv(_MARKET_BACKTEST_CSV)
        if df_res.empty:
            raise RuntimeError("empty results CSV")

        acc_col   = next((c for c in ["accuracy", "Accuracy_%", "Accuracy"] if c in df_res.columns), None)
        brier_col = next((c for c in ["brier_score", "brier", "Brier_Score", "Brier"] if c in df_res.columns), None)
        roi_col   = next((c for c in ["roi_fair", "ROI_%", "ROI", "roi"]   if c in df_res.columns), None)

        acc   = float(df_res[acc_col].mean()   * 100) if acc_col   else 0.0
        brier = float(df_res[brier_col].mean())        if brier_col else round(1 - acc / 100, 4)
        roi   = float(df_res[roi_col].mean())          if roi_col   else 0.0

        return {
            "accuracy": round(acc, 2),
            "brier": round(brier, 4),
            "weighted_brier": round(brier, 4),
            "roi": round(roi, 2),
            "duration_s": round(time.time() - start, 1),
        }

    except Exception as e:
        print(f"  [ERROR] Backtest failed: {e}")
        return {"accuracy": 0.0, "brier": 1.0, "weighted_brier": 1.0,
                "roi": -100.0, "duration_s": round(time.time() - start, 1)}


def score_result(r: dict) -> float:
    """Single scalar score: higher is better.

    Primary: weighted Brier (60%) — calibration is what the post-processing
    layer actually moves, so it should dominate the objective.
    Secondary: ROI (25%) — real-odds profit where available.
    Tertiary: accuracy (15%) — sanity check only.
    """
    brier_contrib = (1 - r.get("weighted_brier", r["brier"])) * 100 * 0.60
    roi_contrib = max(r["roi"], -50) * 0.25
    acc_contrib = r["accuracy"] * 0.15
    return brier_contrib + roi_contrib + acc_contrib


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(state: dict):
    CHECKPOINT.write_text(json.dumps(state, indent=2))


def load_checkpoint() -> dict | None:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return None


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------
def init_csv():
    if not LOG_CSV.exists():
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial", "group", "params_json",
                "accuracy", "brier", "roi", "score", "duration_s", "timestamp",
            ])


def append_csv(trial: int, group: str, params: dict, result: dict, score: float):
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            trial, group, json.dumps(params),
            result["accuracy"], result["brier"], result["roi"],
            round(score, 3), result["duration_s"],
            datetime.now().isoformat(timespec="seconds"),
        ])


# ---------------------------------------------------------------------------
# Signal handling for graceful shutdown
# ---------------------------------------------------------------------------
_shutdown = False


def _handle_signal(sig, frame):
    global _shutdown
    print(f"\n[SIGNAL] Received {signal.Signals(sig).name}. Will stop after current trial.")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Joint refinement: build neighbouring combos around greedy-best values
# ---------------------------------------------------------------------------
def _neighbours(value, candidates):
    """Return the value and its immediate neighbours from a sorted candidate list."""
    if value not in candidates:
        return [value]
    idx = candidates.index(value)
    lo = max(0, idx - 1)
    hi = min(len(candidates), idx + 2)
    return candidates[lo:hi]


def _build_joint_refinement(best_params: dict) -> list[tuple[str, dict]]:
    """Build joint-refinement combos: pairwise and 3-way crosses around the greedy best.

    Tests parameter interactions the greedy pass cannot detect.
    """
    combos = []

    # --- Pair 1: calibration_scalar × style_boost_magnitude ---
    cal_vals = _neighbours(
        best_params.get("calibration_scalar", 0.2),
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    )
    style_vals = _neighbours(
        best_params.get("style_boost_magnitude", 0.05),
        [0.03, 0.05, 0.08, 0.10, 0.12],
    )
    for cal, sty in product(cal_vals, style_vals):
        combo = {"calibration_scalar": cal, "style_boost_magnitude": sty}
        combos.append(("cal×style", combo))

    # --- Pair 2: home_adv × poisson_scale ---
    home_vals = _neighbours(
        best_params.get("home_adv_1x2_h", 0.2),
        [0.15, 0.2, 0.3, 0.4],
    )
    poisson_candidates = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    # Find closest poisson scale from stored blend weights
    best_btts = best_params.get("btts_poisson_weight", 0.35)
    best_scale = round(best_btts / 0.35, 2) if best_btts else 1.0
    closest_scale = min(poisson_candidates, key=lambda x: abs(x - best_scale))
    poisson_vals = _neighbours(closest_scale, poisson_candidates)
    for hv, ps in product(home_vals, poisson_vals):
        combo = {
            "home_adv_1x2_h": hv,
            "home_adv_1x2_a": hv,
            "home_adv_1x2_d": round(hv / 2, 3),
            "poisson_blend_weights": {
                "0_5": round(0.3 * ps, 3),
                "1_5": round(0.4 * ps, 3),
                "2_5": round(0.5 * ps, 3),
                "3_5": round(0.4 * ps, 3),
                "4_5": round(0.3 * ps, 3),
            },
            "btts_poisson_weight": round(0.35 * ps, 3),
        }
        combos.append(("home×poisson", combo))

    # --- Pair 3: time_half_life × ml_weight_cap ---
    time_vals = _neighbours(
        best_params.get("time_half_life", 180),
        [90, 120, 150, 180, 210, 270, 365],
    )
    ml_vals = _neighbours(
        best_params.get("ml_weight_cap", 0.80),
        [0.70, 0.75, 0.80, 0.85, 0.90],
    )
    for tv, mv in product(time_vals, ml_vals):
        combo = {"time_half_life": tv, "ml_weight_cap": mv}
        combos.append(("time×ml", combo))

    # --- Pair 4: dc_temperature × ml_weight_cap ---
    dc_temp_vals = _neighbours(
        best_params.get("dc_temperature", 1.0),
        [1.0, 1.1, 1.25, 1.5, 1.75, 2.0],
    )
    for dt, mv in product(dc_temp_vals, ml_vals):
        combo = {"dc_temperature": dt, "ml_weight_cap": mv}
        combos.append(("dc_temp×ml", combo))

    # --- 3-way cross: calibration × time_half_life × ml_weight_cap ---
    for cal, tv, mv in product(cal_vals, time_vals, ml_vals):
        combo = {"calibration_scalar": cal, "time_half_life": tv, "ml_weight_cap": mv}
        combos.append(("cal×time×ml", combo))

    # Deduplicate (same param dict can appear in multiple pairs)
    seen = set()
    unique = []
    for name, combo in combos:
        key = json.dumps(combo, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append((name, combo))

    return unique


def _score_trial(overrides: dict, combo: dict) -> dict:
    """Route to fast in-process scoring or slow subprocess depending on combo keys.

    In-process: applies overrides to cached P_/DC_ columns (~1s per trial).
    Subprocess: calls market_backtest.py for params that need predict re-run (~17 min).
    """
    needs_subprocess = bool(set(combo.keys()) & _SUBPROCESS_KEYS)
    cache = _PRED_CACHE.get("df")
    if cache is not None and not needs_subprocess:
        start = time.time()
        result = _apply_overrides_and_score(cache, overrides)
        result["duration_s"] = round(time.time() - start, 1)
        return result
    return run_backtest(overrides)


def main():
    parser = argparse.ArgumentParser(description="Overnight automated parameter tuning")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--phase1", action="store_true", help="Run greedy phase only")
    parser.add_argument("--phase2", action="store_true", help="Run joint refinement only (uses existing best)")
    args = parser.parse_args()

    run_phase1 = not args.phase2  # run phase 1 unless --phase2 only
    run_phase2 = not args.phase1  # run phase 2 unless --phase1 only

    greedy_trials = _trial_count()

    if args.dry_run:
        print(f"=== PHASE 1: Greedy search — {greedy_trials} trials across {len(PARAM_GROUPS)} groups ===")
        for g in PARAM_GROUPS:
            combos = list(_expand_combos(g))
            print(f"  {g['name']}: {len(combos)} trials")
            for c in combos:
                print(f"    {c}")
        print(f"\n=== PHASE 2: Joint refinement — ~80-120 trials (depends on Phase 1 results) ===")
        print("  Pairs tested: calibration×style_boost, home_adv×poisson, time_halflife×ml_cap")
        print("  Plus a final 3-way cross of the top movers")
        return

    # Load checkpoint if resuming
    best_params: dict = {}
    best_score: float = -999.0
    start_group = 0
    trial_counter = 0
    phase = 1

    if args.resume:
        ckpt = load_checkpoint()
        if ckpt:
            best_params = ckpt.get("best_params", {})
            best_score = ckpt.get("best_score", -999.0)
            start_group = ckpt.get("next_group", 0)
            trial_counter = ckpt.get("trial_counter", 0)
            phase = ckpt.get("phase", 1)
            print(f"[RESUME] Phase {phase}, group {start_group}, trial {trial_counter}, best score {best_score:.3f}")
            if phase == 2:
                run_phase1 = False
        else:
            print("[RESUME] No checkpoint found, starting fresh.")

    if args.phase2 and not args.resume:
        # Load best params from previous run
        if BEST_JSON.exists():
            best_params = json.loads(BEST_JSON.read_text())
            print(f"[PHASE2] Loaded best params from {BEST_JSON}")
        else:
            print("[PHASE2] No best params found. Run phase 1 first or provide tuning_best_params.json")
            sys.exit(1)

    init_csv()
    overall_start = time.time()

    # Snapshot incumbent best score AND params BEFORE any trial writes to BEST_JSON.
    # run_backtest() overwrites BEST_JSON with trial params (no _tuning_score),
    # so _write_outputs() must use this pre-run snapshot rather than reading the file.
    _incumbent_score = -999.0
    _incumbent_params: dict = {}
    if BEST_JSON.exists():
        try:
            _incumbent_data = json.loads(BEST_JSON.read_text())
            _incumbent_score = float(_incumbent_data.get("_tuning_score", -999.0))
            _incumbent_params = _incumbent_data
            print(f"[TUNE] Incumbent best score: {_incumbent_score:.3f}")
        except Exception:
            pass

    # ===================================================================
    # Load prediction cache for in-process scoring (avoids ~17 min subprocess per trial)
    # ===================================================================
    print("\n[CACHE] Loading prediction cache for in-process scoring...")
    try:
        _ensure_pred_cache()
        print(f"[CACHE] Ready — in-process scoring active for non-subprocess groups")
    except Exception as _ce:
        print(f"[CACHE] Warning: could not load cache ({_ce}). All trials will use subprocess.")

    # ===================================================================
    # PHASE 1: Greedy sequential search
    # ===================================================================
    if run_phase1:
        print(f"\n{'='*60}")
        print(f"  PHASE 1: Greedy Sequential Search ({greedy_trials} trials)")
        print(f"{'='*60}")

        for gi, group in enumerate(PARAM_GROUPS):
            if gi < start_group:
                continue

            group_name = group["name"]
            combos = list(_expand_combos(group))
            print(f"\n--- Group {gi+1}/{len(PARAM_GROUPS)}: {group_name} ({len(combos)} trials) ---")

            group_best_score = -999.0
            group_best_params = {}

            for ci, combo in enumerate(combos):
                if _shutdown:
                    print("[SHUTDOWN] Saving checkpoint and exiting.")
                    save_checkpoint({
                        "best_params": best_params,
                        "best_score": best_score,
                        "next_group": gi,
                        "trial_counter": trial_counter,
                        "phase": 1,
                    })
                    _write_outputs(best_params, best_score, overall_start, _incumbent_score, _incumbent_params)
                    sys.exit(0)

                trial_counter += 1
                overrides = {**best_params, **combo}

                print(f"  Trial {trial_counter}: {combo}")
                result = _score_trial(overrides, combo)
                sc = score_result(result)
                print(f"    -> acc={result['accuracy']:.1f}% brier={result['brier']:.4f} roi={result['roi']:.1f}% score={sc:.3f} ({result['duration_s']:.0f}s)")

                append_csv(trial_counter, group_name, combo, result, sc)

                if sc > group_best_score:
                    group_best_score = sc
                    group_best_params = combo

            best_params.update(group_best_params)
            if group_best_score > best_score:
                best_score = group_best_score
            print(f"  -> Best for {group_name}: {group_best_params} (score={group_best_score:.3f})")

            save_checkpoint({
                "best_params": best_params,
                "best_score": best_score,
                "next_group": gi + 1,
                "trial_counter": trial_counter,
                "phase": 1,
            })

        print(f"\n  Phase 1 complete. Greedy best score: {best_score:.3f}")

    # ===================================================================
    # PHASE 2: Joint refinement
    # ===================================================================
    if run_phase2:
        print(f"\n{'='*60}")
        print(f"  PHASE 2: Joint Refinement (parameter interactions)")
        print(f"{'='*60}")

        save_checkpoint({
            "best_params": best_params,
            "best_score": best_score,
            "next_group": 0,
            "trial_counter": trial_counter,
            "phase": 2,
        })

        phase2_best = deepcopy(best_params)
        phase2_score = best_score

        joint_combos = _build_joint_refinement(best_params)
        total_joint = len(joint_combos)
        print(f"  {total_joint} joint refinement trials planned\n")

        for ji, (joint_name, combo) in enumerate(joint_combos):
            if _shutdown:
                print("[SHUTDOWN] Saving checkpoint and exiting.")
                save_checkpoint({
                    "best_params": phase2_best,
                    "best_score": phase2_score,
                    "next_group": ji,
                    "trial_counter": trial_counter,
                    "phase": 2,
                })
                _write_outputs(phase2_best, phase2_score, overall_start, _incumbent_score, _incumbent_params)
                sys.exit(0)

            trial_counter += 1
            overrides = {**phase2_best, **combo}

            print(f"  Joint {ji+1}/{total_joint} [{joint_name}]: {combo}")
            result = _score_trial(overrides, combo)
            sc = score_result(result)
            print(f"    -> acc={result['accuracy']:.1f}% brier={result['brier']:.4f} roi={result['roi']:.1f}% score={sc:.3f} ({result['duration_s']:.0f}s)")

            append_csv(trial_counter, f"joint_{joint_name}", combo, result, sc)

            if sc > phase2_score:
                phase2_score = sc
                phase2_best.update(combo)
                print(f"    *** NEW BEST: {sc:.3f} ***")

        best_params = phase2_best
        best_score = phase2_score
        print(f"\n  Phase 2 complete. Joint-refined best score: {best_score:.3f}")

    _write_outputs(best_params, best_score, overall_start, _incumbent_score, _incumbent_params)
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

    print(f"\n=== TUNING COMPLETE === Best score: {best_score:.3f}")
    print(f"Best params saved to: {BEST_JSON}")


def _write_outputs(best_params: dict, best_score: float, start_time: float,
                   incumbent_score: float = -999.0, incumbent_params: dict = None):
    """Write final JSON outputs."""
    elapsed = time.time() - start_time

    # Guard: only overwrite if this run beats the pre-run incumbent.
    # Cannot read from BEST_JSON here — run_backtest() overwrites it with trial params
    # (no _tuning_score), so reading it would always return -999.0.
    if best_score >= incumbent_score:
        payload = {**best_params, "_tuning_score": round(best_score, 4)}
        BEST_JSON.write_text(json.dumps(payload, indent=2))
        print(f"[TUNE] Params updated (score {incumbent_score:.3f} → {best_score:.3f})")
    else:
        # Restore the incumbent params so BEST_JSON is not left with trial garbage.
        if incumbent_params:
            BEST_JSON.write_text(json.dumps(incumbent_params, indent=2))
        print(f"[TUNE] Keeping existing params (stored best {incumbent_score:.3f} > this run {best_score:.3f})")

    report = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "best_score": round(best_score, 3),
        "best_params": best_params,
        "log_file": str(LOG_CSV),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
