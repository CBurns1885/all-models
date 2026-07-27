# models_counts.py
"""
Negative-binomial count models for corners and cards.

The classifier stack models each line (Over 6.5, Over 7.5, ... corners) as an
INDEPENDENT binary problem: nothing forces P(Over 7.5) <= P(Over 6.5), each
line trains on its own thin labels, and tail lines are chronically
miscalibrated. Here we instead model the COUNT distribution once per family —

    TotalCorners, HomeCorners, AwayCorners, TotalYC, HomeYC, AwayYC

— with a Poisson-loss gradient-boosted regressor for the mean m(x) plus a
method-of-moments negative-binomial dispersion alpha (Var = m + alpha*m^2;
corners and cards are overdispersed, so plain Poisson underestimates tails).
Every line probability is then derived from the same NB distribution:
monotone across lines by construction, coherent tails, and one model per
family instead of eight.

Integrated as the "nb" pseudo-base in models.py for corners/cards targets.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.impute import SimpleImputer

from feature_rules import feature_exclusions

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
from sklearn.ensemble import HistGradientBoostingRegressor

# Count families: name -> (target column in features.parquet, mean clip)
COUNT_FAMILIES = {
    "TotalCorners": ("__TOTAL_CORNERS__", (2.0, 20.0)),
    "HomeCorners":  ("Home_Corners",      (1.0, 12.0)),
    "AwayCorners":  ("Away_Corners",      (1.0, 12.0)),
    "TotalYC":      ("__TOTAL_YC__",      (0.5, 12.0)),
    "HomeYC":       ("Home_CardsY",       (0.2, 8.0)),
    "AwayYC":       ("Away_CardsY",       (0.2, 8.0)),
}

MAX_COUNT = 30   # support upper bound for pmf sums

_NB_CACHE: Dict[str, "CountBundle"] = {}


@dataclass
class CountModel:
    model: object
    alpha: float          # NB dispersion (0 => Poisson)
    clip: tuple


@dataclass
class CountBundle:
    imputer: SimpleImputer
    feat_cols: list
    models: Dict[str, CountModel] = field(default_factory=dict)


def _numeric_feature_cols(df: pd.DataFrame) -> list:
    excl = feature_exclusions()
    return [c for c in df.columns
            if not c.startswith("y_") and c not in excl
            and pd.api.types.is_numeric_dtype(df[c])]


def _make_regressor(n_estimators: int = 200):
    if _HAS_LGB:
        return lgb.LGBMRegressor(
            objective="poisson", n_estimators=n_estimators,
            learning_rate=0.05, num_leaves=31, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8, verbose=-1)
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=n_estimators,
        learning_rate=0.05, min_samples_leaf=50)


def _family_target(sub: pd.DataFrame, family: str) -> Optional[pd.Series]:
    col, _ = COUNT_FAMILIES[family]
    if col == "__TOTAL_CORNERS__":
        if "Home_Corners" not in sub.columns or "Away_Corners" not in sub.columns:
            return None
        return (pd.to_numeric(sub["Home_Corners"], errors="coerce")
                + pd.to_numeric(sub["Away_Corners"], errors="coerce"))
    if col == "__TOTAL_YC__":
        if "Home_CardsY" not in sub.columns or "Away_CardsY" not in sub.columns:
            return None
        return (pd.to_numeric(sub["Home_CardsY"], errors="coerce")
                + pd.to_numeric(sub["Away_CardsY"], errors="coerce"))
    if col not in sub.columns:
        return None
    return pd.to_numeric(sub[col], errors="coerce")


def _estimate_alpha(y: np.ndarray, m: np.ndarray) -> float:
    """Method-of-moments NB dispersion: solve Var(y|m) = m + alpha*m^2."""
    num = np.sum((y - m) ** 2 - m)
    den = np.sum(m ** 2)
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 2.0))


def fit_counts(train_df: pd.DataFrame, use_cache: bool = True) -> Optional[CountBundle]:
    d = pd.to_datetime(train_df["Date"])
    key = hashlib.md5(f"nb|{str(d.min())[:10]}|{str(d.max())[:10]}".encode()).hexdigest()[:16]
    if use_cache and key in _NB_CACHE:
        return _NB_CACHE[key]

    feat_cols = _numeric_feature_cols(train_df)
    if len(feat_cols) < 5:
        return None

    bundle = None
    for family in COUNT_FAMILIES:
        y_all = _family_target(train_df, family)
        if y_all is None:
            continue
        mask = y_all.notna() & (y_all >= 0)
        if mask.sum() < 500:
            continue
        sub = train_df.loc[mask]
        y = y_all.loc[mask].to_numpy(dtype=float)

        if bundle is None:
            imputer = SimpleImputer(strategy="median")
            imputer.fit(train_df[feat_cols])
            bundle = CountBundle(imputer=imputer, feat_cols=feat_cols)

        X = bundle.imputer.transform(sub[feat_cols])
        model = _make_regressor()
        model.fit(X, y)
        _, clip = COUNT_FAMILIES[family]
        m = np.clip(model.predict(X), *clip)
        alpha = _estimate_alpha(y, m)
        bundle.models[family] = CountModel(model=model, alpha=alpha, clip=clip)

    if bundle is None or not bundle.models:
        return None
    if use_cache:
        _NB_CACHE[key] = bundle
    return bundle


def _count_pmf(m: float, alpha: float, k_max: int = MAX_COUNT) -> np.ndarray:
    """pmf over 0..k_max for NB(mean=m, dispersion=alpha); Poisson if alpha~0."""
    ks = np.arange(k_max + 1)
    if alpha < 1e-6:
        pmf = poisson.pmf(ks, m)
    else:
        n = 1.0 / alpha          # NB size
        p = n / (n + m)          # success prob parameterisation with mean m
        pmf = nbinom.pmf(ks, n, p)
    s = pmf.sum()
    return pmf / s if s > 0 else pmf


def predict_family_means(bundle: CountBundle, rows_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    X = pd.DataFrame(index=rows_df.index)
    for c in bundle.feat_cols:
        X[c] = pd.to_numeric(rows_df[c], errors="coerce") if c in rows_df.columns else np.nan
    Xi = bundle.imputer.transform(X[bundle.feat_cols])
    out = {}
    for family, cm in bundle.models.items():
        out[family] = np.clip(cm.model.predict(Xi), *cm.clip)
    return out


def prob_over(bundle: CountBundle, family: str, mean: float, line: float) -> float:
    """P(count > line) from the family's NB distribution."""
    cm = bundle.models[family]
    pmf = _count_pmf(mean, cm.alpha)
    return float(pmf[np.arange(len(pmf)) > line].sum())


def prob_home_wins_count(bundle: CountBundle, m_home: float, m_away: float) -> float:
    """P(home count > away count) assuming independent NB marginals."""
    ph = _count_pmf(m_home, bundle.models["HomeCorners"].alpha)
    pa = _count_pmf(m_away, bundle.models["AwayCorners"].alpha)
    # P(H > A) = sum_i ph[i] * P(A < i)
    ca = np.cumsum(pa)
    return float(sum(ph[i] * (ca[i - 1] if i > 0 else 0.0) for i in range(len(ph))))


# Targets this pseudo-base can price, mapped to (family, line spec)
def nb_supported(target: str) -> bool:
    return (target.startswith("y_TotalCorners_O")
            or target.startswith("y_HomeCorners_O")
            or target.startswith("y_AwayCorners_O")
            or target.startswith("y_TotalYC_O")
            or target in ("y_HomeTeam_Card", "y_AwayTeam_Card", "y_HomeCorners_Win"))


def _line_from_target(target: str, prefix: str) -> float:
    # e.g. "y_TotalCorners_O9_5" -> 9.5
    tail = target.replace(prefix, "")          # "O9_5"
    return float(tail[1:].replace("_", "."))


def nb_probs_for_rows(train_df: pd.DataFrame, rows_df: pd.DataFrame,
                      target: str) -> np.ndarray:
    """Class-probability matrix for a corners/cards target, classes in the
    SORTED categorical order used by models.py ('N' < 'Y')."""
    n = len(rows_df)
    bundle = fit_counts(train_df)
    if bundle is None:
        return np.zeros((n, 2))

    means = predict_family_means(bundle, rows_df)

    def col(family, fn):
        if family not in bundle.models or family not in means:
            return np.zeros(n)
        return np.array([fn(m) for m in means[family]])

    if target.startswith("y_TotalCorners_O"):
        line = _line_from_target(target, "y_TotalCorners_")
        p_over = col("TotalCorners", lambda m: prob_over(bundle, "TotalCorners", m, line))
    elif target.startswith("y_HomeCorners_O"):
        line = _line_from_target(target, "y_HomeCorners_")
        p_over = col("HomeCorners", lambda m: prob_over(bundle, "HomeCorners", m, line))
    elif target.startswith("y_AwayCorners_O"):
        line = _line_from_target(target, "y_AwayCorners_")
        p_over = col("AwayCorners", lambda m: prob_over(bundle, "AwayCorners", m, line))
    elif target.startswith("y_TotalYC_O"):
        line = _line_from_target(target, "y_TotalYC_")
        p_over = col("TotalYC", lambda m: prob_over(bundle, "TotalYC", m, line))
    elif target == "y_HomeTeam_Card":
        p_over = col("HomeYC", lambda m: prob_over(bundle, "HomeYC", m, 0.5))
    elif target == "y_AwayTeam_Card":
        p_over = col("AwayYC", lambda m: prob_over(bundle, "AwayYC", m, 0.5))
    elif target == "y_HomeCorners_Win":
        if "HomeCorners" not in bundle.models or "AwayCorners" not in bundle.models:
            return np.zeros((n, 2))
        p_over = np.array([
            prob_home_wins_count(bundle, mh, ma)
            for mh, ma in zip(means["HomeCorners"], means["AwayCorners"])
        ])
    else:
        return np.zeros((n, 2))

    p_over = np.clip(p_over, 0.0, 1.0)
    # Sorted class order is ['N', 'Y'] -> column 0 = No/Under, column 1 = Yes/Over
    return np.column_stack([1.0 - p_over, p_over])
