# models_goal.py
"""
GEM — Goal Expectation Model.

Trains two gradient-boosted regressors with POISSON loss to predict each
team's expected goals (lambda_home, lambda_away) from the full feature set,
then derives every goals market coherently from the Dixon-Coles score grid
(shared derive_markets_from_grid). This gives:

  * cross-market consistency by construction (1X2, all O/U lines, BTTS,
    team goals, correct score all come from one score distribution),
  * strength-borrowing for thin markets (they inherit the goal model's fit
    instead of training on sparse binary labels),
  * a model family that is genuinely different from the per-market
    classifiers, so the stacking meta-learner gets a diverse signal.

Integrated as the "gem" pseudo-base in models.py stacking — the meta-learner
decides per market how much weight it deserves.

Uses LightGBM (objective="poisson") when available, otherwise sklearn's
HistGradientBoostingRegressor(loss="poisson").
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from feature_rules import feature_exclusions
from models_dc import price_from_lambdas, _dc_corr

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
from sklearn.ensemble import HistGradientBoostingRegressor

MAX_GOALS = 8
LAMBDA_CLIP = (0.05, 6.0)

# In-process cache: one GEM fit per training window, shared across all
# markets and folds that use the same window (key = date range, day precision).
_GEM_CACHE: Dict[str, "GEMBundle"] = {}


@dataclass
class GEMBundle:
    imputer: SimpleImputer
    feat_cols: list
    model_home: object
    model_away: object
    rho: float


def _numeric_feature_cols(df: pd.DataFrame) -> list:
    excl = feature_exclusions()
    cols = []
    for c in df.columns:
        if c.startswith("y_") or c in excl:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _make_regressor(n_estimators: int = 300):
    if _HAS_LGB:
        return lgb.LGBMRegressor(
            objective="poisson",
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=n_estimators,
        learning_rate=0.05,
        min_samples_leaf=40,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
    )


def _estimate_rho(lam: np.ndarray, mu: np.ndarray, hg: np.ndarray, ag: np.ndarray) -> float:
    """Profile-likelihood estimate of the DC low-score correlation given
    fitted expected goals.

    The likelihood term per match is corr(x,y)/Z where Z(lam,mu,rho)
    renormalises the tau-adjusted grid. Z has a closed form because tau only
    touches the four low-score cells:
        Z = 1 + p00(t00-1) + p01(t01-1) + p10(t10-1) + p11(t11-1).
    Omitting Z (as a naive sum of log-tau would) biases rho to the grid
    boundary; with independent-Poisson data this estimator returns ~0.
    """
    hg_i = hg.astype(int)
    ag_i = ag.astype(int)
    if len(hg_i) < 200:
        return 0.0

    # Poisson probabilities of the four low-score cells per match
    p0h = np.exp(-lam); p1h = lam * p0h
    p0a = np.exp(-mu);  p1a = mu * p0a
    p00, p01, p10, p11 = p0h * p0a, p0h * p1a, p1h * p0a, p1h * p1a

    best_rho, best_ll = 0.0, -np.inf
    for rho in np.linspace(-0.15, 0.10, 26):
        t00 = np.maximum(1 - lam * mu * rho, 1e-6)
        t01 = np.maximum(1 + lam * rho, 1e-6)
        t10 = np.maximum(1 + mu * rho, 1e-6)
        t11 = np.maximum(1 - rho, 1e-6)
        Z = 1 + p00*(t00-1) + p01*(t01-1) + p10*(t10-1) + p11*(t11-1)

        tau = np.ones(len(hg_i))
        tau = np.where((hg_i == 0) & (ag_i == 0), t00, tau)
        tau = np.where((hg_i == 0) & (ag_i == 1), t01, tau)
        tau = np.where((hg_i == 1) & (ag_i == 0), t10, tau)
        tau = np.where((hg_i == 1) & (ag_i == 1), t11, tau)

        ll = float(np.sum(np.log(np.maximum(tau, 1e-12)) - np.log(np.maximum(Z, 1e-12))))
        if ll > best_ll:
            best_ll, best_rho = ll, float(rho)
    return best_rho


def _cache_key(train_df: pd.DataFrame) -> str:
    d = pd.to_datetime(train_df["Date"])
    raw = f"gem|{str(d.min())[:10]}|{str(d.max())[:10]}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def fit_gem(train_df: pd.DataFrame, use_cache: bool = True) -> Optional[GEMBundle]:
    """Fit lambda_home / lambda_away regressors on completed matches."""
    sub = train_df.dropna(subset=["FTHG", "FTAG"]).copy()
    if len(sub) < 500:
        return None

    key = _cache_key(sub)
    if use_cache and key in _GEM_CACHE:
        return _GEM_CACHE[key]

    feat_cols = _numeric_feature_cols(sub)
    if len(feat_cols) < 5:
        return None

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(sub[feat_cols])
    y_home = pd.to_numeric(sub["FTHG"], errors="coerce").to_numpy(dtype=float)
    y_away = pd.to_numeric(sub["FTAG"], errors="coerce").to_numpy(dtype=float)

    model_home = _make_regressor()
    model_away = _make_regressor()
    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

    lam = np.clip(model_home.predict(X), *LAMBDA_CLIP)
    mu = np.clip(model_away.predict(X), *LAMBDA_CLIP)
    rho = _estimate_rho(lam, mu, y_home, y_away)

    bundle = GEMBundle(imputer=imputer, feat_cols=feat_cols,
                       model_home=model_home, model_away=model_away, rho=rho)
    if use_cache:
        _GEM_CACHE[key] = bundle
    return bundle


def predict_lambdas(bundle: GEMBundle, rows_df: pd.DataFrame) -> tuple:
    """Predict (lambda_home, lambda_away) arrays for fixture rows."""
    X = pd.DataFrame(index=rows_df.index)
    for c in bundle.feat_cols:
        if c in rows_df.columns:
            X[c] = pd.to_numeric(rows_df[c], errors="coerce")
        else:
            X[c] = np.nan
    Xi = bundle.imputer.transform(X[bundle.feat_cols])
    lam = np.clip(bundle.model_home.predict(Xi), *LAMBDA_CLIP)
    mu = np.clip(bundle.model_away.predict(Xi), *LAMBDA_CLIP)
    return lam, mu


def gem_prices_for_rows(train_df: pd.DataFrame, rows_df: pd.DataFrame,
                        max_goals: int = MAX_GOALS) -> list:
    """Return a list of DC_*-keyed market dicts, one per fixture row
    (empty dict when the model could not be fitted)."""
    bundle = fit_gem(train_df)
    if bundle is None:
        return [{} for _ in range(len(rows_df))]
    lam, mu = predict_lambdas(bundle, rows_df)
    # price_from_lambdas memoises on rounded (lam, mu) — read-only dicts
    return [price_from_lambdas(float(l), float(m), bundle.rho, max_goals)
            for l, m in zip(lam, mu)]
