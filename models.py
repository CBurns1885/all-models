# models.py
# End-to-end model training, stacking, calibration, and prediction for all markets.
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import copy

import numpy as np
import pandas as pd
try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# Specialized models for different market types
try:
    from model_binary import BinaryMarketModel, is_binary_market
    from model_multiclass import MulticlassMarketModel, is_multiclass_market
    from model_ordinal import OrdinalMarketModel, is_ordinal_market
    _HAS_SPECIALIZED = True
except ImportError:
    _HAS_SPECIALIZED = False
    # Silently fall back to standard models (no warning needed)


# Optional GBMs
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

import joblib

from config import (
    DATA_DIR, OUTPUT_DIR, MODEL_ARTIFACTS_DIR, FEATURES_PARQUET, RANDOM_SEED, log_header
)
from progress_utils import Timer, heartbeat
from tuning import make_time_split, objective_factory, CVData
from ordinal import CORALOrdinal
from calibration import (
    DirichletCalibrator, TemperatureScaler, IsotonicOrdinalCalibrator,
    BetaCalibrator, get_calibrator_for_market
)
from models_dc import fit_all as dc_fit_all, price_match as dc_price_match

# Import market configuration for optimal model selection
try:
    from market_config import (
        get_market_config, get_market_type, get_model_strategy,
        get_base_models_for_market, get_model_params_for_market,
        should_use_dc_blend, should_use_isotonic,
        MarketType, ModelStrategy
    )
    _HAS_MARKET_CONFIG = True
except ImportError:
    _HAS_MARKET_CONFIG = False

# Import speed configuration for performance optimization
try:
    from speed_config import (
        get_speed_mode, get_speed_config, get_pretuned_params,
        should_train_market, get_models_for_mode, get_n_folds,
        get_n_estimators, use_tuning, use_specialized_models,
        print_speed_info, SpeedMode
    )
    _HAS_SPEED_CONFIG = True
except ImportError:
    _HAS_SPEED_CONFIG = False

# Global DC cache to avoid refitting
_DC_PARAMS_CACHE = {}


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def _load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PARQUET)
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["League", "Date"]).reset_index(drop=True)

# --------------------------------------------------------------------------------------
# Targets definition
# --------------------------------------------------------------------------------------
OU_LINES = ["0_5","1_5","2_5","3_5","4_5","5_5"]
AH_LINES = ["-2_0","-1_5","-1_0","-0_5","0_0","+0_5","+1_0","+1_5","+2_0"]
TEAM_GOAL_LINES = ["0_5","1_5","2_5","3_5"]

def _all_targets() -> List[str]:
    """Define all betting market targets for training - COMPREHENSIVE"""
    t = [
        # Core Markets
        "y_1X2",
        "y_BTTS",
        "y_GOAL_RANGE",
        "y_CS",

        # Over/Under Total Goals
        *(f"y_OU_{l}" for l in OU_LINES),

        # Exact Total Goals
        "y_ExactTotal_0", "y_ExactTotal_1", "y_ExactTotal_2",
        "y_ExactTotal_3", "y_ExactTotal_4", "y_ExactTotal_5", "y_ExactTotal_6+",

        # Draw No Bet
        "y_DNB_H", "y_DNB_A",

        # To Score
        "y_HomeToScore", "y_AwayToScore",

        # Half-time Markets
        "y_HT",
        "y_HTFT",
        "y_HT_OU_0_5",
        "y_HT_OU_1_5",
        "y_HT_OU_2_5",
        "y_HT_BTTS",

        # Second Half Markets
        "y_2H_OU_0_5",
        "y_2H_OU_1_5",
        "y_2H_OU_2_5",
        "y_2H_BTTS",

        # Half Comparison
        "y_HigherHalf",
        "y_GoalsBothHalves",
        "y_HomeScoresBothHalves",
        "y_AwayScoresBothHalves",

        # Win Half Markets
        "y_HomeWinEitherHalf",
        "y_AwayWinEitherHalf",
        "y_HomeWinBothHalves",
        "y_AwayWinBothHalves",

        # First to Score
        "y_FirstToScore",

        # Team Goals Over/Under
        *(f"y_HomeTG_{l}" for l in TEAM_GOAL_LINES),
        *(f"y_AwayTG_{l}" for l in TEAM_GOAL_LINES),

        # Exact Team Goals
        "y_HomeExact_0", "y_HomeExact_1", "y_HomeExact_2", "y_HomeExact_3+",
        "y_AwayExact_0", "y_AwayExact_1", "y_AwayExact_2", "y_AwayExact_3+",

        # Asian Handicap (Extended)
        *(f"y_AH_{l}" for l in AH_LINES),

        # European Handicap
        "y_EH_m1_H", "y_EH_m1_D", "y_EH_m1_A",
        "y_EH_m2_H", "y_EH_m2_D", "y_EH_m2_A",
        "y_EH_p1_H", "y_EH_p1_D", "y_EH_p1_A",
        "y_EH_p2_H", "y_EH_p2_D", "y_EH_p2_A",

        # Double Chance
        "y_DC_1X",
        "y_DC_X2",
        "y_DC_12",

        # Win to Nil / Clean Sheets
        "y_HomeWTN",
        "y_AwayWTN",
        "y_HomeCS",
        "y_AwayCS",
        "y_NoGoal",

        # Win by Margin
        "y_HomeWinBy1", "y_HomeWinBy2", "y_HomeWinBy3+",
        "y_AwayWinBy1", "y_AwayWinBy2", "y_AwayWinBy3+",
        "y_HomeWin2+",
        "y_AwayWin2+",

        # Odd/Even
        "y_TotalOddEven",
        "y_HomeOddEven",
        "y_AwayOddEven",

        # Multi-Goal
        "y_Match2+Goals",
        "y_Match3+Goals",
        "y_Match4+Goals",
        "y_Match5+Goals",

        # Result & BTTS Combos
        "y_HomeWin_BTTS_Y", "y_HomeWin_BTTS_N",
        "y_AwayWin_BTTS_Y", "y_AwayWin_BTTS_N",
        "y_Draw_BTTS_Y", "y_Draw_BTTS_N",

        # Result & O/U 2.5 Combos
        "y_HomeWin_O25", "y_HomeWin_U25",
        "y_AwayWin_O25", "y_AwayWin_U25",
        "y_Draw_O25", "y_Draw_U25",

        # Double Chance + O/U Combos
        "y_DC1X_O25", "y_DC1X_U25",
        "y_DCX2_O25", "y_DCX2_U25",
        "y_DC12_O25", "y_DC12_U25",

        # Double Chance + BTTS Combos
        "y_DC1X_BTTS_Y", "y_DC1X_BTTS_N",
        "y_DCX2_BTTS_Y", "y_DCX2_BTTS_N",
    ]
    return t

# banded/ordered targets -> use ordinal model
ORDINAL_TARGETS = {
    "y_GOAL_RANGE": ["0","1","2","3","4","5+"],
    "y_HomeCardsY_BAND": ["0-2","3","4-5","6+"],
    "y_AwayCardsY_BAND": ["0-2","3","4-5","6+"],
    "y_HomeCorners_BAND": ["0-3","4-5","6-7","8-9","10+"],
    "y_AwayCorners_BAND": ["0-3","4-5","6-7","8-9","10+"],
}

# targets where DC can produce probabilities
# Extended to support more market types that can be derived from score probabilities
def _dc_supported(t: str) -> bool:
    """Check if Dixon-Coles can provide probability estimates for this target."""
    # Core DC-supported markets
    core_markets = ["y_1X2", "y_BTTS", "y_GOAL_RANGE", "y_CS"]

    # Markets derivable from score grid
    derivable_prefixes = [
        "y_OU_",      # Over/Under total goals
        "y_AH_",      # Asian Handicap
        "y_HomeTG_",  # Home team goals O/U
        "y_AwayTG_",  # Away team goals O/U
        "y_DC_",      # Double Chance (derivable from 1X2)
        "y_DNB_",     # Draw No Bet (derivable from 1X2)
    ]

    # Exact goal markets (derivable from score grid)
    exact_markets = [
        "y_ExactTotal_0", "y_ExactTotal_1", "y_ExactTotal_2",
        "y_ExactTotal_3", "y_ExactTotal_4", "y_ExactTotal_5", "y_ExactTotal_6+",
        "y_HomeExact_0", "y_HomeExact_1", "y_HomeExact_2", "y_HomeExact_3+",
        "y_AwayExact_0", "y_AwayExact_1", "y_AwayExact_2", "y_AwayExact_3+",
    ]

    # To Score markets (can derive from team goal probs)
    to_score_markets = ["y_HomeToScore", "y_AwayToScore"]

    # Clean sheet / WTN markets (derivable)
    cs_markets = ["y_HomeWTN", "y_AwayWTN", "y_HomeCS", "y_AwayCS", "y_NoGoal"]

    # Multi-goal markets (derivable from total goals distribution)
    multigoal_markets = ["y_Match2+Goals", "y_Match3+Goals", "y_Match4+Goals", "y_Match5+Goals"]

    return (
        t in core_markets
        or t in exact_markets
        or t in to_score_markets
        or t in cs_markets
        or t in multigoal_markets
        or any(t.startswith(p) for p in derivable_prefixes)
    )


# --------------------------------------------------------------------------------------
# Preprocess
# --------------------------------------------------------------------------------------
def _feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    id_cols = {"League","Date","HomeTeam","AwayTeam"}
    target_cols = set([c for c in df.columns if c.startswith("y_")])
    result_cols = {"FTHG", "FTAG", "FTR"}  # Add this line
    cand = [c for c in df.columns if c not in id_cols and c not in target_cols and c not in result_cols]  # Add result_cols here
    cat = [c for c in cand if str(df[c].dtype) in ("object","string","category","bool")]
    num = [c for c in cand if c not in cat]
    return num, cat


def _preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = _feature_columns(df)
    num_trf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_trf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", num_trf, num_cols),
        ("cat", cat_trf, cat_cols),
    ])


# Bayesian Neural Network (optional - disabled due to PyTorch)
try:
    import torch
    import torch.nn as nn
    class SmallBNN(nn.Module):
        def __init__(self, input_dim, output_dim, dropout=0.2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.drop1 = nn.Dropout(p=dropout)
            self.fc2 = nn.Linear(64, 32)
            self.drop2 = nn.Dropout(p=dropout)
            self.fc3 = nn.Linear(32, output_dim)
        def forward(self, x):
            x = self.drop1(torch.relu(self.fc1(x)))
            x = self.drop2(torch.relu(self.fc2(x)))
            return self.fc3(x)
except Exception:
    SmallBNN = None

class BNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes: int, epochs=20, lr=1e-3, dropout=0.2, mc=20, seed=42):
        self.n_classes = n_classes
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.mc = mc
        self.seed = seed
        self.model = None
        self.in_dim = None

    def fit(self, X, y):
        if not _HAS_TORCH:
            raise RuntimeError("Torch not available")
        torch.manual_seed(self.seed)
        self.in_dim = X.shape[1]
        self.model = SmallBNN(self.in_dim, self.n_classes, dropout=self.dropout)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = self.model(X_t)
            loss = crit(logits, y_t)
            loss.backward()
            opt.step()
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Unfitted BNN")
        self.model.train()  # MC dropout
        X_t = torch.tensor(X, dtype=torch.float32)
        outs = []
        for _ in range(self.mc):
            with torch.no_grad():
                logits = self.model(X_t).cpu().numpy()
                probs = _softmax_np(logits)
                outs.append(probs)
        return np.mean(np.stack(outs, axis=0), axis=0)

def _softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# --------------------------------------------------------------------------------------
# TrainedTarget dataclass
# --------------------------------------------------------------------------------------
@dataclass
class TrainedTarget:
    target: str
    classes_: List[str]
    preprocessor: ColumnTransformer
    base_models: Dict[str, object]
    meta: Optional[object]
    calibrator: Optional[object]
    oof_pred: np.ndarray
    oof_y: np.ndarray
    feature_names: List[str]


# --------------------------------------------------------------------------------------
# Build base model zoo
# --------------------------------------------------------------------------------------
def _build_base_model(name: str, n_classes: int, feature_names: List[str], market_type: str = "binary"):
    """Build a base model with pre-tuned or default parameters.

    Args:
        name: Model name (rf, et, xgb, lgb, cat, lr, bnn)
        n_classes: Number of classes
        feature_names: Feature names (for compatibility)
        market_type: "binary", "multiclass", or "ordinal"
    """
    # Get n_estimators from speed config or environment
    if _HAS_SPEED_CONFIG:
        n_est = get_n_estimators()
    else:
        n_est = int(os.environ.get("N_ESTIMATORS", "300"))

    # Get pre-tuned parameters if available
    if _HAS_SPEED_CONFIG:
        params = get_pretuned_params(name, market_type)
    else:
        params = {}

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", n_est),
            max_depth=params.get("max_depth", 12),
            min_samples_split=params.get("min_samples_split", 5),
            min_samples_leaf=params.get("min_samples_leaf", 10),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced_subsample"),
            n_jobs=-1,
            random_state=RANDOM_SEED
        )
    if name == "et":
        return ExtraTreesClassifier(
            n_estimators=params.get("n_estimators", n_est),
            max_depth=params.get("max_depth", 12),
            min_samples_split=params.get("min_samples_split", 4),
            min_samples_leaf=params.get("min_samples_leaf", 8),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced"),
            n_jobs=-1,
            random_state=RANDOM_SEED
        )
    if name == "lr":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            n_jobs=-1,
            class_weight="balanced"
        )
    if name == "xgb" and _HAS_XGB:
        return xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", n_est),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            tree_method="hist",
            random_state=RANDOM_SEED,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            n_jobs=-1
        )
    if name == "lgb" and _HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", n_est),
            num_leaves=params.get("num_leaves", 31),
            learning_rate=params.get("learning_rate", 0.05),
            min_child_samples=params.get("min_child_samples", 20),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 0.1),
            objective="multiclass" if n_classes > 2 else "binary",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1
        )
    if name == "cat" and _HAS_CAT:
        return CatBoostClassifier(
            iterations=params.get("iterations", n_est),
            depth=params.get("depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            verbose=False,
            random_state=RANDOM_SEED
        )
    if name == "bnn" and _HAS_TORCH:
        epochs = 10 if n_est < 200 else 25
        return BNNWrapper(n_classes=n_classes, epochs=epochs, lr=1e-3, dropout=0.2, mc=20, seed=RANDOM_SEED)
    raise RuntimeError(f"Unknown / unavailable base model: {name}")

# --------------------------------------------------------------------------------------
# Optuna tuning wrapper with market-specific trial counts
# --------------------------------------------------------------------------------------

# Market-specific trial counts for optimal speed/accuracy balance
# Base counts - will be scaled adaptively by _adaptive_trials()
TRIALS_BY_MARKET_TYPE = {
    'binary': {'rf': 15, 'et': 15, 'xgb': 20, 'lgb': 20, 'cat': 20, 'lr': 5},
    'multiclass': {'rf': 20, 'et': 20, 'xgb': 30, 'lgb': 30, 'cat': 30, 'lr': 10},
    'ordinal': {'coral': 10, 'rf': 15, 'et': 15, 'xgb': 20, 'lgb': 20, 'lr': 5}
}

def _adaptive_trials(base_trials: int, n_samples: int, n_classes: int) -> int:
    """
    Scale Optuna trial count adaptively based on dataset characteristics.

    - Small datasets (<500 rows): fewer trials (risk of overfitting the tuning)
    - Large datasets (>5000 rows): more trials (can afford richer search)
    - Many classes: more trials (harder optimisation landscape)
    - Returns at least 5 and at most 3x the base count.
    """
    if base_trials == 0:
        return 0
    scale = 1.0
    # Scale down for small data, up for large
    if n_samples < 300:
        scale *= 0.4
    elif n_samples < 500:
        scale *= 0.6
    elif n_samples < 1000:
        scale *= 0.8
    elif n_samples > 5000:
        scale *= 1.3
    elif n_samples > 10000:
        scale *= 1.5
    # More classes = harder problem = more trials
    if n_classes > 10:
        scale *= 1.3
    elif n_classes > 5:
        scale *= 1.1
    elif n_classes == 2:
        scale *= 0.9
    result = int(base_trials * scale)
    return max(5, min(result, base_trials * 3))

def _get_market_type(target_col: str) -> str:
    """Determine market type for trial optimization using market configuration."""
    if _HAS_MARKET_CONFIG:
        market_type = get_market_type(target_col)
        if market_type == MarketType.ORDINAL:
            return 'ordinal'
        elif market_type in [MarketType.MULTICLASS, MarketType.TERNARY]:
            return 'multiclass'
        elif market_type == MarketType.SPARSE:
            return 'multiclass'  # Treat sparse as multiclass for trial allocation
        else:
            return 'binary'
    else:
        # Legacy fallback
        if target_col in ORDINAL_TARGETS:
            return 'ordinal'
        elif any(target_col.startswith(p) for p in ['y_OU_', 'y_BTTS', 'y_AH_']):
            return 'binary'
        elif any(target_col == p for p in ['y_1X2', 'y_HT', 'y_HTFT', 'y_GOAL_RANGE']):
            return 'multiclass'
        else:
            return 'binary'

def _tune_model(alg: str, X: np.ndarray, y: np.ndarray, classes_: np.ndarray, target_col: str = None) -> object:
    """Tune a model with Optuna or return pre-tuned model."""
    market_type = _get_market_type(target_col) if target_col else "binary"

    # Check if we should skip tuning (speed mode)
    if _HAS_SPEED_CONFIG and not use_tuning():
        # Return pre-tuned model directly - no Optuna
        return _build_base_model(alg, len(classes_), [], market_type)

    # Determine trials: adaptive scaling based on data size and class count
    if target_col and os.environ.get("USE_MARKET_SPECIFIC_TRIALS", "1") == "1":
        base_trials = TRIALS_BY_MARKET_TYPE.get(market_type, {}).get(alg, 25)
        n_trials = _adaptive_trials(base_trials, len(X), len(classes_))
        print(f"  [TUNE] {target_col} ({market_type}): {alg} {n_trials} trials "
              f"(base={base_trials}, n={len(X)}, K={len(classes_)})")
    else:
        n_trials = int(os.environ.get("OPTUNA_TRIALS", "5"))

    if n_trials == 0:
        # Skip tuning, return default models
        return _build_base_model(alg, len(classes_), [])
    
    # FIX: Create consistent label encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    # Ensure y values are in range [0, len(classes_)-1]
    y_consistent = np.clip(y, 0, len(classes_) - 1)
    
    ps = make_time_split(len(y_consistent), n_folds=3)  # Reduce folds for speed
    
    # Import the fixed CVData and objective_factory
    from tuning import CVData, objective_factory
    
    cvd = CVData(
        X=X, 
        y=y_consistent, 
        ps=ps, 
        classes_=classes_,
        label_encoder=le
    )
    
    try:
        objective = objective_factory(alg, cvd)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
    except Exception as e:
        print(f"Warning: Optuna tuning failed for {alg}: {e}")
        # Fall back to default model
        return _build_base_model(alg, len(classes_), [])
    
    # Build final model with best params (same logic as before but with error handling)
    try:
        if alg == "rf":
            model = RandomForestClassifier(
                n_estimators=best_params.get("n_estimators", 600),
                max_depth=best_params.get("max_depth", None),
                min_samples_split=best_params.get("min_samples_split", 2),
                min_samples_leaf=best_params.get("min_samples_leaf", 1),
                max_features=best_params.get("max_features", "sqrt"),
                class_weight="balanced_subsample",
                n_jobs=-1, random_state=RANDOM_SEED
            )
        elif alg == "et":
            model = ExtraTreesClassifier(
                n_estimators=best_params.get("n_estimators", 800),
                max_depth=best_params.get("max_depth", None),
                min_samples_split=best_params.get("min_samples_split", 2),
                min_samples_leaf=best_params.get("min_samples_leaf", 1),
                max_features=best_params.get("max_features", "sqrt"),
                class_weight="balanced",
                n_jobs=-1, random_state=RANDOM_SEED
            )
        elif alg == "xgb" and _HAS_XGB:
            try:
                params = dict(
                    n_estimators=best_params.get("n_estimators", 800),
                    max_depth=best_params.get("max_depth", 6),
                    learning_rate=best_params.get("learning_rate", 0.05),
                    subsample=best_params.get("subsample", 0.9),
                    colsample_bytree=best_params.get("colsample_bytree", 0.9),
                    reg_alpha=best_params.get("reg_alpha", 1e-4),
                    reg_lambda=best_params.get("reg_lambda", 1.0),
                    tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED,
                    objective="multi:softprob" if len(classes_)>2 else "binary:logistic",
                )
                model = xgb.XGBClassifier(**params)
            except Exception as e:
                print(f"[WARN]  XGBoost creation failed: {e}. Skipping XGBoost for this target.")
                return None
        elif alg == "lgb" and _HAS_LGB:
            params = dict(
                n_estimators=best_params.get("n_estimators", 1000),
                num_leaves=best_params.get("num_leaves", 64),
                learning_rate=best_params.get("learning_rate", 0.05),
                subsample=best_params.get("subsample", 0.9),
                colsample_bytree=best_params.get("colsample_bytree", 0.9),
                min_child_samples=best_params.get("min_child_samples", 25),
                objective="multiclass" if len(classes_)>2 else "binary",
                random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
            )
            model = lgb.LGBMClassifier(**params)
        elif alg == "cat" and _HAS_CAT:
            model = CatBoostClassifier(
                iterations=best_params.get("iterations", 1200),
                depth=best_params.get("depth", 6),
                learning_rate=best_params.get("learning_rate", 0.05),
                l2_leaf_reg=best_params.get("l2_leaf_reg", 3.0),
                loss_function="MultiClass" if len(classes_)>2 else "Logloss",
                verbose=False, random_state=RANDOM_SEED
            )
        elif alg == "lr":
            C = best_params.get("C", 1.0)
            model = LogisticRegression(
                C=C, max_iter=2000, n_jobs=-1, class_weight="balanced",
            )
        else:
            raise RuntimeError(f"Tuning not supported for {alg}")

        try:
            # Use early stopping for gradient boosting models
            if alg in ("xgb", "lgb", "cat") and len(X) > 500:
                from sklearn.model_selection import train_test_split
                X_tr, X_es, y_tr, y_es = train_test_split(
                    X, y_consistent, test_size=0.15, random_state=RANDOM_SEED, stratify=y_consistent
                )
                if alg == "xgb" and _HAS_XGB:
                    model.set_params(early_stopping_rounds=30)
                    model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
                elif alg == "lgb" and _HAS_LGB:
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_es, y_es)],
                        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
                    )
                elif alg == "cat" and _HAS_CAT:
                    model.fit(X_tr, y_tr, eval_set=(X_es, y_es), early_stopping_rounds=30)
            else:
                model.fit(X, y_consistent)
            return model
        except Exception as e:
            # If gradient boosting fit fails with early stopping, retry without
            if alg in ("xgb", "lgb", "cat"):
                try:
                    model.fit(X, y_consistent)
                    return model
                except Exception as e2:
                    print(f"[WARN]  {alg} fit failed: {e2}. Skipping for this target.")
                    return None
            else:
                # Re-raise for non-boosting failures
                raise
        
    except Exception as e:
        print(f"Warning: Model building failed for {alg}: {e}")
        # Fall back to default model
        return _build_base_model(alg, len(classes_), [])


# --------------------------------------------------------------------------------------
# DC probabilities helper (for OOF & inference)
# --------------------------------------------------------------------------------------
def _dc_probs_for_rows(train_df: pd.DataFrame, rows_df: pd.DataFrame, target: str, max_goals=8, use_cache=True) -> np.ndarray:
    """
    Generate Dixon-Coles probability estimates for various market types.

    Extended to support additional markets derivable from the score probability grid.

    Args:
        use_cache: If True, cache DC params globally to avoid refitting
    """
    global _DC_PARAMS_CACHE

    # Use cached params if available and caching enabled
    # Cache key uses data size + date range hash for collision resistance
    date_hash = hash((len(train_df), str(train_df['Date'].min()), str(train_df['Date'].max())))
    cache_key = date_hash
    if use_cache and cache_key in _DC_PARAMS_CACHE:
        params = _DC_PARAMS_CACHE[cache_key]
    else:
        params = dc_fit_all(train_df[["League","Date","HomeTeam","AwayTeam","FTHG","FTAG"]])
        if use_cache:
            _DC_PARAMS_CACHE[cache_key] = params
    out = []

    for _, r in rows_df[["League","HomeTeam","AwayTeam"]].iterrows():
        lg, ht, at = r["League"], r["HomeTeam"], r["AwayTeam"]
        mp = {}
        if lg in params:
            mp = dc_price_match(params[lg], ht, at, max_goals=max_goals)

        vec = None

        # Core markets
        if target == "y_1X2":
            vec = [mp.get("DC_1X2_H", 0.0), mp.get("DC_1X2_D", 0.0), mp.get("DC_1X2_A", 0.0)]

        elif target == "y_BTTS":
            vec = [mp.get("DC_BTTS_N", 0.0), mp.get("DC_BTTS_Y", 0.0)]

        elif target == "y_GOAL_RANGE":
            labs = ["0", "1", "2", "3", "4", "5+"]
            vec = [mp.get(f"DC_GR_{k}", 0.0) for k in labs]

        elif target == "y_CS":
            vec = [mp.get(f"DC_CS_{a}_{b}", 0.0) for a in range(6) for b in range(6)] + [mp.get("DC_CS_Other", 0.0)]

        # Over/Under total goals
        elif target.startswith("y_OU_"):
            l = target.split("_", 2)[2]
            vec = [mp.get(f"DC_OU_{l}_U", 0.0), mp.get(f"DC_OU_{l}_O", 0.0)]

        # Asian Handicap
        elif target.startswith("y_AH_"):
            l = target.split("_", 2)[2]
            vec = [mp.get(f"DC_AH_{l}_A", 0.0), mp.get(f"DC_AH_{l}_P", 0.0), mp.get(f"DC_AH_{l}_H", 0.0)]

        # Team Goals Over/Under - derive from score grid
        elif target.startswith("y_HomeTG_"):
            line = float(target.split("_", 2)[2].replace("_", "."))
            # Calculate P(HomeGoals > line) from score grid
            p_over = 0.0
            for h in range(max_goals + 1):
                if h > line:
                    for a in range(max_goals + 1):
                        p_over += mp.get(f"DC_CS_{h}_{a}", 0.0)
            vec = [1.0 - p_over, p_over]  # [Under, Over]

        elif target.startswith("y_AwayTG_"):
            line = float(target.split("_", 2)[2].replace("_", "."))
            p_over = 0.0
            for a in range(max_goals + 1):
                if a > line:
                    for h in range(max_goals + 1):
                        p_over += mp.get(f"DC_CS_{h}_{a}", 0.0)
            vec = [1.0 - p_over, p_over]

        # Double Chance - derive from 1X2
        elif target == "y_DC_1X":
            p_h = mp.get("DC_1X2_H", 0.0)
            p_d = mp.get("DC_1X2_D", 0.0)
            vec = [1.0 - (p_h + p_d), p_h + p_d]  # [No, Yes]

        elif target == "y_DC_X2":
            p_d = mp.get("DC_1X2_D", 0.0)
            p_a = mp.get("DC_1X2_A", 0.0)
            vec = [1.0 - (p_d + p_a), p_d + p_a]

        elif target == "y_DC_12":
            p_h = mp.get("DC_1X2_H", 0.0)
            p_a = mp.get("DC_1X2_A", 0.0)
            vec = [1.0 - (p_h + p_a), p_h + p_a]

        # Draw No Bet - derive from 1X2 (excluding draws)
        elif target == "y_DNB_H":
            p_h = mp.get("DC_1X2_H", 0.0)
            p_a = mp.get("DC_1X2_A", 0.0)
            total = p_h + p_a
            if total > 0:
                vec = [p_a / total, p_h / total]  # [No (Away wins), Yes (Home wins)]
            else:
                vec = [0.5, 0.5]

        elif target == "y_DNB_A":
            p_h = mp.get("DC_1X2_H", 0.0)
            p_a = mp.get("DC_1X2_A", 0.0)
            total = p_h + p_a
            if total > 0:
                vec = [p_h / total, p_a / total]
            else:
                vec = [0.5, 0.5]

        # Exact Total Goals - derive from goal range
        elif target.startswith("y_ExactTotal_"):
            n = target.split("_")[-1]
            if n == "6+":
                p_exact = mp.get("DC_GR_5+", 0.0)  # 5+ includes 6+
            else:
                p_exact = mp.get(f"DC_GR_{n}", 0.0)
            vec = [1.0 - p_exact, p_exact]

        # Exact Team Goals - derive from score grid
        elif target.startswith("y_HomeExact_"):
            n = target.split("_")[-1]
            p_exact = 0.0
            if n == "3+":
                for h in range(3, max_goals + 1):
                    for a in range(max_goals + 1):
                        p_exact += mp.get(f"DC_CS_{h}_{a}", 0.0)
            else:
                h = int(n)
                for a in range(max_goals + 1):
                    p_exact += mp.get(f"DC_CS_{h}_{a}", 0.0)
            vec = [1.0 - p_exact, p_exact]

        elif target.startswith("y_AwayExact_"):
            n = target.split("_")[-1]
            p_exact = 0.0
            if n == "3+":
                for a in range(3, max_goals + 1):
                    for h in range(max_goals + 1):
                        p_exact += mp.get(f"DC_CS_{h}_{a}", 0.0)
            else:
                a = int(n)
                for h in range(max_goals + 1):
                    p_exact += mp.get(f"DC_CS_{h}_{a}", 0.0)
            vec = [1.0 - p_exact, p_exact]

        # To Score markets
        elif target == "y_HomeToScore":
            # P(Home scores at least 1) = 1 - P(Home scores 0)
            p_zero = 0.0
            for a in range(max_goals + 1):
                p_zero += mp.get(f"DC_CS_0_{a}", 0.0)
            vec = [p_zero, 1.0 - p_zero]  # [No, Yes]

        elif target == "y_AwayToScore":
            p_zero = 0.0
            for h in range(max_goals + 1):
                p_zero += mp.get(f"DC_CS_{h}_0", 0.0)
            vec = [p_zero, 1.0 - p_zero]

        # Clean Sheet / Win to Nil markets
        elif target == "y_HomeCS":
            # P(Away scores 0)
            p_cs = 0.0
            for h in range(max_goals + 1):
                p_cs += mp.get(f"DC_CS_{h}_0", 0.0)
            vec = [1.0 - p_cs, p_cs]

        elif target == "y_AwayCS":
            p_cs = 0.0
            for a in range(max_goals + 1):
                p_cs += mp.get(f"DC_CS_0_{a}", 0.0)
            vec = [1.0 - p_cs, p_cs]

        elif target == "y_HomeWTN":
            # P(Home wins AND Away scores 0)
            p_wtn = 0.0
            for h in range(1, max_goals + 1):
                p_wtn += mp.get(f"DC_CS_{h}_0", 0.0)
            vec = [1.0 - p_wtn, p_wtn]

        elif target == "y_AwayWTN":
            p_wtn = 0.0
            for a in range(1, max_goals + 1):
                p_wtn += mp.get(f"DC_CS_0_{a}", 0.0)
            vec = [1.0 - p_wtn, p_wtn]

        elif target == "y_NoGoal":
            p_00 = mp.get("DC_CS_0_0", 0.0)
            vec = [1.0 - p_00, p_00]

        # Multi-goal markets
        elif target == "y_Match2+Goals":
            p_over = mp.get("DC_OU_1_5_O", 0.0)  # 2+ goals = Over 1.5
            vec = [1.0 - p_over, p_over]

        elif target == "y_Match3+Goals":
            p_over = mp.get("DC_OU_2_5_O", 0.0)
            vec = [1.0 - p_over, p_over]

        elif target == "y_Match4+Goals":
            p_over = mp.get("DC_OU_3_5_O", 0.0)
            vec = [1.0 - p_over, p_over]

        elif target == "y_Match5+Goals":
            p_over = mp.get("DC_OU_4_5_O", 0.0)
            vec = [1.0 - p_over, p_over]

        out.append(vec)

    first = next((v for v in out if v is not None), None)
    if first is None:
        return np.zeros((len(rows_df), 1))

    W = len(first)
    arr = np.zeros((len(rows_df), W))
    for i, v in enumerate(out):
        if v is not None:
            arr[i, :] = v

    # Renormalize for safety
    s = arr.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return arr / s


def _select_meta_learner(oof_pred: np.ndarray, y_int: np.ndarray,
                         n_classes: int, n_folds: int = 5):
    """
    Select the best meta-learner for stacking by comparing OOF log loss
    across candidates: LogisticRegression, Ridge, ElasticNet.
    Falls back to LogisticRegression if alternatives fail.
    """
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.model_selection import cross_val_score

    candidates = {
        'lr': LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1),
        'lr_l1': LogisticRegression(max_iter=2000, C=1.0, penalty='l1',
                                     solver='saga', n_jobs=-1),
        'ridge': RidgeClassifier(alpha=1.0),
    }

    best_name = 'lr'
    best_score = float('inf')
    best_model = candidates['lr']

    ps_meta = make_time_split(len(y_int), n_folds=min(n_folds, 3))

    for name, model in candidates.items():
        try:
            # Use negative log loss for scoring (higher = better, so we negate)
            if name == 'ridge':
                # RidgeClassifier doesn't support predict_proba natively;
                # wrap in CalibratedClassifierCV for probability output
                from sklearn.calibration import CalibratedClassifierCV
                model = CalibratedClassifierCV(model, cv=3, method='sigmoid')

            scores = cross_val_score(
                model, oof_pred, y_int, cv=ps_meta,
                scoring='neg_log_loss', error_score=-10.0
            )
            mean_ll = -scores.mean()  # Convert back to positive log loss

            if mean_ll < best_score:
                best_score = mean_ll
                best_name = name
                best_model = model
        except Exception:
            continue

    # Refit best on all data
    best_model.fit(oof_pred, y_int)
    print(f"  [META] Selected {best_name} meta-learner (OOF logloss={best_score:.4f})")
    return best_model


# --------------------------------------------------------------------------------------
# Single target training (OOF, stacking, calibration)
# --------------------------------------------------------------------------------------
def _fit_single_target(df: pd.DataFrame, target_col: str) -> TrainedTarget:
    sub = df.dropna(subset=[target_col]).copy()
    if sub.empty:
        raise RuntimeError(f"No data for target {target_col}")
    
    error_count = 0
    
    y = sub[target_col].astype("category")
    classes = list(y.cat.categories)
    y_int = y.cat.codes.values
    
    # Validate class distribution
    class_counts = pd.Series(y_int).value_counts()
    min_class_count = class_counts.min()
    
    if min_class_count < 5:
        print(f"[WARN] Skipping {target_col} - class has only {min_class_count} sample(s), need minimum 5 for CV")
        return None
    
    if len(classes) > 50:
        print(f"[WARN] Skipping {target_col} - too many classes ({len(classes)}), max 50 supported")
        return None
    pre = _preprocessor(sub)
    X_all = pre.fit_transform(sub)
    feature_names = [*(pre.transformers_[0][2] or []), *(pre.transformers_[1][2] or [])]

    # ===== SPEED-AWARE MODEL SELECTION =====
    # Use speed config to determine models, with market config as secondary

    # Determine if we should use specialized models
    if _HAS_SPEED_CONFIG:
        use_specialized = _HAS_SPECIALIZED and use_specialized_models()
    else:
        use_specialized = _HAS_SPECIALIZED and os.environ.get("USE_SPECIALIZED", "1") == "1"

    # Get market type for parameter selection
    market_type_str = _get_market_type(target_col)

    # Speed-based model selection takes priority
    if _HAS_SPEED_CONFIG:
        speed_models = get_models_for_mode()
        # Filter to available models
        base_names = []
        for m in speed_models:
            if m == "rf":
                base_names.append("rf")
            elif m == "et":
                base_names.append("et")
            elif m == "lr":
                base_names.append("lr")
            elif m == "xgb" and _HAS_XGB:
                base_names.append("xgb")
            elif m == "lgb" and _HAS_LGB:
                base_names.append("lgb")
            elif m == "cat" and _HAS_CAT:
                base_names.append("cat")
            elif m == "bnn" and _HAS_TORCH:
                base_names.append("bnn")

        speed_mode = get_speed_mode()
        print(f"  [SPEED] Speed mode: {speed_mode.value} -> Models: {base_names}")

        # Initialize strategy as None when using speed config
        strategy = None

    # Fallback to market config or environment-based selection
    elif _HAS_MARKET_CONFIG:
        market_config = get_market_config(target_col)
        strategy = market_config.strategy
        recommended_models = get_base_models_for_market(target_col, len(classes))
        model_params = get_model_params_for_market(target_col)
        print(f"  [CHART] Market: {market_config.market_type.value}, Strategy: {strategy.value}")
    else:
        strategy = None
        recommended_models = ["rf", "et", "lr"]
        model_params = {}

    # Determine base learners based on strategy and availability
    # ONLY if speed_config didn't already set it
    if os.environ.get("MODELS_ONLY") == "rf":
        base_names = ["rf"]  # Ultra fast mode
    elif _HAS_SPEED_CONFIG and len(base_names) > 0:
        # Speed config already set base_names, don't override
        pass
    elif strategy == ModelStrategy.LIGHTWEIGHT if _HAS_MARKET_CONFIG else False:
        base_names = ["rf", "lr"]
        print(f"  [FAST] Using LIGHTWEIGHT models (RF + LR)")
    elif strategy == ModelStrategy.TREE_ENSEMBLE if _HAS_MARKET_CONFIG else False:
        base_names = ["rf", "et"]
        if _HAS_XGB: base_names.append("xgb")
        if _HAS_LGB: base_names.append("lgb")
        print(f"  🌲 Using TREE ENSEMBLE models")
    elif strategy == ModelStrategy.BOOSTING if _HAS_MARKET_CONFIG else False:
        base_names = []
        if _HAS_XGB: base_names.append("xgb")
        if _HAS_LGB: base_names.append("lgb")
        if _HAS_CAT: base_names.append("cat")
        if not base_names:
            base_names = ["rf", "et"]  # Fallback if no boosting available
        print(f"  [ROCKET] Using BOOSTING models")
    elif strategy == ModelStrategy.POISSON_BASED if _HAS_MARKET_CONFIG else False:
        base_names = ["rf"]  # DC will do heavy lifting
        print(f"  📐 Using POISSON-BASED approach (DC dominant)")
    else:
        # Full ensemble for FULL_ENSEMBLE or HYBRID strategies (only if not already set)
        if not (_HAS_SPEED_CONFIG and len(base_names) > 0):
            base_names = ["rf", "et", "lr"]
            if _HAS_XGB: base_names.append("xgb")
            if _HAS_LGB: base_names.append("lgb")
            if _HAS_CAT: base_names.append("cat")
            if _HAS_TORCH: base_names.append("bnn")
            print(f"  [TARGET] Using FULL ENSEMBLE models")

    # Add specialized models to ensemble (not replacing!)
    base_models: Dict[str, object] = {}

    if use_specialized:
        # Add specialized model as ONE component of the ensemble
        if is_binary_market(target_col) and len(classes) == 2:
            print(f"  [PLUS] Adding binary specialist to ensemble")
            specialist = BinaryMarketModel(target_col, random_state=RANDOM_SEED)
            specialist.fit(X_all, y_int)
            base_models["binary_specialist"] = specialist

        elif is_ordinal_market(target_col):
            print(f"  [PLUS] Adding ordinal specialist to ensemble")
            specialist = OrdinalMarketModel(target_col, classes, random_state=RANDOM_SEED)
            specialist.fit(X_all, y_int)
            base_models["ordinal_specialist"] = specialist

        elif is_multiclass_market(target_col):
            print(f"  [PLUS] Adding multiclass specialist to ensemble")
            specialist = MulticlassMarketModel(target_col, len(classes), random_state=RANDOM_SEED)
            specialist.fit(X_all, y_int)
            base_models["multiclass_specialist"] = specialist

    # Add CORAL for ordinal targets (tune C via Optuna if available)
    _coral_C = 1.0  # default; updated if tuned
    if target_col in ORDINAL_TARGETS:
        print(f"  [PLUS] Adding CORAL ordinal to ensemble")
        K = len(ORDINAL_TARGETS[target_col])
        # Tune CORAL C parameter if Optuna is available and tuning enabled
        if _HAS_OPTUNA and (not _HAS_SPEED_CONFIG or use_tuning()):
            try:
                from sklearn.preprocessing import LabelEncoder
                le_coral = LabelEncoder()
                y_coral = np.clip(y_int, 0, len(classes) - 1)
                ps_coral = make_time_split(len(y_coral), n_folds=3)
                cvd_coral = CVData(X=X_all, y=y_coral, ps=ps_coral,
                                   classes_=np.array(classes), label_encoder=le_coral)
                obj_coral = objective_factory("coral", cvd_coral)
                study_coral = optuna.create_study(direction="minimize")
                coral_trials = _adaptive_trials(10, len(X_all), len(classes))
                study_coral.optimize(obj_coral, n_trials=coral_trials, show_progress_bar=False)
                _coral_C = study_coral.best_params.get("C", 1.0)
                print(f"  [TUNE] CORAL C tuned to {_coral_C:.4f} ({coral_trials} trials)")
            except Exception as e:
                print(f"  [INFO] CORAL C tuning failed ({e}), using C=1.0")
                _coral_C = 1.0
        coral = CORALOrdinal(C=_coral_C, max_iter=2000)
        base_models["coral"] = coral

    # Optuna tune standard models with market-specific trials
    tuned: Dict[str, object] = {}
    for alg in ["rf", "et", "xgb", "lgb", "cat", "lr"]:
        if alg in base_names:
            with Timer(f"Optuna tune {alg} for {target_col}"):
                result = _tune_model(alg, X_all, y_int, np.array(classes), target_col=target_col)
                if result is not None:
                    tuned[alg] = result
                elif alg == "xgb":
                    print(f"[WARN]  Skipping XGBoost for {target_col} due to previous error")

    # Add tuned or default models to ensemble
    for name in base_names:
        if name in tuned:
            base_models[name] = tuned[name]
        elif name != "xgb":  # Skip XGBoost if it failed during tuning
            base_models[name] = _build_base_model(name, n_classes=len(classes), feature_names=feature_names, market_type=market_type_str)

    # Add DC pseudo-base if supported
    supports_dc = _dc_supported(target_col)
    if supports_dc:
        base_models["dc"] = "__DC__"

    # Walk-forward OOF with speed-aware fold count
    if _HAS_SPEED_CONFIG:
        n_folds = get_n_folds()
    else:
        n_folds = 5
    ps = make_time_split(len(y_int), n_folds=n_folds)
    oof_blocks = []
    for fold in np.unique(ps.test_fold):
        tr = ps.test_fold != fold
        va = ps.test_fold == fold
        Xt = X_all[tr]; yt = y_int[tr]
        Xv = X_all[va]; yv = y_int[va]
        fold_stack = []
        for name, model in base_models.items():
            try:
                if name == "dc":
                    proba = _dc_probs_for_rows(sub.iloc[tr], sub.iloc[va], target_col)
                else:
                    m = model
                    # Fit fresh copy per fold to keep OOF strict
                    if isinstance(model, (RandomForestClassifier, ExtraTreesClassifier, LogisticRegression)):
                        m = copy.deepcopy(model)
                    elif (_HAS_XGB and isinstance(model, xgb.XGBClassifier)) or (_HAS_LGB and isinstance(model, lgb.LGBMClassifier)) or (_HAS_CAT and isinstance(model, CatBoostClassifier)):
                        m = model.__class__(**model.get_params())
                    elif name == "coral":
                        m = CORALOrdinal(C=model.C, max_iter=model.max_iter)
                    elif name == "bnn" and _HAS_TORCH:
                        m = BNNWrapper(n_classes=len(classes), epochs=model.epochs, lr=model.lr, dropout=model.dropout, mc=model.mc, seed=model.seed)
                    if name != "dc":
                        m.fit(Xt, yt)
                        proba = m.predict_proba(Xv)
            except Exception as e:
                # Model failed in this fold - insert uniform proba to maintain OOF shape
                print(f"[WARN]  Model {name} failed in fold {fold}: {e}. Using uniform proba.")
                proba = np.full((len(Xv), len(classes)), 1.0 / len(classes))
            # align width
            if proba.shape[1] != len(classes):
                P2 = np.zeros((len(Xv), len(classes)))
                P2[:, :min(P2.shape[1], proba.shape[1])] = proba[:, :min(P2.shape[1], proba.shape[1])]
                s = P2.sum(axis=1, keepdims=True); s[s==0]=1.0
                proba = P2 / s
            fold_stack.append(proba)
        # concat base probs horizontally
        fold_oof = np.hstack(fold_stack)
        oof_blocks.append((va, fold_oof))

    # assemble full OOF in original order
    oof_pred = np.zeros((len(y_int), sum([len(classes) for _ in base_models])))
    for va_idx, block in oof_blocks:
        oof_pred[va_idx] = block

    # meta-learner selection: try multiple candidates and pick best OOF log loss
    meta = _select_meta_learner(oof_pred, y_int, len(classes), n_folds)

    # Calibration on OOF meta outputs
    if hasattr(meta, "decision_function"):
        decision_scores = meta.decision_function(oof_pred)
        if decision_scores.ndim == 1:  # Binary classification
            P_meta_oof = meta.predict_proba(oof_pred)
        else:  # Multi-class
            P_meta_oof = _softmax_np(decision_scores)
    else:
        P_meta_oof = meta.predict_proba(oof_pred)

    # Use market-aware calibration
    if _HAS_MARKET_CONFIG:
        calibrator = get_calibrator_for_market(target_col, len(classes))
        if isinstance(calibrator, IsotonicOrdinalCalibrator):
            print(f"  [TREND] Using ISOTONIC ordinal calibration")
            calibrator.fit(P_meta_oof, y_int)
        elif isinstance(calibrator, BetaCalibrator):
            print(f"  [TREND] Using BETA calibration")
            calibrator.fit(P_meta_oof, y_int)
        elif isinstance(calibrator, DirichletCalibrator):
            print(f"  [TREND] Using DIRICHLET calibration")
            calibrator.fit(P_meta_oof, y_int)
        else:
            logits = np.log(np.clip(P_meta_oof, 1e-12, 1-1e-12))
            calibrator.fit(logits, y_int)
    elif len(classes) > 2:
        calibrator = DirichletCalibrator(C=1.0, max_iter=2000).fit(P_meta_oof, y_int)
    else:
        # build pseudo logits
        logits = np.log(np.clip(P_meta_oof, 1e-12, 1-1e-12))
        calibrator = TemperatureScaler().fit(logits, y_int)

    # Fit base models on FULL data for inference
    full_stack = []
    fitted_bases: Dict[str, object] = {}
    for name, model in base_models.items():
        try:
            if name == "dc":
                proba = _dc_probs_for_rows(sub, sub, target_col)
                fitted_bases[name] = "__DC__"
            else:
                m = model
                if isinstance(model, (RandomForestClassifier, ExtraTreesClassifier, LogisticRegression)):
                    m = copy.deepcopy(model)
                elif (_HAS_XGB and isinstance(model, xgb.XGBClassifier)) or (_HAS_LGB and isinstance(model, lgb.LGBMClassifier)) or (_HAS_CAT and isinstance(model, CatBoostClassifier)):
                    m = model.__class__(**model.get_params())
                elif name == "coral":
                    m = CORALOrdinal(C=model.C, max_iter=model.max_iter)
                elif name == "bnn" and _HAS_TORCH:
                    m = BNNWrapper(n_classes=len(classes), epochs=model.epochs, lr=model.lr, dropout=model.dropout, mc=model.mc, seed=model.seed)
                m.fit(X_all, y_int)
                proba = m.predict_proba(X_all)
                fitted_bases[name] = m
        except Exception as e:
            # If XGBoost (or any model) fails during final fit, skip it and continue
            if name == "xgb":
                print(f"[WARN]  XGBoost failed during final fit: {e}. Continuing without XGBoost.")
                continue
            else:
                print(f"[WARN]  Model {name} failed during final fit: {e}. Continuing without this model.")
                continue
        # align width
        if proba.shape[1] != len(classes):
            P2 = np.zeros((len(X_all), len(classes)))
            P2[:, :min(P2.shape[1], proba.shape[1])] = proba[:, :min(P2.shape[1], proba.shape[1])]
            s = P2.sum(axis=1, keepdims=True); s[s==0]=1.0
            proba = P2 / s
        full_stack.append(proba)
    full_stack = np.hstack(full_stack)
    meta.fit(full_stack, y_int)  # refit meta on full stacked features

    # pack
    return TrainedTarget(
        target=target_col,
        classes_=classes,
        preprocessor=pre,
        base_models=fitted_bases,
        meta=meta,
        calibrator=calibrator,
        oof_pred=oof_pred,
        oof_y=y_int,
        feature_names=feature_names,
    )
# --------------------------------------------------------------------------------------
# Public API: train all targets, save/load, predict_proba
# --------------------------------------------------------------------------------------
def train_all_targets(models_dir: Path = MODEL_ARTIFACTS_DIR) -> Dict[str, TrainedTarget]:
    """Train all target markets with speed-aware optimization."""

    # Print speed configuration at start
    if _HAS_SPEED_CONFIG:
        print_speed_info()

    df = _load_features()
    models: Dict[str, TrainedTarget] = {}
    models_dir.mkdir(parents=True, exist_ok=True)

    # Get all available targets
    all_targets = [t for t in _all_targets() if t in df.columns]

    # Filter targets based on speed mode
    if _HAS_SPEED_CONFIG:
        targets = [t for t in all_targets if should_train_market(t)]
        skipped = len(all_targets) - len(targets)
        if skipped > 0:
            print(f"[SPEED] Skipping {skipped} low-priority markets in current speed mode")
    else:
        targets = all_targets

    print(f"[TRAINING] Training {len(targets)} markets")
    start_time = time.time()

    for i, t in enumerate(targets, 1):
        log_header(f"TRAIN {t} ({i}/{len(targets)})")
        sub = df.dropna(subset=[t])
        if sub.empty:
            continue
        trg = _fit_single_target(df, t)
        if trg is None:
            continue
        joblib.dump(trg, models_dir / f"{t}.joblib", compress=3)
        models[t] = trg

        # Time estimate
        elapsed = time.time() - start_time
        avg_per_target = elapsed / i
        remaining = avg_per_target * (len(targets) - i)
        print(f"[TIME] Est. {remaining/3600:.1f}h remaining ({i}/{len(targets)} done)")

    # save manifest
    with open(models_dir / "manifest.json", "w") as f:
        json.dump(sorted(list(models.keys())), f, indent=2)

    total_time = time.time() - start_time
    print(f"\n[OK] Training complete: {len(models)} models in {total_time/60:.1f} minutes")
    return models


def load_trained_targets(models_dir: Path = MODEL_ARTIFACTS_DIR) -> Dict[str, TrainedTarget]:
    models: Dict[str, TrainedTarget] = {}
    if not models_dir.exists():
        return models
    for p in models_dir.glob("y_*.joblib"):
        try:
            models[p.stem] = joblib.load(p)
        except Exception as e:
            print(f"[WARN] Failed to load model {p.stem}: {e}")
            continue
    return models


def predict_proba(models: Dict[str, TrainedTarget], df_future: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for t, trg in models.items():
        # preprocess
        Xf = trg.preprocessor.transform(df_future)
        # stacked base predictions
        stack_blocks = []
        for name, base in trg.base_models.items():
            if name == "dc":
                # fit DC on all history (safe at inference), then price rows
                hist = _load_features().dropna(subset=["FTHG","FTAG"]).copy()
                proba = _dc_probs_for_rows(hist, df_future, t)
            else:
                proba = base.predict_proba(Xf)
            # align width
            if proba.shape[1] != len(trg.classes_):
                P2 = np.zeros((len(df_future), len(trg.classes_)))
                P2[:, :min(P2.shape[1], proba.shape[1])] = proba[:, :min(P2.shape[1], proba.shape[1])]
                s = P2.sum(axis=1, keepdims=True); s[s==0]=1.0
                proba = P2 / s
            stack_blocks.append(proba)
        S = np.hstack(stack_blocks)
        # meta + calibration
        if hasattr(trg.meta, "decision_function"):
            decision_scores = trg.meta.decision_function(S)
            if decision_scores.ndim == 1:  # Binary classification
                P_meta = trg.meta.predict_proba(S)
            else:  # Multi-class
                P_meta = _softmax_np(decision_scores)
        else:
            P_meta = trg.meta.predict_proba(S)
        if isinstance(trg.calibrator, DirichletCalibrator):
            P = trg.calibrator.transform(P_meta)
        elif isinstance(trg.calibrator, IsotonicOrdinalCalibrator):
            P = trg.calibrator.transform(P_meta)
        elif isinstance(trg.calibrator, BetaCalibrator):
            P = trg.calibrator.transform(P_meta)
        elif isinstance(trg.calibrator, TemperatureScaler):
            # temperature scaler expects logits; rebuild logits via inverse softmax approx
            logits = np.log(np.clip(P_meta, 1e-12, 1-1e-12))
            P = trg.calibrator.transform(logits)
        else:
            P = P_meta
        # ensure valid probs
        eps = 1e-12
        P = np.clip(P, eps, 1.0)
        P = P / P.sum(axis=1, keepdims=True)
        out[t] = P
    return out


def analyse_feature_importance(models_dir: Path = MODEL_ARTIFACTS_DIR, top_n: int = 50) -> Dict[str, List]:
    """
    Analyse feature importance across all trained models.
    Returns per-target importance rankings and identifies consistently unimportant features.

    Args:
        models_dir: Directory with trained model artifacts
        top_n: Number of top features to show per model

    Returns:
        Dict with 'per_target' importances and 'drop_candidates' (features with zero
        importance across all tree-based models)
    """
    models = load_trained_targets(models_dir)
    if not models:
        print("[WARN] No trained models found")
        return {}

    all_importances: Dict[str, List[float]] = {}
    per_target = {}

    for target, trg in models.items():
        for name, base in trg.base_models.items():
            if name == "dc" or name == "__DC__":
                continue
            if hasattr(base, 'feature_importances_'):
                importances = base.feature_importances_
                # Get feature names from preprocessor
                try:
                    num_names = list(trg.preprocessor.transformers_[0][2] or [])
                    cat_names = list(trg.preprocessor.transformers_[1][2] or [])
                    feat_names = num_names + cat_names
                    # OHE may expand cat features
                    if len(feat_names) < len(importances):
                        feat_names = [f"f_{i}" for i in range(len(importances))]
                except Exception:
                    feat_names = [f"f_{i}" for i in range(len(importances))]

                ranked = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
                per_target[f"{target}_{name}"] = ranked[:top_n]

                for fname, imp in zip(feat_names, importances):
                    if fname not in all_importances:
                        all_importances[fname] = []
                    all_importances[fname].append(imp)

    # Features with zero importance across ALL models
    drop_candidates = [
        fname for fname, imps in all_importances.items()
        if all(imp == 0.0 for imp in imps) and len(imps) >= 3
    ]

    # Save analysis
    analysis = {
        'drop_candidates': sorted(drop_candidates),
        'drop_count': len(drop_candidates),
        'total_features': len(all_importances),
        'models_analysed': len(per_target),
    }

    analysis_path = models_dir / "feature_importance_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n[FEATURE IMPORTANCE ANALYSIS]")
    print(f"  Total features: {len(all_importances)}")
    print(f"  Zero-importance candidates for pruning: {len(drop_candidates)}")
    if drop_candidates:
        print(f"  Top drop candidates: {drop_candidates[:20]}")
    print(f"  Saved to {analysis_path}")

    return {'per_target': per_target, 'drop_candidates': drop_candidates, 'all_importances': all_importances}