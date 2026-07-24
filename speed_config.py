# speed_config.py
"""
Speed optimization configuration for the football prediction system.

Provides pre-tuned hyperparameters and speed modes to dramatically reduce
training time while maintaining accuracy.

Speed Modes:
- FAST: ~5-10 min (RF only, no tuning, 3 folds)
- BALANCED: ~20-30 min (RF+LGB, minimal tuning, 3 folds)
- FULL: ~2-3 hours (full ensemble, full tuning, 5 folds)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
import os


class SpeedMode(Enum):
    """Training speed modes."""
    FAST = "fast"           # Fastest - RF only, no tuning
    BALANCED = "balanced"   # Good speed/accuracy tradeoff
    FULL = "full"           # Maximum accuracy, slow


@dataclass
class SpeedConfig:
    """Configuration for a speed mode."""
    n_folds: int                    # CV folds (3 or 5)
    n_estimators: int               # Trees per model
    use_tuning: bool                # Whether to run Optuna
    tuning_trials: int              # Trials if tuning enabled
    models: list                    # Which models to use
    use_specialized: bool           # Use specialized market models
    use_dc: bool                    # Use Dixon-Coles blending
    skip_rare_markets: bool         # Skip markets with few samples
    max_depth: Optional[int]        # Tree depth limit
    early_stopping: bool            # Use early stopping for boosting
    n_jobs: int                     # Parallel jobs


# Speed mode configurations
SPEED_CONFIGS: Dict[SpeedMode, SpeedConfig] = {
    SpeedMode.FAST: SpeedConfig(
        n_folds=3,
        n_estimators=100,
        use_tuning=False,
        tuning_trials=0,
        models=["lgb"],  # LightGBM only - fast & accurate (SINGLE MODEL)
        use_specialized=False,
        use_dc=True,  # DC is fast and accurate
        skip_rare_markets=True,
        max_depth=10,
        early_stopping=True,
        n_jobs=-1
    ),
    SpeedMode.BALANCED: SpeedConfig(
        n_folds=3,
        n_estimators=200,
        use_tuning=False,  # Use pre-tuned params instead
        tuning_trials=0,
        models=["lgb"],  # LightGBM only (SINGLE MODEL - no ensemble)
        use_specialized=False,
        use_dc=True,
        skip_rare_markets=True,
        max_depth=12,
        early_stopping=True,
        n_jobs=-1
    ),
    SpeedMode.FULL: SpeedConfig(
        n_folds=3,
        n_estimators=300,
        use_tuning=False,
        tuning_trials=0,
        models=["lgb", "rf", "et", "xgb", "cat"],  # Full 5-model ensemble
        use_specialized=False,
        use_dc=True,
        skip_rare_markets=False,
        max_depth=None,
        early_stopping=True,
        n_jobs=-1
    ),
}


# ============================================================================
# PRE-TUNED HYPERPARAMETERS
# These were optimized on historical data - skip Optuna and use directly
# ============================================================================

PRETUNED_PARAMS = {
    # Random Forest - works well with these defaults
    "rf": {
        "binary": {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_split": 5,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
        "multiclass": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 4,
            "min_samples_leaf": 8,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
        "ordinal": {
            "n_estimators": 150,
            "max_depth": 10,
            "min_samples_split": 6,
            "min_samples_leaf": 15,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
    },

    # Extra Trees
    "et": {
        "binary": {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_split": 4,
            "min_samples_leaf": 8,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
        "multiclass": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 3,
            "min_samples_leaf": 6,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
        "ordinal": {
            "n_estimators": 150,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 12,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
    },

    # LightGBM - very fast, good accuracy
    "lgb": {
        "binary": {
            "n_estimators": 200,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        "multiclass": {
            "n_estimators": 250,
            "num_leaves": 48,
            "learning_rate": 0.05,
            "min_child_samples": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        "ordinal": {
            "n_estimators": 200,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 25,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
    },

    # XGBoost
    "xgb": {
        "binary": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
        "multiclass": {
            "n_estimators": 250,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
        "ordinal": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
    },

    # Logistic Regression
    "lr": {
        "binary": {"C": 1.0, "max_iter": 1000},
        "multiclass": {"C": 0.5, "max_iter": 1500},
        "ordinal": {"C": 1.0, "max_iter": 1000},
    },

    # CatBoost
    "cat": {
        "binary": {
            "iterations": 200,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
        },
        "multiclass": {
            "iterations": 250,
            "depth": 7,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
        },
        "ordinal": {
            "iterations": 200,
            "depth": 5,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
        },
    },
}


# ============================================================================
# MARKET TIERS - Focused on high-value betting markets
# ============================================================================

# PRIMARY TRAINED MARKETS — only these get their own ML model.
# Every market here has a genuinely independent signal worth learning.
# Everything else (DC, DNB, combos) is DERIVED from these mathematically.
PRIMARY_TRAINED_MARKETS = [
    # Match Result — the foundation; DC & DNB are derived from its probs
    "y_1X2",

    # Goals Markets — each O/U line is its own binary classification
    "y_BTTS",
    "y_OU_0_5",
    "y_OU_1_5",
    "y_OU_2_5",      # most popular line
    "y_OU_3_5",
    "y_OU_4_5",

    # Team-specific goals (independent signal from total goals)
    "y_HomeTG_0_5",
    "y_HomeTG_1_5",
    "y_AwayTG_0_5",
    "y_AwayTG_1_5",
]
# 11 models total — focused, fast, each with real independent signal

# Cards and corners markets — trained in FULL mode only
# Each represents a genuine independent betting market with separate signal
CARDS_CORNERS_MARKETS = [
    # Half-time / Full-time result (9-class multiclass)
    "y_HTFT",
    # Total yellow cards O/U
    "y_TotalYC_O1_5",
    "y_TotalYC_O2_5",
    "y_TotalYC_O3_5",
    "y_TotalYC_O4_5",
    "y_TotalYC_O5_5",
    "y_TotalYC_O6_5",
    # Booking points
    "y_BookingPts_O20_5",
    "y_BookingPts_O30_5",
    "y_BookingPts_O40_5",
    # Either team gets a card
    "y_HomeTeam_Card",
    "y_AwayTeam_Card",
    # Total corners O/U
    "y_TotalCorners_O6_5",
    "y_TotalCorners_O7_5",
    "y_TotalCorners_O8_5",
    "y_TotalCorners_O9_5",
    "y_TotalCorners_O10_5",
    "y_TotalCorners_O11_5",
    "y_TotalCorners_O12_5",
    "y_TotalCorners_O13_5",
    # Team corners
    "y_HomeCorners_O3_5",
    "y_HomeCorners_O4_5",
    "y_HomeCorners_O5_5",
    "y_HomeCorners_O6_5",
    "y_AwayCorners_O3_5",
    "y_AwayCorners_O4_5",
    "y_AwayCorners_O5_5",
    "y_AwayCorners_O6_5",
    "y_HomeCorners_Win",
    # Booking points (extra line)
    "y_BookingPts_O50_5",
]

# DERIVED MARKETS — NOT trained, computed from PRIMARY predictions:
#   y_DC_1X   = P(H) + P(D)                from y_1X2
#   y_DC_X2   = P(D) + P(A)                from y_1X2
#   y_DC_12   = P(H) + P(A)                from y_1X2
#   y_DNB_H   = P(H) / (P(H) + P(A))      from y_1X2 (exclude draw)
#   y_DNB_A   = P(A) / (P(H) + P(A))      from y_1X2 (exclude draw)
#   y_HomeToScore = P(HomeTG > 0.5)        = y_HomeTG_0_5 Over
#   y_AwayToScore = P(AwayTG > 0.5)        = y_AwayTG_0_5 Over
#   y_HomeWin_BTTS_Y  = P(H) * P(BTTS_Y)  approx, from y_1X2 + y_BTTS
#   y_HomeCS  = P(AwayTG < 0.5)            = y_AwayTG_0_5 Under
#   y_AwayCS  = P(HomeTG < 0.5)            = y_HomeTG_0_5 Under
#   etc.

# Legacy names kept for backward compatibility
CORE_MARKETS = PRIMARY_TRAINED_MARKETS

TIER1_CORE_MARKETS = PRIMARY_TRAINED_MARKETS

TIER2_VALUE_MARKETS = [
    # These are still listed for output generation, but are DERIVED not trained
    "y_DC_1X", "y_DC_X2", "y_DC_12",
    "y_DNB_H", "y_DNB_A",
    "y_HomeToScore", "y_AwayToScore",
    "y_AH_-0_5", "y_AH_+0_5", "y_AH_-1_0", "y_AH_+1_0", "y_AH_0_0",
    "y_HomeWin_BTTS_Y", "y_HomeWin_BTTS_N",
    "y_AwayWin_BTTS_Y", "y_AwayWin_BTTS_N",
    "y_DC1X_O25", "y_DCX2_O25",
]

PROFITABLE_MARKETS = PRIMARY_TRAINED_MARKETS + TIER2_VALUE_MARKETS

SECONDARY_MARKETS = TIER2_VALUE_MARKETS

# Skip these in FAST mode (rarely bet, complex)
SKIP_IN_FAST_MODE = [
    "y_CS",  # Correct score - 37 classes, very hard
    # Half-time markets (less data, harder to predict)
    "y_HT",
    "y_HT_OU_0_5", "y_HT_OU_1_5", "y_HT_OU_2_5",
    "y_HT_BTTS",

    # Second half (even less predictable)
    "y_2H_OU_0_5", "y_2H_OU_1_5", "y_2H_OU_2_5",
    "y_2H_BTTS",

    # Rare/complex
    "y_HigherHalf",
    "y_GoalsBothHalves",
    "y_HomeScoresBothHalves", "y_AwayScoresBothHalves",
    "y_HomeWinEitherHalf", "y_AwayWinEitherHalf",
    "y_HomeWinBothHalves", "y_AwayWinBothHalves",
    "y_FirstToScore",
    "y_TotalOddEven", "y_HomeOddEven", "y_AwayOddEven",
    "y_Match4+Goals", "y_Match5+Goals",  # Rare outcomes

    # Extended European Handicap (less common)
    "y_EH_m2_H", "y_EH_m2_D", "y_EH_m2_A",
    "y_EH_p2_H", "y_EH_p2_D", "y_EH_p2_A",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_speed_mode() -> SpeedMode:
    """Get current speed mode from environment or default."""
    mode_str = os.environ.get("SPEED_MODE", "balanced").lower()
    try:
        return SpeedMode(mode_str)
    except ValueError:
        return SpeedMode.BALANCED


def get_speed_config() -> SpeedConfig:
    """Get configuration for current speed mode."""
    mode = get_speed_mode()
    return SPEED_CONFIGS[mode]


def get_pretuned_params(model: str, market_type: str) -> Dict:
    """Get pre-tuned parameters for a model and market type."""
    if model not in PRETUNED_PARAMS:
        return {}
    if market_type not in PRETUNED_PARAMS[model]:
        # Default to binary params
        return PRETUNED_PARAMS[model].get("binary", {})
    return PRETUNED_PARAMS[model][market_type]


def should_train_market(target: str) -> bool:
    """Check if a market needs its own trained ML model.

    - FAST: core goals markets only (5 models)
    - BALANCED: all PRIMARY_TRAINED_MARKETS (11 models)
    - FULL: PRIMARY + CARDS_CORNERS_MARKETS (34 models)

    DC, DNB, combos, etc. are derived mathematically — no model needed.
    """
    mode = get_speed_mode()

    if mode == SpeedMode.FAST:
        fast_subset = [
            "y_1X2", "y_BTTS",
            "y_OU_1_5", "y_OU_2_5", "y_OU_3_5",
        ]
        return target in fast_subset
    elif mode == SpeedMode.FULL:
        return target in PRIMARY_TRAINED_MARKETS or target in CARDS_CORNERS_MARKETS
    else:
        # BALANCED: primary markets only (fast enough for daily use)
        return target in PRIMARY_TRAINED_MARKETS


def get_models_for_mode() -> list:
    """Get list of models to use for current speed mode."""
    config = get_speed_config()
    return config.models


def get_n_folds() -> int:
    """Get number of CV folds for current speed mode."""
    config = get_speed_config()
    return config.n_folds


def get_n_estimators() -> int:
    """Get number of estimators for current speed mode."""
    config = get_speed_config()
    return config.n_estimators


def use_tuning() -> bool:
    """Check if Optuna tuning should be used."""
    config = get_speed_config()
    return config.use_tuning


def use_specialized_models() -> bool:
    """Check if specialized market models should be used."""
    config = get_speed_config()
    return config.use_specialized


def print_speed_info():
    """Print current speed configuration."""
    mode = get_speed_mode()
    config = get_speed_config()

    print(f"\n{'='*60}")
    print(f"[FAST] SPEED MODE: {mode.value.upper()}")
    print(f"{'='*60}")
    print(f"  Models: {', '.join(config.models)}")
    print(f"  CV Folds: {config.n_folds}")
    print(f"  Trees: {config.n_estimators}")
    print(f"  Tuning: {'Yes' if config.use_tuning else 'No (using pre-tuned params)'}")
    print(f"  Specialized: {'Yes' if config.use_specialized else 'No'}")
    print(f"  DC Blend: {'Yes' if config.use_dc else 'No'}")

    if mode == SpeedMode.FAST:
        print(f"  Markets: Core only ({len(CORE_MARKETS)} targets)")
        print(f"  Est. Time: 5-10 minutes")
    elif mode == SpeedMode.BALANCED:
        print(f"  Markets: Core + Secondary (skip complex)")
        print(f"  Est. Time: 20-30 minutes")
    else:
        print(f"  Markets: All available")
        print(f"  Est. Time: 2-3 hours")
    print(f"{'='*60}\n")
