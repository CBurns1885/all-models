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
        n_folds=3,  # Reduced from 5 for speed
        n_estimators=300,
        use_tuning=False,  # Use pre-tuned params (skip Optuna for speed)
        tuning_trials=0,
        models=["lgb"],  # LightGBM only (SINGLE MODEL - best overall)
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

CORE_MARKETS = [
    # Match Result
    "y_1X2",        # Match result - most important

    # Goals Markets (most popular)
    "y_BTTS",       # Both teams to score
    "y_OU_0_5",     # Over/Under 0.5
    "y_OU_1_5",     # Over/Under 1.5
    "y_OU_2_5",     # Over/Under 2.5 - most popular
    "y_OU_3_5",     # Over/Under 3.5
    "y_OU_4_5",     # Over/Under 4.5
    "y_GOAL_RANGE", # Goal bands

    # Double Chance
    "y_DC_1X",      # Home or Draw
    "y_DC_X2",      # Away or Draw
    "y_DC_12",      # Home or Away (no draw)

    # Draw No Bet
    "y_DNB_H",
    "y_DNB_A",

    # Team Goals
    "y_HomeTG_0_5", "y_HomeTG_1_5", "y_HomeTG_2_5",
    "y_AwayTG_0_5", "y_AwayTG_1_5", "y_AwayTG_2_5",

    # To Score
    "y_HomeToScore",
    "y_AwayToScore",

    # Asian Handicap (core lines)
    "y_AH_-0_5", "y_AH_0_0", "y_AH_+0_5",
    "y_AH_-1_0", "y_AH_+1_0",

    # Win to Nil
    "y_HomeWTN",
    "y_AwayWTN",

    # Multi-goal
    "y_Match2+Goals",
    "y_Match3+Goals",
]

SECONDARY_MARKETS = [
    # Extended Asian Handicap
    "y_AH_-1_5", "y_AH_-2_0", "y_AH_+1_5", "y_AH_+2_0",

    # Team exact goals
    "y_HomeExact_0", "y_HomeExact_1", "y_HomeExact_2", "y_HomeExact_3+",
    "y_AwayExact_0", "y_AwayExact_1", "y_AwayExact_2", "y_AwayExact_3+",

    # Exact total goals
    "y_ExactTotal_0", "y_ExactTotal_1", "y_ExactTotal_2",
    "y_ExactTotal_3", "y_ExactTotal_4", "y_ExactTotal_5", "y_ExactTotal_6+",

    # Clean sheets
    "y_HomeCS", "y_AwayCS", "y_NoGoal",

    # Win by margin
    "y_HomeWinBy1", "y_HomeWinBy2", "y_HomeWinBy3+",
    "y_AwayWinBy1", "y_AwayWinBy2", "y_AwayWinBy3+",
    "y_HomeWin2+", "y_AwayWin2+",

    # Result + BTTS combos
    "y_HomeWin_BTTS_Y", "y_HomeWin_BTTS_N",
    "y_AwayWin_BTTS_Y", "y_AwayWin_BTTS_N",
    "y_Draw_BTTS_Y", "y_Draw_BTTS_N",

    # Result + O/U combos
    "y_HomeWin_O25", "y_HomeWin_U25",
    "y_AwayWin_O25", "y_AwayWin_U25",
    "y_Draw_O25", "y_Draw_U25",

    # DC + combos
    "y_DC1X_O25", "y_DC1X_U25",
    "y_DCX2_O25", "y_DCX2_U25",
    "y_DC12_O25", "y_DC12_U25",
    "y_DC1X_BTTS_Y", "y_DC1X_BTTS_N",
    "y_DCX2_BTTS_Y", "y_DCX2_BTTS_N",

    # European Handicap
    "y_EH_m1_H", "y_EH_m1_D", "y_EH_m1_A",
    "y_EH_p1_H", "y_EH_p1_D", "y_EH_p1_A",
]

# Skip these in BALANCED mode (complex/rare)
# Tier 1: Core high-value markets (13 markets - ALWAYS train)
TIER1_CORE_MARKETS = [
    "y_1X2",              # Match result
    "y_BTTS",             # Both teams to score
    "y_OU_0_5", "y_OU_1_5", "y_OU_2_5", "y_OU_3_5", "y_OU_4_5",  # O/U goals
    "y_DC_1X", "y_DC_12", "y_DC_X2",  # Double chance
    "y_DNB_H", "y_DNB_A",  # Draw no bet
    "y_HomeToScore", "y_AwayToScore",  # Team to score
]

# Tier 2: Advanced value markets (15 markets - train in BALANCED/FULL)
TIER2_VALUE_MARKETS = [
    "y_HomeTG_0_5", "y_HomeTG_1_5",  # Home team goals O/U
    "y_AwayTG_0_5", "y_AwayTG_1_5",  # Away team goals O/U
    "y_AH_-0_5", "y_AH_+0_5", "y_AH_-1_0", "y_AH_+1_0", "y_AH_0_0",  # Asian handicap
    "y_HomeWin_BTTS_Y", "y_AwayWin_BTTS_Y",  # Result + BTTS combos
    "y_HomeWin_BTTS_N", "y_AwayWin_BTTS_N",
    "y_DC1X_O25", "y_DCX2_O25",  # DC + O/U combos
]

# Combined tier 1+2 (28 markets total)
PROFITABLE_MARKETS = TIER1_CORE_MARKETS + TIER2_VALUE_MARKETS

# Legacy names for backward compatibility
CORE_MARKETS = TIER1_CORE_MARKETS
SECONDARY_MARKETS = TIER2_VALUE_MARKETS

# Skip these in FAST mode (rarely bet, complex)
SKIP_IN_FAST_MODE = [
    "y_CS",  # Correct score - 37 classes, very hard
    "y_HTFT",  # 9 classes, needs lots of data

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
    """Check if market should be trained in current speed mode."""
    config = get_speed_config()

    if not config.skip_rare_markets:
        return True

    mode = get_speed_mode()

    if mode == SpeedMode.FAST:
        # Only train Tier 1 core markets (13 markets)
        return target in TIER1_CORE_MARKETS
    elif mode == SpeedMode.BALANCED:
        # Train Tier 1 + Tier 2 (28 markets total)
        return target in PROFITABLE_MARKETS
    else:
        # FULL mode: train all except explicitly skipped
        return target not in SKIP_IN_FAST_MODE


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
