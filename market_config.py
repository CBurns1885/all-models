# market_config.py
# Comprehensive market configuration for optimal model selection
# Maps each market type to its optimal modeling approach

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum

class MarketType(Enum):
    """Types of betting market prediction problems."""
    BINARY = "binary"           # 2 outcomes (Yes/No, Over/Under)
    TERNARY = "ternary"         # 3 outcomes (Home/Draw/Away, AH with push)
    MULTICLASS = "multiclass"   # 3+ unordered outcomes (HTFT 9 combos)
    ORDINAL = "ordinal"         # Ordered categories (0,1,2,3,4,5+ goals)
    POISSON = "poisson"         # Count data (exact scores)
    SPARSE = "sparse"           # Many classes with rare outcomes (CS)


class ModelStrategy(Enum):
    """Model strategy for different market characteristics."""
    FULL_ENSEMBLE = "full_ensemble"     # All models: RF, ET, XGB, LGB, CAT, LR, BNN
    TREE_ENSEMBLE = "tree_ensemble"     # Trees only: RF, ET, XGB, LGB
    BOOSTING = "boosting"               # Gradient boosting: XGB, LGB, CAT
    LIGHTWEIGHT = "lightweight"          # Fast: RF, LR only
    ORDINAL_CORAL = "ordinal_coral"     # CORAL ordinal regression
    POISSON_BASED = "poisson_based"     # Dixon-Coles + Poisson
    HYBRID = "hybrid"                   # ML + Parametric blending


@dataclass
class MarketConfig:
    """Configuration for a specific market type."""
    market_type: MarketType
    strategy: ModelStrategy
    use_dc_blend: bool              # Whether to blend with Dixon-Coles
    use_isotonic_calibration: bool  # Whether to use isotonic ordinal calibration
    ensemble_weight_boost: float    # Boost for high-confidence predictions
    class_weight_strategy: str      # "balanced", "balanced_subsample", None
    min_samples_leaf: int           # Minimum samples per leaf
    max_depth: Optional[int]        # Tree depth limit
    n_estimators_boost: float       # Multiplier for number of estimators
    special_handling: Optional[str] # Special processing notes


# ============================================================================
# COMPREHENSIVE MARKET CONFIGURATION
# ============================================================================

MARKET_CONFIGS: Dict[str, MarketConfig] = {
    # =========================================================================
    # CORE MARKETS
    # =========================================================================
    "y_1X2": MarketConfig(
        market_type=MarketType.TERNARY,
        strategy=ModelStrategy.FULL_ENSEMBLE,
        use_dc_blend=True,
        use_isotonic_calibration=False,
        ensemble_weight_boost=1.2,
        class_weight_strategy="balanced",
        min_samples_leaf=10,
        max_depth=None,
        n_estimators_boost=1.0,
        special_handling="Core market - use all available models including DC"
    ),

    "y_BTTS": MarketConfig(
        market_type=MarketType.BINARY,
        strategy=ModelStrategy.FULL_ENSEMBLE,
        use_dc_blend=True,
        use_isotonic_calibration=False,
        ensemble_weight_boost=1.0,
        class_weight_strategy="balanced",
        min_samples_leaf=10,
        max_depth=15,
        n_estimators_boost=1.0,
        special_handling="Binary - DC provides strong baseline"
    ),

    "y_GOAL_RANGE": MarketConfig(
        market_type=MarketType.ORDINAL,  # NOT multiclass - it has natural order!
        strategy=ModelStrategy.HYBRID,
        use_dc_blend=True,
        use_isotonic_calibration=True,  # Enforce monotonicity
        ensemble_weight_boost=1.0,
        class_weight_strategy="balanced",
        min_samples_leaf=15,
        max_depth=12,
        n_estimators_boost=1.0,
        special_handling="Ordinal with 6 ordered classes - use isotonic calibration"
    ),

    "y_CS": MarketConfig(
        market_type=MarketType.SPARSE,
        strategy=ModelStrategy.POISSON_BASED,
        use_dc_blend=True,
        use_isotonic_calibration=False,
        ensemble_weight_boost=0.8,  # Lower confidence for sparse outcomes
        class_weight_strategy="balanced",
        min_samples_leaf=5,
        max_depth=8,
        n_estimators_boost=0.5,  # Fewer trees - let DC do heavy lifting
        special_handling="37 classes (36 scores + other) - DC-dominant, ML for edge cases"
    ),

    # =========================================================================
    # OVER/UNDER MARKETS - Pure Binary
    # =========================================================================
}

# Generate O/U configs programmatically
for line in ["0_5", "1_5", "2_5", "3_5", "4_5", "5_5"]:
    MARKET_CONFIGS[f"y_OU_{line}"] = MarketConfig(
        market_type=MarketType.BINARY,
        strategy=ModelStrategy.TREE_ENSEMBLE,
        use_dc_blend=True,
        use_isotonic_calibration=False,
        ensemble_weight_boost=1.0 if line in ["1_5", "2_5", "3_5"] else 0.9,
        class_weight_strategy="balanced",
        min_samples_leaf=10,
        max_depth=15,
        n_estimators_boost=1.0,
        special_handling=f"Binary O/U {line.replace('_','.')} - DC provides excellent baseline"
    )

# =========================================================================
# ASIAN HANDICAP MARKETS - Binary/Ternary with push
# =========================================================================
for line in ["-2_0", "-1_5", "-1_0", "-0_5", "0_0", "+0_5", "+1_0", "+1_5", "+2_0"]:
    is_half = line.endswith("_5")
    MARKET_CONFIGS[f"y_AH_{line}"] = MarketConfig(
        market_type=MarketType.BINARY if is_half else MarketType.TERNARY,
        strategy=ModelStrategy.TREE_ENSEMBLE,
        use_dc_blend=True,
        use_isotonic_calibration=False,
        ensemble_weight_boost=0.95,
        class_weight_strategy="balanced",
        min_samples_leaf=10,
        max_depth=12,
        n_estimators_boost=1.0,
        special_handling="AH - half lines are binary, whole lines have push outcome"
    )

# =========================================================================
# TEAM GOALS O/U - Binary
# =========================================================================
for team in ["Home", "Away"]:
    for line in ["0_5", "1_5", "2_5", "3_5"]:
        MARKET_CONFIGS[f"y_{team}TG_{line}"] = MarketConfig(
            market_type=MarketType.BINARY,
            strategy=ModelStrategy.TREE_ENSEMBLE,
            use_dc_blend=False,  # No direct DC support
            use_isotonic_calibration=False,
            ensemble_weight_boost=1.0,
            class_weight_strategy="balanced",
            min_samples_leaf=10,
            max_depth=12,
            n_estimators_boost=1.0,
            special_handling=f"{team} team goals O/U - pure ML binary"
        )

# =========================================================================
# EXACT TOTAL GOALS - Ordinal (0,1,2,3,4,5,6+)
# =========================================================================
for i in range(7):
    label = f"{i}+" if i == 6 else str(i)
    MARKET_CONFIGS[f"y_ExactTotal_{label}"] = MarketConfig(
        market_type=MarketType.BINARY,  # Each is "exactly N goals or not"
        strategy=ModelStrategy.LIGHTWEIGHT,
        use_dc_blend=False,
        use_isotonic_calibration=False,
        ensemble_weight_boost=0.9,
        class_weight_strategy="balanced",
        min_samples_leaf=15,
        max_depth=10,
        n_estimators_boost=0.8,
        special_handling="Binary: is total exactly N goals?"
    )

# =========================================================================
# EXACT TEAM GOALS - Binary for each option
# =========================================================================
for team in ["Home", "Away"]:
    for i in ["0", "1", "2", "3+"]:
        MARKET_CONFIGS[f"y_{team}Exact_{i}"] = MarketConfig(
            market_type=MarketType.BINARY,
            strategy=ModelStrategy.LIGHTWEIGHT,
            use_dc_blend=False,
            use_isotonic_calibration=False,
            ensemble_weight_boost=0.9,
            class_weight_strategy="balanced",
            min_samples_leaf=15,
            max_depth=10,
            n_estimators_boost=0.8,
            special_handling="Binary: does team score exactly N goals?"
        )

# =========================================================================
# DOUBLE CHANCE - Pure Binary
# =========================================================================
for dc in ["1X", "X2", "12"]:
    MARKET_CONFIGS[f"y_DC_{dc}"] = MarketConfig(
        market_type=MarketType.BINARY,
        strategy=ModelStrategy.TREE_ENSEMBLE,
        use_dc_blend=False,  # Can derive from 1X2 but train separately
        use_isotonic_calibration=False,
        ensemble_weight_boost=1.0,
        class_weight_strategy="balanced",
        min_samples_leaf=10,
        max_depth=12,
        n_estimators_boost=1.0,
        special_handling="Binary double chance"
    )

# =========================================================================
# DRAW NO BET - Binary
# =========================================================================
for team in ["H", "A"]:
    MARKET_CONFIGS[f"y_DNB_{team}"] = MarketConfig(
        market_type=MarketType.BINARY,
        strategy=ModelStrategy.TREE_ENSEMBLE,
        use_dc_blend=False,
        use_isotonic_calibration=False,
        ensemble_weight_boost=1.0,
        class_weight_strategy="balanced",
        min_samples_leaf=10,
        max_depth=12,
        n_estimators_boost=1.0,
        special_handling="DNB - binary (draws refunded)"
    )

# =========================================================================
# HALF-TIME MARKETS
# =========================================================================
MARKET_CONFIGS["y_HT"] = MarketConfig(
    market_type=MarketType.TERNARY,
    strategy=ModelStrategy.TREE_ENSEMBLE,
    use_dc_blend=False,
    use_isotonic_calibration=False,
    ensemble_weight_boost=0.95,
    class_weight_strategy="balanced",
    min_samples_leaf=15,
    max_depth=10,
    n_estimators_boost=1.0,
    special_handling="HT 1X2 - less data correlation than FT"
)

MARKET_CONFIGS["y_HTFT"] = MarketConfig(
    market_type=MarketType.MULTICLASS,  # 9 unordered combinations
    strategy=ModelStrategy.TREE_ENSEMBLE,
    use_dc_blend=False,
    use_isotonic_calibration=False,
    ensemble_weight_boost=0.85,  # Lower - 9 classes harder
    class_weight_strategy="balanced",
    min_samples_leaf=20,
    max_depth=12,
    n_estimators_boost=1.2,  # More trees for complex task
    special_handling="HTFT 9 combos - multiclass, no natural order"
)

# HT O/U markets
for line in ["0_5", "1_5", "2_5"]:
    MARKET_CONFIGS[f"y_HT_OU_{line}"] = MarketConfig(
        market_type=MarketType.BINARY,
        strategy=ModelStrategy.LIGHTWEIGHT,
        use_dc_blend=False,
        use_isotonic_calibration=False,
        ensemble_weight_boost=0.9,
        class_weight_strategy="balanced",
        min_samples_leaf=15,
        max_depth=10,
        n_estimators_boost=0.8,
        special_handling="HT O/U - less predictable than FT"
    )

MARKET_CONFIGS["y_HT_BTTS"] = MarketConfig(
    market_type=MarketType.BINARY,
    strategy=ModelStrategy.LIGHTWEIGHT,
    use_dc_blend=False,
    use_isotonic_calibration=False,
    ensemble_weight_boost=0.85,
    class_weight_strategy="balanced",
    min_samples_leaf=15,
    max_depth=10,
    n_estimators_boost=0.8,
    special_handling="HT BTTS - rare, lower confidence"
)

# =========================================================================
# SECOND HALF MARKETS
# =========================================================================
for line in ["0_5", "1_5", "2_5"]:
    MARKET_CONFIGS[f"y_2H_OU_{line}"] = MarketConfig(
        market_type=MarketType.BINARY,
        strategy=ModelStrategy.LIGHTWEIGHT,
        use_dc_blend=False,
        use_isotonic_calibration=False,
        ensemble_weight_boost=0.9,
        class_weight_strategy="balanced",
        min_samples_leaf=15,
        max_depth=10,
        n_estimators_boost=0.8,
        special_handling="2H O/U - derived market"
    )

MARKET_CONFIGS["y_2H_BTTS"] = MarketConfig(
    market_type=MarketType.BINARY,
    strategy=ModelStrategy.LIGHTWEIGHT,
    use_dc_blend=False,
    use_isotonic_calibration=False,
    ensemble_weight_boost=0.85,
    class_weight_strategy="balanced",
    min_samples_leaf=15,
    max_depth=10,
    n_estimators_boost=0.8,
    special_handling="2H BTTS - rare outcome"
)

# =========================================================================
# EUROPEAN HANDICAP - Ternary (H/D/A after handicap)
# =========================================================================
for hc in ["m1", "m2", "p1", "p2"]:
    for outcome in ["H", "D", "A"]:
        MARKET_CONFIGS[f"y_EH_{hc}_{outcome}"] = MarketConfig(
            market_type=MarketType.BINARY,  # Each outcome is binary
            strategy=ModelStrategy.LIGHTWEIGHT,
            use_dc_blend=False,
            use_isotonic_calibration=False,
            ensemble_weight_boost=0.9,
            class_weight_strategy="balanced",
            min_samples_leaf=15,
            max_depth=10,
            n_estimators_boost=0.8,
            special_handling="EH outcome - binary for each"
        )

# =========================================================================
# FIRST TO SCORE - Ternary
# =========================================================================
MARKET_CONFIGS["y_FirstToScore"] = MarketConfig(
    market_type=MarketType.TERNARY,
    strategy=ModelStrategy.LIGHTWEIGHT,
    use_dc_blend=False,
    use_isotonic_calibration=False,
    ensemble_weight_boost=0.9,
    class_weight_strategy="balanced",
    min_samples_leaf=15,
    max_depth=10,
    n_estimators_boost=0.8,
    special_handling="First to score - home/away/none"
)

# =========================================================================
# TO SCORE MARKETS - Binary
# =========================================================================
for team in ["Home", "Away"]:
    MARKET_CONFIGS[f"y_{team}ToScore"] = MarketConfig(
        market_type=MarketType.BINARY,
        strategy=ModelStrategy.TREE_ENSEMBLE,
        use_dc_blend=False,
        use_isotonic_calibration=False,
        ensemble_weight_boost=1.0,
        class_weight_strategy="balanced",
        min_samples_leaf=10,
        max_depth=12,
        n_estimators_boost=1.0,
        special_handling="Will team score at least 1 goal"
    )

# =========================================================================
# WIN TO NIL / CLEAN SHEET - Binary
# =========================================================================
for market in ["HomeWTN", "AwayWTN", "HomeCS", "AwayCS", "NoGoal"]:
    MARKET_CONFIGS[f"y_{market}"] = MarketConfig(
        market_type=MarketType.BINARY,
        strategy=ModelStrategy.TREE_ENSEMBLE,
        use_dc_blend=False,
        use_isotonic_calibration=False,
        ensemble_weight_boost=0.95,
        class_weight_strategy="balanced",
        min_samples_leaf=12,
        max_depth=12,
        n_estimators_boost=1.0,
        special_handling="Clean sheet / WTN - binary"
    )

# =========================================================================
# WIN BY MARGIN - Binary for each margin
# =========================================================================
for team in ["Home", "Away"]:
    for margin in ["WinBy1", "WinBy2", "WinBy3+", "Win2+"]:
        MARKET_CONFIGS[f"y_{team}{margin}"] = MarketConfig(
            market_type=MarketType.BINARY,
            strategy=ModelStrategy.LIGHTWEIGHT,
            use_dc_blend=False,
            use_isotonic_calibration=False,
            ensemble_weight_boost=0.85,
            class_weight_strategy="balanced",
            min_samples_leaf=15,
            max_depth=10,
            n_estimators_boost=0.8,
            special_handling="Win by margin - specific outcomes"
        )

# =========================================================================
# HALF COMPARISON MARKETS
# =========================================================================
MARKET_CONFIGS["y_HigherHalf"] = MarketConfig(
    market_type=MarketType.TERNARY,  # 1H higher / Equal / 2H higher
    strategy=ModelStrategy.LIGHTWEIGHT,
    use_dc_blend=False,
    use_isotonic_calibration=False,
    ensemble_weight_boost=0.85,
    class_weight_strategy="balanced",
    min_samples_leaf=15,
    max_depth=10,
    n_estimators_boost=0.8,
    special_handling="Which half has more goals"
)

for market in ["GoalsBothHalves", "HomeScoresBothHalves", "AwayScoresBothHalves",
               "HomeWinEitherHalf", "AwayWinEitherHalf", "HomeWinBothHalves", "AwayWinBothHalves"]:
    MARKET_CONFIGS[f"y_{market}"] = MarketConfig(
        market_type=MarketType.BINARY,
        strategy=ModelStrategy.LIGHTWEIGHT,
        use_dc_blend=False,
        use_isotonic_calibration=False,
        ensemble_weight_boost=0.85,
        class_weight_strategy="balanced",
        min_samples_leaf=15,
        max_depth=10,
        n_estimators_boost=0.8,
        special_handling="Half-based outcomes"
    )

# =========================================================================
# ODD/EVEN MARKETS - Pure 50/50 baseline
# =========================================================================
for market in ["TotalOddEven", "HomeOddEven", "AwayOddEven"]:
    MARKET_CONFIGS[f"y_{market}"] = MarketConfig(
        market_type=MarketType.BINARY,
        strategy=ModelStrategy.LIGHTWEIGHT,
        use_dc_blend=False,
        use_isotonic_calibration=False,
        ensemble_weight_boost=0.7,  # Low - close to random
        class_weight_strategy=None,  # Near 50/50
        min_samples_leaf=20,
        max_depth=8,
        n_estimators_boost=0.5,
        special_handling="Odd/Even - near 50/50, limited edge"
    )

# =========================================================================
# MULTI-GOAL MARKETS - Binary
# =========================================================================
for n in [2, 3, 4, 5]:
    MARKET_CONFIGS[f"y_Match{n}+Goals"] = MarketConfig(
        market_type=MarketType.BINARY,
        strategy=ModelStrategy.TREE_ENSEMBLE,
        use_dc_blend=False,
        use_isotonic_calibration=False,
        ensemble_weight_boost=1.0 if n <= 3 else 0.9,
        class_weight_strategy="balanced",
        min_samples_leaf=10,
        max_depth=12,
        n_estimators_boost=1.0,
        special_handling=f"At least {n} goals in match"
    )

# =========================================================================
# COMBO MARKETS - Binary combinations
# =========================================================================

# Result + BTTS combos
for result in ["HomeWin", "AwayWin", "Draw"]:
    for btts in ["BTTS_Y", "BTTS_N"]:
        MARKET_CONFIGS[f"y_{result}_{btts}"] = MarketConfig(
            market_type=MarketType.BINARY,
            strategy=ModelStrategy.TREE_ENSEMBLE,
            use_dc_blend=False,
            use_isotonic_calibration=False,
            ensemble_weight_boost=0.9,
            class_weight_strategy="balanced",
            min_samples_leaf=15,
            max_depth=12,
            n_estimators_boost=1.0,
            special_handling="Result + BTTS combo"
        )

# Result + O/U combos
for result in ["HomeWin", "AwayWin", "Draw"]:
    for ou in ["O25", "U25"]:
        MARKET_CONFIGS[f"y_{result}_{ou}"] = MarketConfig(
            market_type=MarketType.BINARY,
            strategy=ModelStrategy.TREE_ENSEMBLE,
            use_dc_blend=False,
            use_isotonic_calibration=False,
            ensemble_weight_boost=0.9,
            class_weight_strategy="balanced",
            min_samples_leaf=15,
            max_depth=12,
            n_estimators_boost=1.0,
            special_handling="Result + O/U 2.5 combo"
        )

# DC + O/U combos
for dc in ["DC1X", "DCX2", "DC12"]:
    for ou in ["O25", "U25"]:
        MARKET_CONFIGS[f"y_{dc}_{ou}"] = MarketConfig(
            market_type=MarketType.BINARY,
            strategy=ModelStrategy.TREE_ENSEMBLE,
            use_dc_blend=False,
            use_isotonic_calibration=False,
            ensemble_weight_boost=0.9,
            class_weight_strategy="balanced",
            min_samples_leaf=15,
            max_depth=12,
            n_estimators_boost=1.0,
            special_handling="Double Chance + O/U 2.5 combo"
        )

# DC + BTTS combos
for dc in ["DC1X", "DCX2"]:
    for btts in ["BTTS_Y", "BTTS_N"]:
        MARKET_CONFIGS[f"y_{dc}_{btts}"] = MarketConfig(
            market_type=MarketType.BINARY,
            strategy=ModelStrategy.TREE_ENSEMBLE,
            use_dc_blend=False,
            use_isotonic_calibration=False,
            ensemble_weight_boost=0.9,
            class_weight_strategy="balanced",
            min_samples_leaf=15,
            max_depth=12,
            n_estimators_boost=1.0,
            special_handling="Double Chance + BTTS combo"
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_market_config(target_col: str) -> MarketConfig:
    """Get configuration for a specific target, with intelligent fallback."""
    if target_col in MARKET_CONFIGS:
        return MARKET_CONFIGS[target_col]

    # Intelligent fallback based on pattern matching
    if target_col.startswith("y_OU_"):
        return MarketConfig(
            market_type=MarketType.BINARY,
            strategy=ModelStrategy.TREE_ENSEMBLE,
            use_dc_blend=True,
            use_isotonic_calibration=False,
            ensemble_weight_boost=1.0,
            class_weight_strategy="balanced",
            min_samples_leaf=10,
            max_depth=15,
            n_estimators_boost=1.0,
            special_handling="O/U fallback"
        )
    elif target_col.startswith("y_AH_"):
        return MarketConfig(
            market_type=MarketType.BINARY,
            strategy=ModelStrategy.TREE_ENSEMBLE,
            use_dc_blend=True,
            use_isotonic_calibration=False,
            ensemble_weight_boost=0.95,
            class_weight_strategy="balanced",
            min_samples_leaf=10,
            max_depth=12,
            n_estimators_boost=1.0,
            special_handling="AH fallback"
        )
    else:
        # Default fallback for unknown markets
        return MarketConfig(
            market_type=MarketType.BINARY,
            strategy=ModelStrategy.LIGHTWEIGHT,
            use_dc_blend=False,
            use_isotonic_calibration=False,
            ensemble_weight_boost=0.8,
            class_weight_strategy="balanced",
            min_samples_leaf=15,
            max_depth=10,
            n_estimators_boost=0.8,
            special_handling="Unknown market - conservative defaults"
        )


def get_market_type(target_col: str) -> MarketType:
    """Get the market type for a target."""
    config = get_market_config(target_col)
    return config.market_type


def get_model_strategy(target_col: str) -> ModelStrategy:
    """Get the recommended model strategy for a target."""
    config = get_market_config(target_col)
    return config.strategy


def should_use_dc_blend(target_col: str) -> bool:
    """Check if target should be blended with Dixon-Coles."""
    config = get_market_config(target_col)
    return config.use_dc_blend


def should_use_isotonic(target_col: str) -> bool:
    """Check if target should use isotonic ordinal calibration."""
    config = get_market_config(target_col)
    return config.use_isotonic_calibration


def get_ensemble_weight_boost(target_col: str) -> float:
    """Get the confidence weight boost for a target."""
    config = get_market_config(target_col)
    return config.ensemble_weight_boost


# ============================================================================
# MARKET TYPE MAPPING FOR LEGACY COMPATIBILITY
# ============================================================================

ORDINAL_MARKETS: Set[str] = {
    "y_GOAL_RANGE",
}

TERNARY_MARKETS: Set[str] = {
    "y_1X2", "y_HT", "y_FirstToScore", "y_HigherHalf",
    "y_AH_-2_0", "y_AH_-1_0", "y_AH_0_0", "y_AH_+1_0", "y_AH_+2_0",  # Whole line AH
}

MULTICLASS_MARKETS: Set[str] = {
    "y_HTFT",  # 9 combinations, no natural order
}

SPARSE_MARKETS: Set[str] = {
    "y_CS",  # 37 classes
}

# All other markets are binary by default


def is_binary_market_v2(target_col: str) -> bool:
    """Enhanced binary market detection."""
    config = get_market_config(target_col)
    return config.market_type == MarketType.BINARY


def is_ordinal_market_v2(target_col: str) -> bool:
    """Enhanced ordinal market detection."""
    config = get_market_config(target_col)
    return config.market_type == MarketType.ORDINAL


def is_multiclass_market_v2(target_col: str) -> bool:
    """Enhanced multiclass market detection."""
    config = get_market_config(target_col)
    return config.market_type in [MarketType.MULTICLASS, MarketType.TERNARY]


def is_sparse_market(target_col: str) -> bool:
    """Check if market has sparse outcomes (many rare classes)."""
    config = get_market_config(target_col)
    return config.market_type == MarketType.SPARSE


# ============================================================================
# MODEL SELECTION BASED ON CONFIG
# ============================================================================

def get_base_models_for_market(target_col: str, n_classes: int) -> List[str]:
    """Get list of base model names to use for a target."""
    config = get_market_config(target_col)
    strategy = config.strategy

    if strategy == ModelStrategy.FULL_ENSEMBLE:
        return ["rf", "et", "xgb", "lgb", "cat", "lr"]
    elif strategy == ModelStrategy.TREE_ENSEMBLE:
        return ["rf", "et", "xgb", "lgb"]
    elif strategy == ModelStrategy.BOOSTING:
        return ["xgb", "lgb", "cat"]
    elif strategy == ModelStrategy.LIGHTWEIGHT:
        return ["rf", "lr"]
    elif strategy == ModelStrategy.ORDINAL_CORAL:
        return ["coral", "rf"]
    elif strategy == ModelStrategy.POISSON_BASED:
        return ["rf"]  # Let DC handle most of the work
    elif strategy == ModelStrategy.HYBRID:
        return ["rf", "et", "xgb", "lr"]
    else:
        return ["rf", "lr"]  # Default fallback


def get_model_params_for_market(target_col: str, base_n_estimators: int = 300) -> Dict:
    """Get model hyperparameters tuned for this market type."""
    config = get_market_config(target_col)

    n_estimators = int(base_n_estimators * config.n_estimators_boost)

    return {
        "n_estimators": n_estimators,
        "max_depth": config.max_depth,
        "min_samples_leaf": config.min_samples_leaf,
        "class_weight": config.class_weight_strategy,
    }
