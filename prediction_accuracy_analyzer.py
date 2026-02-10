#!/usr/bin/env python3
"""
Prediction Accuracy Analyzer & Improvement Recommender

Analyzes historical prediction accuracy across all markets, diagnoses
systemic issues (overconfidence, calibration drift, feature gaps), and
proposes concrete updates to features, models, and calibration that
will improve future accuracy.

Usage:
    python prediction_accuracy_analyzer.py                  # Full analysis
    python prediction_accuracy_analyzer.py --market 1X2     # Single market deep-dive
    python prediction_accuracy_analyzer.py --apply           # Apply safe auto-fixes
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR, DATA_DIR, FEATURES_PARQUET, MODEL_ARTIFACTS_DIR,
    FORM_WINDOWS, EWM_SPAN, LEAGUE_CODES, PRIORITY_LEAGUES,
    log_header,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path("outputs/accuracy_database.db")
REPORT_DIR = OUTPUT_DIR  # same dated folder as other outputs
MIN_SAMPLES = 20  # minimum predictions to analyse a bucket

# Confidence buckets for calibration analysis
CONF_EDGES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
CONF_LABELS = [f"{int(lo*100)}-{int(hi*100)-1}%"
               for lo, hi in zip(CONF_EDGES[:-1], CONF_EDGES[1:])]

# Market tiers for prioritised reporting
TIER1_MARKETS = [
    "1X2", "BTTS", "OU_0_5", "OU_1_5", "OU_2_5", "OU_3_5", "OU_4_5",
    "DC_1X", "DC_12", "DC_X2", "DNB_H", "DNB_A", "HomeToScore", "AwayToScore",
]
TIER2_MARKETS = [
    "HomeTG_0_5", "HomeTG_1_5", "AwayTG_0_5", "AwayTG_1_5",
    "AH_m1_0", "AH_m0_5", "AH_0_0", "AH_p0_5", "AH_p1_0",
    "HWin_BTTS_Y", "HWin_BTTS_N", "AWin_BTTS_Y", "AWin_BTTS_N",
    "DC1X_O2_5", "DCX2_O2_5",
]


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBucket:
    """One row in a calibration table."""
    confidence_lo: float
    confidence_hi: float
    n_predictions: int
    n_correct: int
    accuracy: float
    expected_accuracy: float  # midpoint of the bucket
    gap: float               # accuracy - expected_accuracy (negative = overconfident)


@dataclass
class MarketDiagnosis:
    """Diagnosis for a single market."""
    market: str
    total_predictions: int = 0
    total_with_results: int = 0
    overall_accuracy: float = 0.0
    brier_score: float = 0.0
    calibration_buckets: List[CalibrationBucket] = field(default_factory=list)
    overconfidence_score: float = 0.0  # avg gap in top buckets (negative = bad)
    league_accuracies: Dict[str, float] = field(default_factory=dict)
    worst_leagues: List[str] = field(default_factory=list)
    best_leagues: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FeatureRecommendation:
    """A concrete feature engineering recommendation."""
    name: str
    description: str
    priority: str            # "high", "medium", "low"
    affected_markets: List[str]
    implementation_hint: str  # pseudo-code / file + function to change
    expected_impact: str


@dataclass
class ModelRecommendation:
    """A concrete model / calibration recommendation."""
    name: str
    description: str
    priority: str
    affected_markets: List[str]
    implementation_hint: str
    expected_impact: str


@dataclass
class AnalysisReport:
    """Full report produced by the analyser."""
    generated_at: str
    total_predictions: int
    total_with_results: int
    overall_accuracy: float
    overall_brier: float
    market_diagnoses: List[MarketDiagnosis]
    feature_recommendations: List[FeatureRecommendation]
    model_recommendations: List[ModelRecommendation]
    config_recommendations: List[str]


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def load_predictions(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load all predictions from the accuracy database."""
    if not db_path.exists():
        print(f"[WARN] Accuracy database not found at {db_path}")
        print("       Run the weekly pipeline first, then update results.")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
            prediction_id,
            week_id,
            prediction_date,
            match_date,
            league,
            home_team,
            away_team,
            market,
            predicted_outcome,
            predicted_probability,
            actual_outcome,
            correct
        FROM predictions
    """, conn)
    conn.close()

    if df.empty:
        print("[WARN] No predictions in the database yet.")
    else:
        print(f"[OK] Loaded {len(df)} predictions "
              f"({df['correct'].notna().sum()} with results)")
    return df


def compute_calibration(df: pd.DataFrame) -> List[CalibrationBucket]:
    """Compute calibration buckets for a set of predictions."""
    buckets = []
    for lo, hi, label in zip(CONF_EDGES[:-1], CONF_EDGES[1:], CONF_LABELS):
        mask = (df["predicted_probability"] >= lo) & (df["predicted_probability"] < hi)
        sub = df[mask]
        if len(sub) < MIN_SAMPLES:
            continue
        n = len(sub)
        correct = int(sub["correct"].sum())
        acc = correct / n
        expected = (lo + hi) / 2
        gap = acc - expected
        buckets.append(CalibrationBucket(
            confidence_lo=lo,
            confidence_hi=hi,
            n_predictions=n,
            n_correct=correct,
            accuracy=round(acc, 4),
            expected_accuracy=round(expected, 4),
            gap=round(gap, 4),
        ))
    return buckets


def diagnose_market(df_market: pd.DataFrame, market_name: str) -> MarketDiagnosis:
    """Deep diagnosis for one market."""
    diag = MarketDiagnosis(market=market_name)

    diag.total_predictions = len(df_market)
    with_results = df_market[df_market["correct"].notna()].copy()
    diag.total_with_results = len(with_results)

    if diag.total_with_results < MIN_SAMPLES:
        diag.issues.append(f"Insufficient data ({diag.total_with_results} results)")
        return diag

    diag.overall_accuracy = round(with_results["correct"].mean(), 4)

    # Brier score
    prob = with_results["predicted_probability"].values
    correct = with_results["correct"].values.astype(float)
    diag.brier_score = round(float(np.mean((prob - correct) ** 2)), 4)

    # Calibration
    diag.calibration_buckets = compute_calibration(with_results)

    # Overconfidence score: mean gap in buckets >= 70%
    high_buckets = [b for b in diag.calibration_buckets if b.confidence_lo >= 0.70]
    if high_buckets:
        diag.overconfidence_score = round(
            np.mean([b.gap for b in high_buckets]), 4
        )

    # League breakdown
    league_acc = (
        with_results.groupby("league")["correct"]
        .agg(["mean", "count"])
        .query("count >= 10")
        .sort_values("mean")
    )
    diag.league_accuracies = {
        lg: round(float(row["mean"]), 4)
        for lg, row in league_acc.iterrows()
    }
    if len(league_acc) >= 2:
        diag.worst_leagues = league_acc.head(3).index.tolist()
        diag.best_leagues = league_acc.tail(3).index.tolist()

    # ---- Issue detection ----
    if diag.overall_accuracy < 0.45:
        diag.issues.append(
            f"Very low accuracy ({diag.overall_accuracy:.1%}) — "
            "model may be worse than baseline"
        )
    elif diag.overall_accuracy < 0.52:
        diag.issues.append(
            f"Marginal accuracy ({diag.overall_accuracy:.1%}) — "
            "barely above coin-flip"
        )

    if diag.overconfidence_score < -0.10:
        diag.issues.append(
            f"Severe overconfidence (gap={diag.overconfidence_score:+.1%}) — "
            "high-confidence predictions are unreliable"
        )
    elif diag.overconfidence_score < -0.05:
        diag.issues.append(
            f"Moderate overconfidence (gap={diag.overconfidence_score:+.1%})"
        )

    if diag.brier_score > 0.30:
        diag.issues.append(
            f"High Brier score ({diag.brier_score:.3f}) — "
            "probability estimates are poorly calibrated"
        )

    # Check for leagues dragging down accuracy
    for lg in diag.worst_leagues[:2]:
        lg_acc = diag.league_accuracies.get(lg, 0)
        if lg_acc < diag.overall_accuracy - 0.10:
            diag.issues.append(
                f"League {lg} accuracy ({lg_acc:.1%}) is far below average"
            )

    # ---- Per-market recommendations ----
    if diag.overconfidence_score < -0.05:
        diag.recommendations.append(
            "Apply stricter probability calibration (isotonic regression "
            "or temperature scaling with higher T)"
        )
    if diag.brier_score > 0.25:
        diag.recommendations.append(
            "Consider Platt scaling or Beta calibration for this market"
        )
    if market_name == "1X2" and diag.overall_accuracy < 0.50:
        diag.recommendations.append(
            "Blend Dixon-Coles probabilities more heavily for 1X2 "
            "(increase DC weight from default)"
        )
    if market_name.startswith("AH") and diag.overall_accuracy < 0.45:
        diag.recommendations.append(
            "Asian Handicap accuracy is very low — consider dropping "
            "this market or using a Poisson-only approach"
        )
    if market_name.startswith("OU") and diag.overconfidence_score < -0.08:
        diag.recommendations.append(
            "Over/Under calibration is off — use league-specific "
            "goal-rate priors to anchor probabilities"
        )
    for lg in diag.worst_leagues[:2]:
        lg_acc = diag.league_accuracies.get(lg, 0)
        if lg_acc < 0.40:
            diag.recommendations.append(
                f"Consider excluding or down-weighting league {lg} "
                f"(accuracy {lg_acc:.1%})"
            )

    return diag


# ---------------------------------------------------------------------------
# Feature gap analysis
# ---------------------------------------------------------------------------

def analyse_features() -> List[FeatureRecommendation]:
    """Inspect the current feature set and recommend additions."""
    recs: List[FeatureRecommendation] = []

    # Load feature parquet to see what columns exist
    try:
        feat_df = pd.read_parquet(FEATURES_PARQUET)
        cols = set(feat_df.columns)
    except Exception:
        cols = set()
        recs.append(FeatureRecommendation(
            name="Rebuild features",
            description="features.parquet is missing or unreadable — rebuild it",
            priority="high",
            affected_markets=["all"],
            implementation_hint="python run_weekly.py (steps 1-2)",
            expected_impact="Cannot predict without features",
        ))
        return recs

    print(f"[OK] Feature matrix has {len(cols)} columns, "
          f"{len(feat_df)} rows")

    # Check 1: Rolling window gaps
    has_ma3 = any("ma3" in c for c in cols)
    has_ma5 = any("ma5" in c for c in cols)
    has_ma10 = any("ma10" in c for c in cols)
    has_ma20 = any("ma20" in c for c in cols)
    has_ewm = any("ewm" in c.lower() for c in cols)

    if has_ma3 and has_ma5 and not has_ma10:
        recs.append(FeatureRecommendation(
            name="Add ma10 rolling windows",
            description=(
                "Currently only ma3 and ma5 windows exist. A 10-match "
                "window captures medium-term form and is standard in "
                "football prediction literature."
            ),
            priority="high",
            affected_markets=["1X2", "BTTS", "OU_2_5", "DC_1X", "DNB_H"],
            implementation_hint=(
                "features.py:_rolling_stats() — FORM_WINDOWS in config.py "
                "already includes 10 but the feature builder may be pruning "
                "columns. Verify that config.FORM_WINDOWS = [3, 5, 10, 20] "
                "and that _pivot_back() preserves ma10 columns."
            ),
            expected_impact="~1-3% accuracy gain on form-dependent markets",
        ))

    if not has_ma20:
        recs.append(FeatureRecommendation(
            name="Add ma20 rolling windows",
            description=(
                "A 20-match (~half-season) window provides a stable "
                "baseline for team quality. Useful for reducing noise "
                "in smaller leagues."
            ),
            priority="medium",
            affected_markets=["1X2", "DC_1X", "DC_X2"],
            implementation_hint=(
                "Same as ma10 — config.FORM_WINDOWS already has 20 "
                "but columns are lost during feature pivot. Check "
                "_pivot_back() and _build_side_features() in features.py."
            ),
            expected_impact="~0.5-1% accuracy gain, better for cups/lesser leagues",
        ))

    if not has_ewm:
        recs.append(FeatureRecommendation(
            name="Add EWM (exponentially weighted) features",
            description=(
                "Exponential weighting emphasises recent matches. "
                "Configured (EWM_SPAN=10) but may not be propagated "
                "to the final feature matrix."
            ),
            priority="high",
            affected_markets=["1X2", "BTTS", "OU_2_5"],
            implementation_hint=(
                "features.py:_rolling_stats() computes GF_ewm, GA_ewm, "
                "PPG_ewm. Confirm they survive _pivot_back() and appear "
                "as Home_GF_ewm / Away_GF_ewm in the parquet."
            ),
            expected_impact="~1-2% accuracy gain — momentum is predictive",
        ))

    # Check 2: Elo features
    has_elo = "Elo_Home" in cols and "Elo_Away" in cols
    has_elo_diff = "Elo_Diff" in cols
    has_elo_mom = "Elo_Mom_Home" in cols

    if not has_elo:
        recs.append(FeatureRecommendation(
            name="Restore Elo ratings",
            description=(
                "Elo columns were previously fixed but may have been "
                "lost again during a rebuild. Elo is a top-3 feature "
                "for 1X2 prediction."
            ),
            priority="high",
            affected_markets=["1X2", "DC_1X", "DC_12", "DC_X2", "DNB_H", "DNB_A"],
            implementation_hint=(
                "features.py — ensure _elo_by_league() output columns "
                "(Elo_Home, Elo_Away, Elo_Diff) are preserved through "
                "_build_side_features() and _pivot_back()."
            ),
            expected_impact="~2-5% accuracy gain on result-based markets",
        ))

    if has_elo and not has_elo_mom:
        recs.append(FeatureRecommendation(
            name="Add Elo momentum features",
            description=(
                "Elo_Mom_Home / Elo_Mom_Away track recent Elo trajectory. "
                "Helps distinguish rising teams from established strong teams."
            ),
            priority="medium",
            affected_markets=["1X2", "DC_1X", "DNB_H"],
            implementation_hint=(
                "features.py:_elo_by_league() already computes momentum. "
                "Ensure Elo_Mom_Home, Elo_Mom_Away, Elo_Mom_Diff are "
                "preserved after the pivot step."
            ),
            expected_impact="~0.5-1% accuracy gain",
        ))

    # Check 3: xG features
    has_xg = "Home_xG" in cols or any("xG" in c for c in cols)
    if not has_xg:
        recs.append(FeatureRecommendation(
            name="Enable xG features",
            description=(
                "Expected goals (xG) are among the most predictive "
                "features in football analytics. Missing from the "
                "current feature set."
            ),
            priority="high",
            affected_markets=["OU_2_5", "BTTS", "1X2", "HomeToScore", "AwayToScore"],
            implementation_hint=(
                "Ensure USE_XG_FEATURES=1 in config.py and that "
                "api_football_adapter.py is populating Home_xG / Away_xG. "
                "features.py already handles xG in _add_team_side()."
            ),
            expected_impact="~2-4% accuracy gain on goals-related markets",
        ))

    # Check 4: Head-to-head features
    has_h2h = any("h2h" in c.lower() for c in cols)
    if not has_h2h:
        recs.append(FeatureRecommendation(
            name="Add head-to-head (H2H) features",
            description=(
                "Historical matchup record between the two specific teams. "
                "Captures rivalry dynamics and stylistic matchup effects. "
                "Common in professional prediction systems."
            ),
            priority="medium",
            affected_markets=["1X2", "BTTS", "OU_2_5"],
            implementation_hint=(
                "In features.py, add a function that for each row looks "
                "back at the last 5-10 meetings between HomeTeam and "
                "AwayTeam (in either direction). Compute: h2h_home_win_rate, "
                "h2h_avg_goals, h2h_btts_rate. Requires sorting by Date "
                "and using .shift(1) to prevent leakage."
            ),
            expected_impact="~1-2% accuracy gain on derby/rivalry matches",
        ))

    # Check 5: Odds-implied probability features
    has_odds = any("B365" in c or "Odds_" in c for c in cols)
    if not has_odds:
        recs.append(FeatureRecommendation(
            name="Add bookmaker odds as features",
            description=(
                "Market-implied probabilities (from B365, Pinnacle etc.) "
                "encode public information that is hard to replicate. "
                "Using them as input features is a proven accuracy booster."
            ),
            priority="medium",
            affected_markets=["1X2", "OU_2_5", "BTTS"],
            implementation_hint=(
                "In features.py:_pivot_back(), ensure odds columns "
                "(B365H, B365D, B365A, Odds_O25, Odds_BTTS_Y) are "
                "preserved. Convert to implied probabilities: "
                "imp_prob = 1/odds, then normalise to remove overround."
            ),
            expected_impact="~2-5% accuracy gain — market is very efficient",
        ))

    # Check 6: Goal difference / form streak features
    has_streak = any("streak" in c.lower() or "unbeaten" in c.lower() for c in cols)
    if not has_streak:
        recs.append(FeatureRecommendation(
            name="Add form streak features",
            description=(
                "Current win/unbeaten/loss streak captures psychological "
                "momentum. Teams on long winning streaks behave differently "
                "from their rolling averages."
            ),
            priority="low",
            affected_markets=["1X2", "DC_1X", "DNB_H"],
            implementation_hint=(
                "In features.py:_rolling_stats(), after computing rolling "
                "averages, add: current_win_streak, current_unbeaten_streak, "
                "current_loss_streak. Use a simple loop or cumsum approach "
                "on the Win/Loss columns."
            ),
            expected_impact="~0.5-1% marginal gain",
        ))

    # Check 7: Rest days between matches
    has_rest = any("rest" in c.lower() or "days_since" in c.lower() for c in cols)
    if not has_rest:
        recs.append(FeatureRecommendation(
            name="Add rest days between matches",
            description=(
                "The number of days since each team's last match affects "
                "fatigue and performance. Especially important for cup "
                "matches and midweek fixtures."
            ),
            priority="medium",
            affected_markets=["1X2", "BTTS", "OU_2_5"],
            implementation_hint=(
                "In features.py:_rolling_stats(), for each team compute "
                "days_since_last = Date - previous_match_Date. Also compute "
                "rest_diff = home_rest - away_rest at the match level."
            ),
            expected_impact="~0.5-1.5% gain, especially for midweek fixtures",
        ))

    return recs


# ---------------------------------------------------------------------------
# Model & calibration analysis
# ---------------------------------------------------------------------------

def analyse_models_and_calibration(
    diagnoses: List[MarketDiagnosis],
) -> List[ModelRecommendation]:
    """Generate model/calibration recommendations from market diagnoses."""
    recs: List[ModelRecommendation] = []

    # Aggregate issues across all markets
    overconfident_markets = [
        d.market for d in diagnoses
        if d.overconfidence_score < -0.05 and d.total_with_results >= MIN_SAMPLES
    ]
    low_accuracy_markets = [
        d.market for d in diagnoses
        if d.overall_accuracy < 0.48 and d.total_with_results >= MIN_SAMPLES
    ]
    high_brier_markets = [
        d.market for d in diagnoses
        if d.brier_score > 0.25 and d.total_with_results >= MIN_SAMPLES
    ]

    # --- Recommendation 1: Global recalibration ---
    if len(overconfident_markets) >= 3:
        recs.append(ModelRecommendation(
            name="Apply global temperature scaling",
            description=(
                f"{len(overconfident_markets)} markets are overconfident "
                f"({', '.join(overconfident_markets[:5])}). The model is "
                "systematically producing probabilities that are too extreme. "
                "Temperature scaling with T > 1 will soften predictions."
            ),
            priority="high",
            affected_markets=overconfident_markets,
            implementation_hint=(
                "calibration.py:TemperatureScaler — after fitting, if the "
                "learned temperature T < 1.5, force T = max(T, 1.5) as a "
                "floor. Alternatively, in models.py after calibration, "
                "apply: probs = probs ** (1/T) / sum(probs ** (1/T)) with "
                "T ~1.5-2.0. Or switch binary markets to IsotonicRegression "
                "which is more robust than parametric approaches."
            ),
            expected_impact=(
                "Significant — fixes the main accuracy issue. Overconfident "
                "predictions at 80%+ would drop to 65-70%, matching actual "
                "hit rates and improving Brier scores by 10-20%."
            ),
        ))

    # --- Recommendation 2: Per-market isotonic calibration ---
    if high_brier_markets:
        recs.append(ModelRecommendation(
            name="Switch to isotonic calibration for binary markets",
            description=(
                f"Markets with high Brier scores: {', '.join(high_brier_markets[:5])}. "
                "Isotonic regression is non-parametric and adapts better "
                "to non-uniform miscalibration patterns than temperature "
                "or Platt scaling."
            ),
            priority="high",
            affected_markets=high_brier_markets,
            implementation_hint=(
                "calibration.py:get_calibrator_for_market() — for binary "
                "markets, replace BetaCalibrator with sklearn's "
                "CalibratedClassifierCV(method='isotonic') or use the "
                "existing IsotonicOrdinalCalibrator with method='adjacent'. "
                "Requires a held-out calibration fold (already done in "
                "models.py with the validation split)."
            ),
            expected_impact="5-15% Brier score improvement on affected markets",
        ))

    # --- Recommendation 3: Increase DC blend weight ---
    result_markets_weak = [
        m for m in ["1X2", "DC_1X", "DC_12", "DC_X2", "DNB_H", "DNB_A"]
        if m in low_accuracy_markets
    ]
    if result_markets_weak:
        recs.append(ModelRecommendation(
            name="Increase Dixon-Coles blend weight for result markets",
            description=(
                f"Result-based markets ({', '.join(result_markets_weak)}) "
                "have low accuracy. The Dixon-Coles model provides a "
                "strong statistical prior for match results. Increasing "
                "its blend weight (e.g. from 0.3 to 0.5) may help."
            ),
            priority="high",
            affected_markets=result_markets_weak,
            implementation_hint=(
                "blending.py — when computing final probabilities, increase "
                "the DC weight for y_1X2 and related targets. In "
                "predict.py:_blend_dc_ml(), change dc_weight from 0.3 to "
                "0.4-0.5 for 1X2/DC/DNB targets."
            ),
            expected_impact="~2-4% accuracy gain on match result prediction",
        ))

    # --- Recommendation 4: Drop unprofitable markets ---
    if low_accuracy_markets:
        truly_bad = [
            m for m in low_accuracy_markets
            if any(
                d.market == m and d.overall_accuracy < 0.40
                for d in diagnoses
            )
        ]
        if truly_bad:
            recs.append(ModelRecommendation(
                name="Drop or disable very low accuracy markets",
                description=(
                    f"Markets with <40% accuracy: {', '.join(truly_bad)}. "
                    "These are actively harmful to include in accumulators "
                    "and top-50 picks. Either disable them or exclude them "
                    "from accumulator/top-50 selection."
                ),
                priority="high",
                affected_markets=truly_bad,
                implementation_hint=(
                    "speed_config.py:should_train_market() — add these "
                    "markets to a DISABLED_MARKETS set. Or in "
                    "weighted_top50.py, filter out picks from these markets."
                ),
                expected_impact=(
                    "Prevents bad picks from contaminating accumulators. "
                    "Indirect accuracy gain by focusing resources."
                ),
            ))

    # --- Recommendation 5: League-specific model weighting ---
    weak_league_markets = []
    for d in diagnoses:
        for lg, acc in d.league_accuracies.items():
            if acc < 0.35 and d.total_with_results >= MIN_SAMPLES:
                weak_league_markets.append((d.market, lg, acc))

    if weak_league_markets:
        unique_leagues = sorted(set(lg for _, lg, _ in weak_league_markets))
        recs.append(ModelRecommendation(
            name="Add league-specific confidence dampening",
            description=(
                f"Some leagues have very low accuracy across markets: "
                f"{', '.join(unique_leagues[:5])}. Apply a confidence "
                "dampening factor for these leagues."
            ),
            priority="medium",
            affected_markets=list(set(m for m, _, _ in weak_league_markets)),
            implementation_hint=(
                "predict.py — in the league adjustment step, for leagues "
                "in a DAMPEN_LEAGUES set, multiply all probabilities by a "
                "shrinkage factor toward 0.5: p_adj = p * (1-alpha) + "
                "0.5 * alpha, where alpha ~0.2-0.3 for weak leagues."
            ),
            expected_impact="Reduces overconfidence in unpredictable leagues",
        ))

    # --- Recommendation 6: Ensemble diversity ---
    try:
        from speed_config import get_speed_config, get_speed_mode
        cfg = get_speed_config()
        if len(cfg.models) == 1:
            recs.append(ModelRecommendation(
                name="Add a second model type for ensemble diversity",
                description=(
                    f"Currently using single model ({cfg.models[0]}). "
                    "Adding a second model (e.g. XGBoost or CatBoost) "
                    "and averaging predictions reduces variance and "
                    "improves calibration."
                ),
                priority="medium",
                affected_markets=["all"],
                implementation_hint=(
                    "speed_config.py — change BALANCED mode models from "
                    '["lgb"] to ["lgb", "xgb"]. The stacking meta-learner '
                    "in models.py will handle blending."
                ),
                expected_impact="~1-2% accuracy gain from ensemble effect",
            ))
    except Exception:
        pass

    # --- Recommendation 7: Cross-validation strategy ---
    recs.append(ModelRecommendation(
        name="Use time-series CV instead of random CV",
        description=(
            "Football data is temporal — random cross-validation leaks "
            "future information. If not already using time-based splits, "
            "switching to expanding-window CV prevents overly optimistic "
            "training metrics that cause overconfidence."
        ),
        priority="medium",
        affected_markets=["all"],
        implementation_hint=(
            "models.py / tuning.py:make_time_split() — ensure the CV "
            "split always trains on past data and validates on future. "
            "The current PredefinedSplit may not enforce temporal ordering. "
            "Use sklearn TimeSeriesSplit or a custom date-based splitter."
        ),
        expected_impact="Better calibrated predictions, fewer surprises",
    ))

    # --- Recommendation 8: Probability clipping ---
    severe_overconf = [
        d.market for d in diagnoses
        if d.overconfidence_score < -0.15
    ]
    if severe_overconf:
        recs.append(ModelRecommendation(
            name="Apply probability clipping (cap at 85%)",
            description=(
                f"Markets with severe overconfidence: {', '.join(severe_overconf)}. "
                "As an immediate fix, clip all output probabilities to "
                "a maximum of 85%. No football market truly has 95%+ "
                "predictability given variance in the sport."
            ),
            priority="high",
            affected_markets=severe_overconf,
            implementation_hint=(
                "predict.py — after all calibration steps, add: "
                "probs = np.clip(probs, 0.05, 0.85) then re-normalise. "
                "This is a blunt instrument but prevents the worst "
                "overconfidence while better calibration is developed."
            ),
            expected_impact=(
                "Immediate Brier score improvement. Prevents 90%+ "
                "confident predictions that hit at 40%."
            ),
        ))

    return recs


# ---------------------------------------------------------------------------
# Configuration analysis
# ---------------------------------------------------------------------------

def analyse_configuration() -> List[str]:
    """Check config for known issues and recommend changes."""
    config_recs = []

    # Check FORM_WINDOWS
    if FORM_WINDOWS != [3, 5, 10, 20]:
        config_recs.append(
            f"FORM_WINDOWS is {FORM_WINDOWS} — recommend [3, 5, 10, 20] "
            "for comprehensive form capture."
        )

    # Check feature parquet freshness
    try:
        feat_path = Path(FEATURES_PARQUET)
        if feat_path.exists():
            age_days = (datetime.now() - datetime.fromtimestamp(
                feat_path.stat().st_mtime
            )).days
            if age_days > 14:
                config_recs.append(
                    f"features.parquet is {age_days} days old — consider "
                    "rebuilding to capture recent matches."
                )
    except Exception:
        pass

    # Check model freshness
    model_dir = Path(MODEL_ARTIFACTS_DIR)
    if model_dir.exists():
        joblib_files = list(model_dir.glob("*.joblib"))
        if joblib_files:
            newest = max(f.stat().st_mtime for f in joblib_files)
            age_days = (datetime.now() - datetime.fromtimestamp(newest)).days
            if age_days > 14:
                config_recs.append(
                    f"Models are {age_days} days old — retrain to pick up "
                    "new features and recent form data."
                )
        else:
            config_recs.append(
                "No trained models found — run the pipeline to train."
            )

    # Check database size
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE correct IS NOT NULL"
            ).fetchone()[0]
            if count < 100:
                config_recs.append(
                    f"Only {count} resolved predictions in the accuracy DB. "
                    "Analysis will improve with more data — keep running "
                    "the weekly pipeline and updating results."
                )
        except Exception:
            pass
        finally:
            conn.close()

    return config_recs


# ---------------------------------------------------------------------------
# Auto-fix: generate a patch file with safe calibration changes
# ---------------------------------------------------------------------------

def generate_calibration_patch(
    diagnoses: List[MarketDiagnosis],
    output_path: Path,
) -> Optional[Path]:
    """
    Write a JSON config file that can be loaded by predict.py to apply
    per-market calibration overrides without changing code.
    """
    overrides = {}

    for d in diagnoses:
        if d.total_with_results < MIN_SAMPLES:
            continue

        entry = {}

        # Probability cap
        if d.overconfidence_score < -0.10:
            entry["max_probability"] = 0.80
        elif d.overconfidence_score < -0.05:
            entry["max_probability"] = 0.85

        # Temperature suggestion
        if d.overconfidence_score < -0.15:
            entry["temperature"] = 2.0
        elif d.overconfidence_score < -0.08:
            entry["temperature"] = 1.5

        # League dampening
        dampen_leagues = {}
        for lg, acc in d.league_accuracies.items():
            if acc < 0.35:
                dampen_leagues[lg] = 0.3  # heavy dampening
            elif acc < 0.42:
                dampen_leagues[lg] = 0.15  # moderate dampening
        if dampen_leagues:
            entry["league_dampening"] = dampen_leagues

        if entry:
            overrides[d.market] = entry

    if not overrides:
        return None

    patch = {
        "generated_at": datetime.now().isoformat(),
        "description": (
            "Auto-generated calibration overrides from prediction_accuracy_analyzer. "
            "Load this in predict.py to apply per-market probability caps, "
            "temperature adjustments, and league dampening."
        ),
        "market_overrides": overrides,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(patch, f, indent=2)

    print(f"[OK] Calibration patch written to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(report: AnalysisReport) -> None:
    """Pretty-print the analysis report to stdout."""
    print("\n" + "=" * 70)
    print("  PREDICTION ACCURACY ANALYSIS REPORT")
    print("=" * 70)
    print(f"  Generated: {report.generated_at}")
    print(f"  Total predictions: {report.total_predictions}")
    print(f"  With results: {report.total_with_results}")
    print(f"  Overall accuracy: {report.overall_accuracy:.1%}")
    print(f"  Overall Brier score: {report.overall_brier:.4f}")

    # --- Market Summary Table ---
    print("\n" + "-" * 70)
    print("  MARKET ACCURACY SUMMARY")
    print("-" * 70)
    print(f"  {'Market':<18} {'Accuracy':>8} {'Brier':>8} {'Overconf':>10} "
          f"{'N':>6}  Issues")
    print(f"  {'------':<18} {'--------':>8} {'-----':>8} {'--------':>10} "
          f"{'---':>6}  ------")

    for d in sorted(report.market_diagnoses,
                    key=lambda x: x.overall_accuracy, reverse=True):
        if d.total_with_results < MIN_SAMPLES:
            continue
        issue_flag = " !!!" if d.issues else ""
        print(
            f"  {d.market:<18} {d.overall_accuracy:>7.1%} "
            f"{d.brier_score:>8.4f} {d.overconfidence_score:>+9.1%} "
            f"{d.total_with_results:>6}{issue_flag}"
        )

    # --- Calibration Detail ---
    print("\n" + "-" * 70)
    print("  CALIBRATION ANALYSIS (predicted confidence vs actual accuracy)")
    print("-" * 70)

    # Aggregate calibration across all markets
    for d in report.market_diagnoses:
        if not d.calibration_buckets or d.total_with_results < MIN_SAMPLES:
            continue
        print(f"\n  {d.market}:")
        print(f"    {'Confidence':>12} {'Predicted':>10} {'Actual':>8} "
              f"{'Gap':>8} {'N':>6}")
        for b in d.calibration_buckets:
            gap_marker = " <<" if b.gap < -0.10 else ""
            print(
                f"    {int(b.confidence_lo*100):>3}-{int(b.confidence_hi*100)-1:>3}%"
                f"    {b.expected_accuracy:>7.1%}  {b.accuracy:>7.1%} "
                f"{b.gap:>+7.1%} {b.n_predictions:>6}{gap_marker}"
            )

    # --- Issues ---
    all_issues = []
    for d in report.market_diagnoses:
        for issue in d.issues:
            all_issues.append(f"[{d.market}] {issue}")

    if all_issues:
        print("\n" + "-" * 70)
        print("  ISSUES DETECTED")
        print("-" * 70)
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")

    # --- Feature Recommendations ---
    if report.feature_recommendations:
        print("\n" + "-" * 70)
        print("  FEATURE ENGINEERING RECOMMENDATIONS")
        print("-" * 70)
        for i, rec in enumerate(report.feature_recommendations, 1):
            print(f"\n  {i}. [{rec.priority.upper()}] {rec.name}")
            print(f"     {rec.description}")
            print(f"     Markets: {', '.join(rec.affected_markets)}")
            print(f"     How: {rec.implementation_hint}")
            print(f"     Impact: {rec.expected_impact}")

    # --- Model Recommendations ---
    if report.model_recommendations:
        print("\n" + "-" * 70)
        print("  MODEL & CALIBRATION RECOMMENDATIONS")
        print("-" * 70)
        for i, rec in enumerate(report.model_recommendations, 1):
            print(f"\n  {i}. [{rec.priority.upper()}] {rec.name}")
            print(f"     {rec.description}")
            print(f"     Markets: {', '.join(rec.affected_markets[:5])}")
            print(f"     How: {rec.implementation_hint}")
            print(f"     Impact: {rec.expected_impact}")

    # --- Config Recommendations ---
    if report.config_recommendations:
        print("\n" + "-" * 70)
        print("  CONFIGURATION RECOMMENDATIONS")
        print("-" * 70)
        for i, rec in enumerate(report.config_recommendations, 1):
            print(f"  {i}. {rec}")

    print("\n" + "=" * 70)
    print("  END OF REPORT")
    print("=" * 70)


def save_html_report(report: AnalysisReport, output_path: Path) -> Path:
    """Generate an HTML version of the report."""
    html_parts = [f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Prediction Accuracy Analysis</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial;
               padding: 20px; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white;
                     padding: 30px; border-radius: 10px;
                     box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; }}
        h3 {{ color: #34495e; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                 gap: 15px; margin: 20px 0; }}
        .stat-box {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white;
                    padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-box .number {{ font-size: 2em; font-weight: bold; }}
        .stat-box .label {{ font-size: 0.85em; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }}
        th {{ background: #f8f9fa; color: #2c3e50; padding: 10px 8px; text-align: left;
             border-bottom: 2px solid #dee2e6; }}
        td {{ padding: 8px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}
        .good {{ color: #27ae60; font-weight: bold; }}
        .bad {{ color: #e74c3c; font-weight: bold; }}
        .warn {{ color: #f39c12; font-weight: bold; }}
        .rec-card {{ background: #f8f9fa; border-left: 4px solid #3498db;
                    padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }}
        .rec-card.high {{ border-left-color: #e74c3c; }}
        .rec-card.medium {{ border-left-color: #f39c12; }}
        .rec-card.low {{ border-left-color: #27ae60; }}
        .rec-card h4 {{ margin: 0 0 8px 0; }}
        .rec-card .hint {{ background: #eee; padding: 8px; border-radius: 4px;
                          font-family: monospace; font-size: 0.85em; margin-top: 8px; }}
        .issue {{ background: #fff3cd; padding: 8px 12px; margin: 5px 0;
                 border-radius: 4px; border-left: 3px solid #ffc107; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
                 font-size: 0.8em; font-weight: bold; color: white; }}
        .badge.high {{ background: #e74c3c; }}
        .badge.medium {{ background: #f39c12; }}
        .badge.low {{ background: #27ae60; }}
    </style>
</head>
<body>
<div class='container'>
    <h1>Prediction Accuracy Analysis</h1>
    <p>Generated: {report.generated_at}</p>

    <div class='stats'>
        <div class='stat-box'>
            <div class='number'>{report.total_predictions:,}</div>
            <div class='label'>Total Predictions</div>
        </div>
        <div class='stat-box'>
            <div class='number'>{report.total_with_results:,}</div>
            <div class='label'>With Results</div>
        </div>
        <div class='stat-box'>
            <div class='number'>{report.overall_accuracy:.1%}</div>
            <div class='label'>Overall Accuracy</div>
        </div>
        <div class='stat-box'>
            <div class='number'>{report.overall_brier:.4f}</div>
            <div class='label'>Brier Score</div>
        </div>
    </div>
"""]

    # Market summary table
    html_parts.append("<h2>Market Accuracy Summary</h2><table>")
    html_parts.append(
        "<tr><th>Market</th><th>Accuracy</th><th>Brier</th>"
        "<th>Overconfidence</th><th>N</th><th>Issues</th></tr>"
    )
    for d in sorted(report.market_diagnoses,
                    key=lambda x: x.overall_accuracy, reverse=True):
        if d.total_with_results < MIN_SAMPLES:
            continue
        acc_cls = "good" if d.overall_accuracy >= 0.55 else (
            "bad" if d.overall_accuracy < 0.45 else "warn")
        oc_cls = "bad" if d.overconfidence_score < -0.10 else (
            "warn" if d.overconfidence_score < -0.05 else "good")
        issue_text = "; ".join(d.issues) if d.issues else "-"
        html_parts.append(
            f"<tr><td><strong>{d.market}</strong></td>"
            f"<td class='{acc_cls}'>{d.overall_accuracy:.1%}</td>"
            f"<td>{d.brier_score:.4f}</td>"
            f"<td class='{oc_cls}'>{d.overconfidence_score:+.1%}</td>"
            f"<td>{d.total_with_results}</td>"
            f"<td>{issue_text}</td></tr>"
        )
    html_parts.append("</table>")

    # Calibration tables
    html_parts.append("<h2>Calibration Analysis</h2>")
    html_parts.append(
        "<p>Compares predicted confidence with actual hit rate. "
        "Negative gap = overconfident.</p>"
    )
    for d in report.market_diagnoses:
        if not d.calibration_buckets or d.total_with_results < MIN_SAMPLES:
            continue
        html_parts.append(f"<h3>{d.market}</h3><table>")
        html_parts.append(
            "<tr><th>Confidence</th><th>Expected</th><th>Actual</th>"
            "<th>Gap</th><th>N</th></tr>"
        )
        for b in d.calibration_buckets:
            gap_cls = "bad" if b.gap < -0.10 else (
                "warn" if b.gap < -0.05 else "good")
            html_parts.append(
                f"<tr><td>{int(b.confidence_lo*100)}-{int(b.confidence_hi*100)-1}%</td>"
                f"<td>{b.expected_accuracy:.1%}</td>"
                f"<td>{b.accuracy:.1%}</td>"
                f"<td class='{gap_cls}'>{b.gap:+.1%}</td>"
                f"<td>{b.n_predictions}</td></tr>"
            )
        html_parts.append("</table>")

    # Issues
    all_issues = []
    for d in report.market_diagnoses:
        for issue in d.issues:
            all_issues.append(f"[{d.market}] {issue}")
    if all_issues:
        html_parts.append("<h2>Issues Detected</h2>")
        for issue in all_issues:
            html_parts.append(f"<div class='issue'>{issue}</div>")

    # Feature recommendations
    if report.feature_recommendations:
        html_parts.append("<h2>Feature Engineering Recommendations</h2>")
        for rec in report.feature_recommendations:
            html_parts.append(
                f"<div class='rec-card {rec.priority}'>"
                f"<h4><span class='badge {rec.priority}'>{rec.priority.upper()}</span> "
                f"{rec.name}</h4>"
                f"<p>{rec.description}</p>"
                f"<p><strong>Affected markets:</strong> {', '.join(rec.affected_markets)}</p>"
                f"<div class='hint'>{rec.implementation_hint}</div>"
                f"<p><strong>Expected impact:</strong> {rec.expected_impact}</p>"
                f"</div>"
            )

    # Model recommendations
    if report.model_recommendations:
        html_parts.append("<h2>Model &amp; Calibration Recommendations</h2>")
        for rec in report.model_recommendations:
            html_parts.append(
                f"<div class='rec-card {rec.priority}'>"
                f"<h4><span class='badge {rec.priority}'>{rec.priority.upper()}</span> "
                f"{rec.name}</h4>"
                f"<p>{rec.description}</p>"
                f"<p><strong>Affected markets:</strong> {', '.join(rec.affected_markets[:5])}</p>"
                f"<div class='hint'>{rec.implementation_hint}</div>"
                f"<p><strong>Expected impact:</strong> {rec.expected_impact}</p>"
                f"</div>"
            )

    # Config recommendations
    if report.config_recommendations:
        html_parts.append("<h2>Configuration Recommendations</h2>")
        html_parts.append("<ul>")
        for rec in report.config_recommendations:
            html_parts.append(f"<li>{rec}</li>")
        html_parts.append("</ul>")

    html_parts.append("</div></body></html>")

    html = "\n".join(html_parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"[OK] HTML report saved: {output_path}")
    return output_path


def save_json_report(report: AnalysisReport, output_path: Path) -> Path:
    """Save a machine-readable JSON version of the report."""
    data = {
        "generated_at": report.generated_at,
        "total_predictions": report.total_predictions,
        "total_with_results": report.total_with_results,
        "overall_accuracy": report.overall_accuracy,
        "overall_brier": report.overall_brier,
        "market_diagnoses": [
            {
                "market": d.market,
                "total_predictions": d.total_predictions,
                "total_with_results": d.total_with_results,
                "overall_accuracy": d.overall_accuracy,
                "brier_score": d.brier_score,
                "overconfidence_score": d.overconfidence_score,
                "league_accuracies": d.league_accuracies,
                "worst_leagues": d.worst_leagues,
                "best_leagues": d.best_leagues,
                "issues": d.issues,
                "recommendations": d.recommendations,
                "calibration_buckets": [
                    {
                        "confidence_range": f"{int(b.confidence_lo*100)}-{int(b.confidence_hi*100)-1}%",
                        "n_predictions": b.n_predictions,
                        "accuracy": b.accuracy,
                        "expected": b.expected_accuracy,
                        "gap": b.gap,
                    }
                    for b in d.calibration_buckets
                ],
            }
            for d in report.market_diagnoses
        ],
        "feature_recommendations": [asdict(r) for r in report.feature_recommendations],
        "model_recommendations": [asdict(r) for r in report.model_recommendations],
        "config_recommendations": report.config_recommendations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] JSON report saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_analysis(
    market_filter: Optional[str] = None,
    apply_fixes: bool = False,
) -> AnalysisReport:
    """Run the full analysis pipeline."""
    log_header("PREDICTION ACCURACY ANALYZER")

    # 1. Load predictions
    df = load_predictions()

    if df.empty:
        print("\n[INFO] No predictions to analyse.")
        print("       Run the weekly pipeline and update results first.")
        print("       Pipeline steps: run_weekly.py -> update_results.py")
        return AnalysisReport(
            generated_at=datetime.now().isoformat(),
            total_predictions=0,
            total_with_results=0,
            overall_accuracy=0.0,
            overall_brier=0.0,
            market_diagnoses=[],
            feature_recommendations=analyse_features(),
            model_recommendations=[],
            config_recommendations=analyse_configuration(),
        )

    # Filter to market if requested
    if market_filter:
        df = df[df["market"].str.contains(market_filter, case=False, na=False)]
        if df.empty:
            print(f"[WARN] No predictions found for market '{market_filter}'")
            return None

    # 2. Overall stats
    with_results = df[df["correct"].notna()].copy()
    total_predictions = len(df)
    total_with_results = len(with_results)

    if total_with_results > 0:
        overall_accuracy = round(with_results["correct"].mean(), 4)
        prob = with_results["predicted_probability"].values
        correct = with_results["correct"].values.astype(float)
        overall_brier = round(float(np.mean((prob - correct) ** 2)), 4)
    else:
        overall_accuracy = 0.0
        overall_brier = 0.0

    print(f"\n[STATS] {total_predictions} predictions, "
          f"{total_with_results} with results")
    if total_with_results > 0:
        print(f"[STATS] Overall accuracy: {overall_accuracy:.1%}")
        print(f"[STATS] Overall Brier score: {overall_brier:.4f}")

    # 3. Per-market diagnosis
    print("\n[ANALYSIS] Diagnosing markets...")
    diagnoses = []
    for market_name, group in with_results.groupby("market"):
        diag = diagnose_market(group, market_name)
        diagnoses.append(diag)

    # Also diagnose markets with only pending predictions
    pending_markets = set(df["market"].unique()) - set(with_results["market"].unique())
    for market_name in pending_markets:
        group = df[df["market"] == market_name]
        diag = MarketDiagnosis(
            market=market_name,
            total_predictions=len(group),
            total_with_results=0,
        )
        diag.issues.append("No results yet — all predictions pending")
        diagnoses.append(diag)

    print(f"[OK] Diagnosed {len(diagnoses)} markets")

    # 4. Feature analysis
    print("\n[ANALYSIS] Analysing feature set...")
    feature_recs = analyse_features()
    print(f"[OK] {len(feature_recs)} feature recommendations")

    # 5. Model & calibration analysis
    print("\n[ANALYSIS] Analysing models and calibration...")
    model_recs = analyse_models_and_calibration(diagnoses)
    print(f"[OK] {len(model_recs)} model recommendations")

    # 6. Configuration analysis
    print("\n[ANALYSIS] Checking configuration...")
    config_recs = analyse_configuration()
    print(f"[OK] {len(config_recs)} config recommendations")

    # 7. Build report
    report = AnalysisReport(
        generated_at=datetime.now().isoformat(),
        total_predictions=total_predictions,
        total_with_results=total_with_results,
        overall_accuracy=overall_accuracy,
        overall_brier=overall_brier,
        market_diagnoses=diagnoses,
        feature_recommendations=feature_recs,
        model_recommendations=model_recs,
        config_recommendations=config_recs,
    )

    # 8. Output
    print_report(report)

    # Save reports
    report_dir = REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    save_html_report(report, report_dir / "accuracy_analysis.html")
    save_json_report(report, report_dir / "accuracy_analysis.json")

    # 9. Generate calibration patch if requested
    if apply_fixes:
        patch_path = MODEL_ARTIFACTS_DIR / "calibration_overrides.json"
        result = generate_calibration_patch(diagnoses, patch_path)
        if result:
            print(f"\n[AUTO-FIX] Calibration overrides written to {result}")
            print("           predict.py will load these on next run")
            print("           (if it supports calibration_overrides.json)")
        else:
            print("\n[AUTO-FIX] No calibration overrides needed")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse prediction accuracy and recommend improvements"
    )
    parser.add_argument(
        "--market", type=str, default=None,
        help="Filter to a specific market (e.g., '1X2', 'BTTS', 'OU_2_5')"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Generate and save calibration overrides for auto-fix"
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Path to accuracy database (default: outputs/accuracy_database.db)"
    )
    args = parser.parse_args()

    if args.db:
        global DB_PATH
        DB_PATH = Path(args.db)

    report = run_analysis(
        market_filter=args.market,
        apply_fixes=args.apply,
    )

    if report and report.total_with_results > 0:
        # Print summary priority action items
        print("\n" + "=" * 70)
        print("  PRIORITY ACTION ITEMS")
        print("=" * 70)

        high_items = []
        for rec in report.feature_recommendations:
            if rec.priority == "high":
                high_items.append(f"[FEATURE] {rec.name}")
        for rec in report.model_recommendations:
            if rec.priority == "high":
                high_items.append(f"[MODEL] {rec.name}")

        if high_items:
            for i, item in enumerate(high_items, 1):
                print(f"  {i}. {item}")
        else:
            print("  No high-priority items — system looks healthy!")

        print("=" * 70)

    elif report and report.total_with_results == 0:
        print("\n[INFO] Run the weekly pipeline and update results to get accuracy data:")
        print("       1. python run_weekly.py --speed balanced --non-interactive")
        print("       2. (wait for matches to complete)")
        print("       3. python update_results.py")
        print("       4. python prediction_accuracy_analyzer.py")


if __name__ == "__main__":
    main()
