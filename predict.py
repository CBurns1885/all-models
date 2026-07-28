#!/usr/bin/env python3
"""
ULTIMATE predict.py - Maximum Accuracy Prediction Engine
Combines:
- League-specific calibration
- Cross-market mathematical constraints
- Poisson statistical adjustments
- Time-weighted recent form
- Dynamic blend weights by league quality
- Confidence scoring with model agreement
- Enhanced HTML reporting
"""

import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from scipy.stats import poisson
from datetime import datetime, timedelta

from config import FEATURES_PARQUET, OUTPUT_DIR, MODEL_ARTIFACTS_DIR, ALL_CUPS, EUROPEAN_CUPS, log_header
from models import load_trained_targets, predict_proba as model_predict
from dc_predict import build_dc_for_fixtures
from nb_predict import build_nb_for_frame
from models_counts import nb_supported
from progress_utils import heartbeat
from blending import BLEND_WEIGHTS_JSON

ID_COLS = ["League","Date","HomeTeam","AwayTeam"]
OU_LINES = ["0_5","1_5","2_5","3_5","4_5"]
AH_LINES = ["-1_0","-0_5","0_0","+0_5","+1_0"]

# ---------------------------------------------------------------------------
# TUNING_OVERRIDES -- auto_tune.py writes into this dict at runtime.
# When empty, every parameter falls back to its hardcoded default.
# ---------------------------------------------------------------------------
TUNING_OVERRIDES: dict = {}

# Auto-load tuned params if file exists (written by auto_tune.py)
_tuned_params_path = Path(__file__).resolve().parent / "outputs" / "tuning_best_params.json"
if _tuned_params_path.exists():
    try:
        with open(_tuned_params_path) as _f:
            TUNING_OVERRIDES.update(json.load(_f))
    except (json.JSONDecodeError, ValueError):
        pass

# League scoring profiles (learned from historical data)
# style: 'attacking' (>2.8 avg), 'balanced' (2.5-2.8), 'defensive' (<2.5)
# clean_sheet_rate: Probability of at least one team keeping a clean sheet
LEAGUE_PROFILES = {
    # England
    'E0': {'avg_goals': 2.72, 'home_adv': 0.12, 'btts_rate': 0.53, 'over25_rate': 0.52, 'over15_rate': 0.78, 'over35_rate': 0.28, 'over45_rate': 0.12, 'clean_sheet_rate': 0.47, 'quality': 'elite', 'style': 'balanced'},
    'E1': {'avg_goals': 2.65, 'home_adv': 0.10, 'btts_rate': 0.51, 'over25_rate': 0.50, 'over15_rate': 0.76, 'over35_rate': 0.26, 'over45_rate': 0.10, 'clean_sheet_rate': 0.49, 'quality': 'high', 'style': 'balanced'},
    'E2': {'avg_goals': 2.58, 'home_adv': 0.11, 'btts_rate': 0.49, 'over25_rate': 0.48, 'over15_rate': 0.74, 'over35_rate': 0.24, 'over45_rate': 0.09, 'clean_sheet_rate': 0.51, 'quality': 'medium', 'style': 'balanced'},
    'E3': {'avg_goals': 2.61, 'home_adv': 0.13, 'btts_rate': 0.50, 'over25_rate': 0.49, 'over15_rate': 0.75, 'over35_rate': 0.25, 'over45_rate': 0.10, 'clean_sheet_rate': 0.50, 'quality': 'medium', 'style': 'balanced'},
    'EC': {'avg_goals': 2.65, 'home_adv': 0.12, 'btts_rate': 0.51, 'over25_rate': 0.50, 'over15_rate': 0.76, 'over35_rate': 0.26, 'over45_rate': 0.10, 'clean_sheet_rate': 0.49, 'quality': 'medium', 'style': 'balanced'},

    # Spain (more defensive, tactical)
    'SP1': {'avg_goals': 2.48, 'home_adv': 0.15, 'btts_rate': 0.46, 'over25_rate': 0.45, 'over15_rate': 0.70, 'over35_rate': 0.20, 'over45_rate': 0.07, 'clean_sheet_rate': 0.54, 'quality': 'elite', 'style': 'defensive'},
    'SP2': {'avg_goals': 2.35, 'home_adv': 0.14, 'btts_rate': 0.43, 'over25_rate': 0.41, 'over15_rate': 0.67, 'over35_rate': 0.17, 'over45_rate': 0.06, 'clean_sheet_rate': 0.57, 'quality': 'high', 'style': 'defensive'},

    # Italy (tactically aware, balanced)
    'I1': {'avg_goals': 2.68, 'home_adv': 0.11, 'btts_rate': 0.52, 'over25_rate': 0.51, 'over15_rate': 0.77, 'over35_rate': 0.27, 'over45_rate': 0.11, 'clean_sheet_rate': 0.48, 'quality': 'elite', 'style': 'balanced'},
    'I2': {'avg_goals': 2.45, 'home_adv': 0.12, 'btts_rate': 0.47, 'over25_rate': 0.44, 'over15_rate': 0.69, 'over35_rate': 0.19, 'over45_rate': 0.07, 'clean_sheet_rate': 0.53, 'quality': 'high', 'style': 'defensive'},

    # Germany (high scoring, end-to-end)
    'D1': {'avg_goals': 3.05, 'home_adv': 0.09, 'btts_rate': 0.58, 'over25_rate': 0.60, 'over15_rate': 0.85, 'over35_rate': 0.35, 'over45_rate': 0.18, 'clean_sheet_rate': 0.42, 'quality': 'elite', 'style': 'attacking'},
    'D2': {'avg_goals': 2.85, 'home_adv': 0.10, 'btts_rate': 0.55, 'over25_rate': 0.56, 'over15_rate': 0.82, 'over35_rate': 0.32, 'over45_rate': 0.15, 'clean_sheet_rate': 0.45, 'quality': 'high', 'style': 'attacking'},

    # France (physical, moderate scoring)
    'F1': {'avg_goals': 2.55, 'home_adv': 0.13, 'btts_rate': 0.48, 'over25_rate': 0.47, 'over15_rate': 0.73, 'over35_rate': 0.23, 'over45_rate': 0.09, 'clean_sheet_rate': 0.52, 'quality': 'elite', 'style': 'balanced'},
    'F2': {'avg_goals': 2.42, 'home_adv': 0.12, 'btts_rate': 0.45, 'over25_rate': 0.43, 'over15_rate': 0.68, 'over35_rate': 0.18, 'over45_rate': 0.06, 'clean_sheet_rate': 0.55, 'quality': 'high', 'style': 'defensive'},

    # Netherlands (attacking football culture)
    'N1': {'avg_goals': 2.95, 'home_adv': 0.08, 'btts_rate': 0.60, 'over25_rate': 0.59, 'over15_rate': 0.84, 'over35_rate': 0.34, 'over45_rate': 0.16, 'clean_sheet_rate': 0.40, 'quality': 'high', 'style': 'attacking'},

    # Belgium (high scoring, open games)
    'B1': {'avg_goals': 2.78, 'home_adv': 0.10, 'btts_rate': 0.54, 'over25_rate': 0.53, 'over15_rate': 0.79, 'over35_rate': 0.29, 'over45_rate': 0.13, 'clean_sheet_rate': 0.46, 'quality': 'high', 'style': 'balanced'},

    # Portugal (tactical, strong home advantage)
    'P1': {'avg_goals': 2.52, 'home_adv': 0.16, 'btts_rate': 0.47, 'over25_rate': 0.46, 'over15_rate': 0.72, 'over35_rate': 0.22, 'over45_rate': 0.08, 'clean_sheet_rate': 0.53, 'quality': 'high', 'style': 'defensive'},

    # Scotland
    'SC0': {'avg_goals': 2.65, 'home_adv': 0.11, 'btts_rate': 0.51, 'over25_rate': 0.50, 'over15_rate': 0.76, 'over35_rate': 0.26, 'over45_rate': 0.10, 'clean_sheet_rate': 0.49, 'quality': 'high', 'style': 'balanced'},
    'SC1': {'avg_goals': 2.58, 'home_adv': 0.13, 'btts_rate': 0.49, 'over25_rate': 0.48, 'over15_rate': 0.74, 'over35_rate': 0.24, 'over45_rate': 0.09, 'clean_sheet_rate': 0.51, 'quality': 'medium', 'style': 'balanced'},

    # Turkey (high scoring, volatile)
    'T1': {'avg_goals': 3.10, 'home_adv': 0.14, 'btts_rate': 0.59, 'over25_rate': 0.61, 'over15_rate': 0.86, 'over35_rate': 0.36, 'over45_rate': 0.19, 'clean_sheet_rate': 0.41, 'quality': 'elite', 'style': 'attacking'},

    # Greece
    'G1': {'avg_goals': 2.35, 'home_adv': 0.18, 'btts_rate': 0.42, 'over25_rate': 0.40, 'over15_rate': 0.66, 'over35_rate': 0.16, 'over45_rate': 0.05, 'clean_sheet_rate': 0.58, 'quality': 'medium', 'style': 'defensive'},

    # Austria
    'A1': {'avg_goals': 2.92, 'home_adv': 0.10, 'btts_rate': 0.56, 'over25_rate': 0.58, 'over15_rate': 0.83, 'over35_rate': 0.33, 'over45_rate': 0.15, 'clean_sheet_rate': 0.44, 'quality': 'medium', 'style': 'attacking'},

    # Switzerland
    'SWZ': {'avg_goals': 2.75, 'home_adv': 0.09, 'btts_rate': 0.53, 'over25_rate': 0.52, 'over15_rate': 0.78, 'over35_rate': 0.28, 'over45_rate': 0.12, 'clean_sheet_rate': 0.47, 'quality': 'medium', 'style': 'balanced'},

    # Poland
    'POL': {'avg_goals': 2.62, 'home_adv': 0.12, 'btts_rate': 0.50, 'over25_rate': 0.49, 'over15_rate': 0.75, 'over35_rate': 0.25, 'over45_rate': 0.10, 'clean_sheet_rate': 0.50, 'quality': 'medium', 'style': 'balanced'},

    # Russia
    'RUS': {'avg_goals': 2.45, 'home_adv': 0.14, 'btts_rate': 0.46, 'over25_rate': 0.44, 'over15_rate': 0.69, 'over35_rate': 0.19, 'over45_rate': 0.07, 'clean_sheet_rate': 0.54, 'quality': 'medium', 'style': 'defensive'},

    # ========== DOMESTIC CUPS (knockout - more unpredictable) ==========
    # FA Cup (England) - giant killings common, high scoring
    'FAC': {'avg_goals': 2.85, 'home_adv': 0.08, 'btts_rate': 0.54, 'over25_rate': 0.55, 'over15_rate': 0.80, 'over35_rate': 0.30, 'over45_rate': 0.14, 'clean_sheet_rate': 0.46, 'quality': 'high', 'style': 'attacking', 'is_cup': True},

    # DFB Pokal (Germany) - similar to Bundesliga style
    'DFB': {'avg_goals': 3.15, 'home_adv': 0.06, 'btts_rate': 0.59, 'over25_rate': 0.62, 'over15_rate': 0.86, 'over35_rate': 0.38, 'over45_rate': 0.20, 'clean_sheet_rate': 0.41, 'quality': 'high', 'style': 'attacking', 'is_cup': True},

    # Copa del Rey (Spain) - tactical but with upsets
    'CDR': {'avg_goals': 2.55, 'home_adv': 0.10, 'btts_rate': 0.48, 'over25_rate': 0.48, 'over15_rate': 0.73, 'over35_rate': 0.23, 'over45_rate': 0.09, 'clean_sheet_rate': 0.52, 'quality': 'high', 'style': 'balanced', 'is_cup': True},

    # Coppa Italia (Italy) - conservative approach in cups
    'CIT': {'avg_goals': 2.58, 'home_adv': 0.09, 'btts_rate': 0.50, 'over25_rate': 0.49, 'over15_rate': 0.75, 'over35_rate': 0.25, 'over45_rate': 0.10, 'clean_sheet_rate': 0.50, 'quality': 'high', 'style': 'balanced', 'is_cup': True},

    # Coupe de France (France) - amateur teams cause upsets
    'CDF': {'avg_goals': 2.72, 'home_adv': 0.07, 'btts_rate': 0.51, 'over25_rate': 0.52, 'over15_rate': 0.77, 'over35_rate': 0.27, 'over45_rate': 0.11, 'clean_sheet_rate': 0.49, 'quality': 'medium', 'style': 'balanced', 'is_cup': True},

    # KNVB Beker (Netherlands) - Dutch attacking style
    'KNVB': {'avg_goals': 3.05, 'home_adv': 0.06, 'btts_rate': 0.61, 'over25_rate': 0.61, 'over15_rate': 0.85, 'over35_rate': 0.36, 'over45_rate': 0.18, 'clean_sheet_rate': 0.39, 'quality': 'medium', 'style': 'attacking', 'is_cup': True},

    # Belgian Cup
    'BEC': {'avg_goals': 2.88, 'home_adv': 0.08, 'btts_rate': 0.55, 'over25_rate': 0.56, 'over15_rate': 0.81, 'over35_rate': 0.31, 'over45_rate': 0.14, 'clean_sheet_rate': 0.45, 'quality': 'medium', 'style': 'balanced', 'is_cup': True},

    # Taça de Portugal
    'TCP': {'avg_goals': 2.68, 'home_adv': 0.12, 'btts_rate': 0.49, 'over25_rate': 0.50, 'over15_rate': 0.75, 'over35_rate': 0.26, 'over45_rate': 0.11, 'clean_sheet_rate': 0.51, 'quality': 'medium', 'style': 'balanced', 'is_cup': True},

    # Scottish FA Cup
    'SFC': {'avg_goals': 2.78, 'home_adv': 0.08, 'btts_rate': 0.52, 'over25_rate': 0.53, 'over15_rate': 0.78, 'over35_rate': 0.28, 'over45_rate': 0.12, 'clean_sheet_rate': 0.48, 'quality': 'medium', 'style': 'balanced', 'is_cup': True},

    # Turkish Cup
    'TFC': {'avg_goals': 3.18, 'home_adv': 0.10, 'btts_rate': 0.60, 'over25_rate': 0.63, 'over15_rate': 0.87, 'over35_rate': 0.38, 'over45_rate': 0.21, 'clean_sheet_rate': 0.40, 'quality': 'medium', 'style': 'attacking', 'is_cup': True},

    # ========== ADDITIONAL EUROPEAN LEAGUES ==========
    # Denmark Superliga
    'DEN': {'avg_goals': 2.82, 'home_adv': 0.10, 'btts_rate': 0.54, 'over25_rate': 0.55, 'over15_rate': 0.80, 'over35_rate': 0.30, 'over45_rate': 0.13, 'clean_sheet_rate': 0.46, 'quality': 'medium', 'style': 'attacking'},

    # Norway Eliteserien
    'NOR': {'avg_goals': 2.95, 'home_adv': 0.12, 'btts_rate': 0.57, 'over25_rate': 0.58, 'over15_rate': 0.83, 'over35_rate': 0.33, 'over45_rate': 0.16, 'clean_sheet_rate': 0.43, 'quality': 'medium', 'style': 'attacking'},

    # Sweden Allsvenskan
    'SWE': {'avg_goals': 2.78, 'home_adv': 0.11, 'btts_rate': 0.53, 'over25_rate': 0.53, 'over15_rate': 0.79, 'over35_rate': 0.29, 'over45_rate': 0.12, 'clean_sheet_rate': 0.47, 'quality': 'medium', 'style': 'balanced'},

    # Czech First League
    'CZE': {'avg_goals': 2.65, 'home_adv': 0.13, 'btts_rate': 0.50, 'over25_rate': 0.50, 'over15_rate': 0.76, 'over35_rate': 0.26, 'over45_rate': 0.10, 'clean_sheet_rate': 0.50, 'quality': 'medium', 'style': 'balanced'},

    # Croatia HNL
    'CRO': {'avg_goals': 2.72, 'home_adv': 0.14, 'btts_rate': 0.52, 'over25_rate': 0.52, 'over15_rate': 0.78, 'over35_rate': 0.28, 'over45_rate': 0.12, 'clean_sheet_rate': 0.48, 'quality': 'medium', 'style': 'balanced'},
}

def _load_base_features() -> pd.DataFrame:
    """Load historical features from parquet."""
    df = pd.read_parquet(FEATURES_PARQUET)
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["League", "Date"])

def calculate_league_profiles(df: pd.DataFrame) -> Dict:
    """Calculate league profiles from historical data.

    Derives all fields (avg_goals, style, quality, clean_sheet_rate, home_adv, etc.)
    from the data so any league — including new ones or European competitions — gets
    accurate calibration without manual static entries.
    """
    profiles = {}

    # Per-league avg Elo for quality tiering (percentile-based, fully data-driven)
    league_elo: Dict[str, float] = {}
    for league in df['League'].unique():
        ld = df[df['League'] == league]
        elo_cols = [c for c in ['Elo_Home', 'Elo_Away'] if c in ld.columns]
        if elo_cols:
            vals = pd.concat([ld[c] for c in elo_cols]).dropna()
            if len(vals) > 0:
                league_elo[league] = float(vals.mean())

    if league_elo:
        elo_vals = list(league_elo.values())
        q75 = np.percentile(elo_vals, 75)
        q50 = np.percentile(elo_vals, 50)
        q25 = np.percentile(elo_vals, 25)
    else:
        q75 = q50 = q25 = 1500.0

    # Global avg goals (used to classify style relative to the overall distribution)
    total_goals_global = (df['FTHG'].fillna(0) + df['FTAG'].fillna(0))
    global_avg = float(total_goals_global.mean()) if len(total_goals_global) > 0 else 2.7

    for league in df['League'].unique():
        league_data = df[df['League'] == league]
        if len(league_data) < 50:
            continue

        total_goals = league_data['FTHG'].fillna(0) + league_data['FTAG'].fillna(0)
        home_wins = (league_data['FTR'] == 'H').mean()
        away_wins = (league_data['FTR'] == 'A').mean()
        avg_goals = float(total_goals.mean())

        # Style: attacking/balanced/defensive relative to global average
        if avg_goals > global_avg * 1.08:
            style = 'attacking'
        elif avg_goals < global_avg * 0.92:
            style = 'defensive'
        else:
            style = 'balanced'

        # Quality: percentile-rank of avg Elo among all leagues
        avg_elo = league_elo.get(league)
        if avg_elo is not None:
            if avg_elo >= q75:
                quality = 'elite'
            elif avg_elo >= q50:
                quality = 'high'
            elif avg_elo >= q25:
                quality = 'medium'
            else:
                quality = 'low'
        else:
            quality = 'medium'

        # Clean sheet rate: proportion of matches where at least one team kept a clean sheet
        # Equivalent to 1 - btts_rate (consistent with static LEAGUE_PROFILES convention)
        clean_sheet_rate = float(((league_data['FTHG'] == 0) | (league_data['FTAG'] == 0)).mean())

        profiles[league] = {
            'avg_goals': avg_goals,
            'home_adv': float(home_wins - away_wins),
            'btts_rate': float(((league_data['FTHG'] > 0) & (league_data['FTAG'] > 0)).mean()),
            'over25_rate': float((total_goals > 2.5).mean()),
            'over15_rate': float((total_goals > 1.5).mean()),
            'over35_rate': float((total_goals > 3.5).mean()),
            'over45_rate': float((total_goals > 4.5).mean()),
            'style': style,
            'quality': quality,
            'clean_sheet_rate': clean_sheet_rate,
        }

    return profiles

def apply_league_calibration(prob: float, market: str, league: str, league_profiles: Dict) -> float:
    """Calibrate probability based on league-specific patterns and style."""
    if league not in league_profiles:
        return prob

    profile = league_profiles[league]
    style = profile.get('style', 'balanced')
    home_adv = profile.get('home_adv', 0.1)
    clean_sheet_rate = profile.get('clean_sheet_rate', 0.47)

    # Stronger calibration for lower confidence predictions
    confidence = abs(prob - 0.5) * 2  # 0 to 1 scale
    calibration_weight = TUNING_OVERRIDES.get('calibration_scalar', 0.3) * (1 - confidence)

    # Style-based adjustments
    _sb = TUNING_OVERRIDES.get('style_boost_magnitude', 0.08)
    style_boost = {'attacking': _sb, 'balanced': 0.0, 'defensive': -_sb}
    goal_style_adj = style_boost.get(style, 0.0)

    # ========== GOALS MARKETS ==========
    if 'BTTS_Y' in market:
        league_avg = profile.get('btts_rate', 0.5)
        calibrated = prob * (1 - calibration_weight) + league_avg * calibration_weight
        # Attacking leagues boost BTTS, defensive leagues reduce it
        return max(0.01, min(0.99, calibrated + goal_style_adj * 0.5))

    elif 'BTTS_N' in market:
        league_avg = 1 - profile.get('btts_rate', 0.5)
        calibrated = prob * (1 - calibration_weight) + league_avg * calibration_weight
        # Defensive leagues boost BTTS_N
        return max(0.01, min(0.99, calibrated - goal_style_adj * 0.5))

    elif 'OU_0_5_O' in market:
        return min(prob * 1.05, 0.99)  # Boost slightly (0.5 goals is very likely)

    elif 'OU_1_5_O' in market:
        league_avg = profile.get('over15_rate', 0.7)
        calibrated = prob * (1 - calibration_weight * 0.5) + league_avg * (calibration_weight * 0.5)
        return max(0.01, min(0.99, calibrated + goal_style_adj))

    elif 'OU_2_5_O' in market:
        league_avg = profile.get('over25_rate', 0.5)
        calibrated = prob * (1 - calibration_weight) + league_avg * calibration_weight
        return max(0.01, min(0.99, calibrated + goal_style_adj))

    elif 'OU_3_5_O' in market:
        league_avg = profile.get('over35_rate', 0.25)
        calibrated = prob * (1 - calibration_weight) + league_avg * calibration_weight
        return max(0.01, min(0.99, calibrated + goal_style_adj * 0.8))

    elif 'OU_4_5_O' in market:
        league_avg = profile.get('over45_rate', 0.15)
        calibrated = prob * (1 - calibration_weight) + league_avg * calibration_weight
        return max(0.01, min(0.99, calibrated + goal_style_adj * 0.6))

    elif '_U' in market and 'OU_' in market:
        # Under markets - inverse of style adjustment
        return max(0.01, min(0.99, prob - goal_style_adj))

    # ========== HOME/AWAY ADVANTAGE CALIBRATION ==========

    # Match result markets
    elif '1X2_H' in market:
        return min(prob * (1 + home_adv * TUNING_OVERRIDES.get('home_adv_1x2_h', 0.3)), 0.95)

    elif '1X2_A' in market:
        return max(prob * (1 - home_adv * TUNING_OVERRIDES.get('home_adv_1x2_a', 0.3)), 0.05)

    elif '1X2_D' in market:
        # Draw less likely in leagues with high home advantage
        return max(0.05, min(0.45, prob * (1 - home_adv * TUNING_OVERRIDES.get('home_adv_1x2_d', 0.15))))

    # Double Chance markets
    elif 'DC_1X' in market or 'DC1X' in market:
        # Home or Draw - boosted by home advantage
        return min(prob * (1 + home_adv * 0.15), 0.95)

    elif 'DC_X2' in market or 'DCX2' in market:
        # Away or Draw - reduced by home advantage
        return max(prob * (1 - home_adv * 0.15), 0.15)

    elif 'DC_12' in market or 'DC12' in market:
        # Home or Away (no draw) - slight boost in high home adv leagues
        return min(prob * (1 + home_adv * 0.08), 0.95)

    # Draw No Bet
    elif 'DNB_H' in market:
        return min(prob * (1 + home_adv * 0.25), 0.90)

    elif 'DNB_A' in market:
        return max(prob * (1 - home_adv * 0.25), 0.10)

    # Home/Away team goals
    elif 'HomeTG_' in market or 'HomeExact' in market:
        # Home team goals - boosted by home advantage
        line_boost = home_adv * 0.12
        return max(0.01, min(0.99, prob + line_boost))

    elif 'AwayTG_' in market or 'AwayExact' in market:
        # Away team goals - reduced by home advantage
        line_boost = home_adv * 0.12
        return max(0.01, min(0.99, prob - line_boost))

    # Team to score
    elif 'HomeToScore' in market:
        return min(prob * (1 + home_adv * 0.10), 0.95)

    elif 'AwayToScore' in market:
        return max(prob * (1 - home_adv * 0.10), 0.20)

    # Win to Nil / Clean Sheet - with home/away distinction
    elif 'HomeWTN' in market:
        base = prob * (1 - calibration_weight) + (clean_sheet_rate * 0.5) * calibration_weight
        # Home WTN boosted by home advantage
        return max(0.01, min(0.60, base * (1 + home_adv * 0.20) - goal_style_adj * 0.3))

    elif 'AwayWTN' in market:
        base = prob * (1 - calibration_weight) + (clean_sheet_rate * 0.35) * calibration_weight
        # Away WTN reduced by home advantage
        return max(0.01, min(0.40, base * (1 - home_adv * 0.20) - goal_style_adj * 0.3))

    elif 'HomeCS' in market:
        base = prob * (1 - calibration_weight) + clean_sheet_rate * calibration_weight
        return max(0.01, min(0.65, base * (1 + home_adv * 0.15) - goal_style_adj * 0.4))

    elif 'AwayCS' in market:
        base = prob * (1 - calibration_weight) + (clean_sheet_rate * 0.8) * calibration_weight
        return max(0.01, min(0.55, base * (1 - home_adv * 0.15) - goal_style_adj * 0.4))

    elif 'NoGoal' in market:
        # 0-0 draw - defensive leagues boost, high home adv reduces
        base = prob * (1 - calibration_weight) + (clean_sheet_rate * 0.15) * calibration_weight
        return max(0.01, min(0.15, base - goal_style_adj * 0.5))

    # Win by margin - home/away
    elif 'HomeWin' in market:
        boost = home_adv * 0.15
        return max(0.01, min(0.85, prob + boost))

    elif 'AwayWin' in market:
        boost = home_adv * 0.15
        return max(0.01, min(0.60, prob - boost))

    # Asian Handicap - home advantage affects line perception
    elif 'AH_' in market:
        if '_H' in market or market.endswith('_H'):
            # Home covers handicap
            return max(0.01, min(0.85, prob + home_adv * 0.08))
        elif '_A' in market or market.endswith('_A'):
            # Away covers handicap
            return max(0.01, min(0.85, prob - home_adv * 0.08))

    # Pure-ML binary markets (corners, YC, cards) — no DC signal, near-random → heavy compression
    # Note: market arrives as a P_ column name (e.g. "P_TotalCorners_O9_5_Y")
    binary_prefixes_ml = ('P_TotalCorners_', 'P_HomeCorners_', 'P_AwayCorners_',
                          'P_TotalYC_', 'P_BookingPts_', 'P_HomeTeam_Card', 'P_AwayTeam_Card')
    if any(market.startswith(pfx) for pfx in binary_prefixes_ml):
        temp = float(TUNING_OVERRIDES.get('temperature_binary', 1.0))
        if temp != 1.0 and 0.0 < prob < 1.0:
            import math
            log_odds = math.log(prob / (1.0 - prob)) / temp
            prob = 1.0 / (1.0 + math.exp(-log_odds))
        return max(0.01, min(0.99, prob))

    # DC-supported TG markets (HomeTG, AwayTG) — better calibrated, gentler compression
    hometg_prefixes = ('P_HomeTG_', 'P_AwayTG_')
    if any(market.startswith(pfx) for pfx in hometg_prefixes):
        temp = float(TUNING_OVERRIDES.get('temperature_hometg', 1.0))
        if temp != 1.0 and 0.0 < prob < 1.0:
            import math
            log_odds = math.log(prob / (1.0 - prob)) / temp
            prob = 1.0 / (1.0 + math.exp(-log_odds))
        return max(0.01, min(0.99, prob))

    return prob

def enforce_cross_market_constraints(row: pd.Series) -> pd.Series:
    """Ensure mathematical consistency between related markets"""
    row = row.copy()
    
    # 1. O/U probabilities must be monotonically decreasing
    if all(f'P_OU_{line}_O' in row for line in OU_LINES):
        for i in range(len(OU_LINES) - 1):
            curr_line = OU_LINES[i]
            next_line = OU_LINES[i + 1]
            curr_col = f'P_OU_{curr_line}_O'
            next_col = f'P_OU_{next_line}_O'
            
            if pd.notna(row[curr_col]) and pd.notna(row[next_col]):
                if row[curr_col] < row[next_col]:
                    avg = (row[curr_col] + row[next_col]) / 2
                    row[curr_col] = min(avg + 0.05, 0.99)
                    row[next_col] = max(avg - 0.05, 0.01)
                
                # Update Under probabilities
                row[f'P_OU_{curr_line}_U'] = 1 - row[curr_col]
                row[f'P_OU_{next_line}_U'] = 1 - row[next_col]
    
    # 2. BTTS and O/U 0.5 logical consistency.
    # BTTS is a SUBSET of Over 0.5 (both scoring => at least one goal), so the
    # only valid implication is P(Over 0.5) >= P(BTTS). The previous version
    # forced Over 0.5 to 0.98 whenever BTTS > 0.7 and hard-set BTTS to 0/1 —
    # unjustified distortions of calibrated probabilities.
    if 'P_BTTS_Y' in row and 'P_OU_0_5_O' in row:
        if pd.notna(row['P_BTTS_Y']) and pd.notna(row['P_OU_0_5_O']):
            if row['P_OU_0_5_O'] < row['P_BTTS_Y']:
                row['P_OU_0_5_O'] = row['P_BTTS_Y']
                row['P_OU_0_5_U'] = 1 - row['P_OU_0_5_O']
            # Conversely BTTS cannot exceed P(Over 0.5)
            if row['P_BTTS_Y'] > row['P_OU_0_5_O']:
                row['P_BTTS_Y'] = row['P_OU_0_5_O']
                row['P_BTTS_N'] = 1 - row['P_BTTS_Y']

    # 3. BTTS and O/U 1.5 consistency: BTTS implies >= 2 goals, so
    # P(Over 1.5) >= P(BTTS) exactly (no 0.9 fudge factor).
    if 'P_BTTS_Y' in row and 'P_OU_1_5_O' in row:
        if pd.notna(row['P_BTTS_Y']) and pd.notna(row['P_OU_1_5_O']):
            if row['P_OU_1_5_O'] < row['P_BTTS_Y']:
                row['P_OU_1_5_O'] = row['P_BTTS_Y']
                row['P_OU_1_5_U'] = 1 - row['P_OU_1_5_O']
    
    # 4. 1X2 probabilities sum to 1.0
    if all(f'P_1X2_{x}' in row for x in ['H', 'D', 'A']):
        if all(pd.notna(row[f'P_1X2_{x}']) for x in ['H', 'D', 'A']):
            total = row['P_1X2_H'] + row['P_1X2_D'] + row['P_1X2_A']
            if total > 0:
                row['P_1X2_H'] /= total
                row['P_1X2_D'] /= total
                row['P_1X2_A'] /= total
    
    # 5. Correct Score 0-0 cannot exceed Under 0.5
    if 'P_CS_0_0' in row and 'P_OU_0_5_U' in row:
        if pd.notna(row['P_CS_0_0']) and pd.notna(row['P_OU_0_5_U']):
            row['P_CS_0_0'] = min(row['P_CS_0_0'], row['P_OU_0_5_U'])
    
    # 6. Team goals and BTTS consistency.
    # Valid bounds without an independence assumption (Fréchet):
    #   P(BTTS) <= min(P(home scores), P(away scores))
    #   P(BTTS) >= P(home scores) + P(away scores) - 1
    # The old version pushed BTTS UP toward the independence product x0.85,
    # which is not a valid lower bound when goals are negatively correlated.
    if all(col in row for col in ['P_BTTS_Y', 'P_HomeTG_0_5_O', 'P_AwayTG_0_5_O']):
        if all(pd.notna(row[col]) for col in ['P_BTTS_Y', 'P_HomeTG_0_5_O', 'P_AwayTG_0_5_O']):
            upper = min(row['P_HomeTG_0_5_O'], row['P_AwayTG_0_5_O'])
            lower = max(0.0, row['P_HomeTG_0_5_O'] + row['P_AwayTG_0_5_O'] - 1.0)
            clipped = min(max(row['P_BTTS_Y'], lower), upper)
            if clipped != row['P_BTTS_Y']:
                row['P_BTTS_Y'] = clipped
                row['P_BTTS_N'] = 1 - clipped

    return row

def apply_poisson_adjustment(row: pd.Series, home_xg: float = None, away_xg: float = None,
                             league: str = None, league_profiles: Dict = None) -> pd.Series:
    """Apply Poisson distribution for goal-based markets.

    Prefers per-team rolling xG; falls back to the DYNAMIC league profile
    (computed from historical data) and only then to the static table.
    """
    row = row.copy()

    # Use league-specific or default xG
    profile = None
    if league:
        if league_profiles and league in league_profiles:
            profile = league_profiles[league]      # data-driven (preferred)
        elif league in LEAGUE_PROFILES:
            profile = LEAGUE_PROFILES[league]      # static fallback
    if profile:
        total_expected = profile['avg_goals']
        home_share = float(TUNING_OVERRIDES.get('poisson_home_share', 0.54))
        home_xg = home_xg or (total_expected * home_share)
        away_xg = away_xg or (total_expected * (1 - home_share))
    else:
        home_xg = home_xg or 1.4
        away_xg = away_xg or 1.1
    
    total_xg = home_xg + away_xg
    
    # Calculate Poisson probabilities for O/U lines
    for line in OU_LINES:
        line_value = float(line.replace('_', '.'))
        
        # Poisson probability of over this line
        poisson_over = 1 - poisson.cdf(line_value, total_xg)
        
        # Adaptive blending based on line
        _poisson_weights = TUNING_OVERRIDES.get('poisson_blend_weights',
            {'0_5': 0.3, '1_5': 0.4, '2_5': 0.5, '3_5': 0.4, '4_5': 0.3})
        blend_weight = _poisson_weights.get(line, 0.4)
        
        if f'P_OU_{line}_O' in row and pd.notna(row[f'P_OU_{line}_O']):
            row[f'P_OU_{line}_O'] = row[f'P_OU_{line}_O'] * (1 - blend_weight) + poisson_over * blend_weight
            row[f'P_OU_{line}_U'] = 1 - row[f'P_OU_{line}_O']
    
    # BTTS using Poisson
    prob_home_scores = 1 - poisson.pmf(0, home_xg)
    prob_away_scores = 1 - poisson.pmf(0, away_xg)
    poisson_btts = prob_home_scores * prob_away_scores
    
    if 'P_BTTS_Y' in row and pd.notna(row['P_BTTS_Y']):
        _btts_pw = TUNING_OVERRIDES.get('btts_poisson_weight', 0.35)
        row['P_BTTS_Y'] = row['P_BTTS_Y'] * (1 - _btts_pw) + poisson_btts * _btts_pw
        row['P_BTTS_N'] = 1 - row['P_BTTS_Y']
    
    return row

_FUTURE_FRAME_CACHE_DIR = Path(__file__).parent / "outputs" / "future_frame_cache"


def _build_future_frame(fixtures_csv: Path) -> pd.DataFrame:
    """Enhanced feature building with time weighting.

    Disk-caches the result keyed by fixture content + time_half_life so
    auto_tune trials with the same test period skip the ~10-min build step.
    """
    # Build cache key: fixture content + time_half_life + features.parquet mtime
    # (cache is automatically invalidated when features are rebuilt)
    fx_content = Path(fixtures_csv).read_text(errors='replace')
    thl = TUNING_OVERRIDES.get('time_half_life', 180)
    feat_mtime = int(Path(FEATURES_PARQUET).stat().st_mtime) if Path(FEATURES_PARQUET).exists() else 0
    ck = hashlib.md5(f"{fx_content}|{thl}|{feat_mtime}".encode()).hexdigest()[:16]
    cache_path = _FUTURE_FRAME_CACHE_DIR / f"ff_{ck}.parquet"

    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if len(cached) > 0:
                return cached
        except Exception:
            pass

    base = _load_base_features()
    base["Date"] = pd.to_datetime(base["Date"])
    fx = pd.read_csv(fixtures_csv)
    fx["Date"] = pd.to_datetime(fx["Date"])

    thl = TUNING_OVERRIDES.get('time_half_life', 180)

    # Only use pre-computed rolling/EWM numeric features for the form calculation.
    # Exclude raw result cols, metadata strings, and target cols.
    # Also exclude opponent-specific cols (Elo_Away, Elo_Diff) from home lookup —
    # those depend on the opponent, not just the home team's form.
    _meta = {"League", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
             "HTHG", "HTAG", "HTR", "referee", "venue_name", "Season",
             "fixture_id", "league_type"}
    # These are derived from BOTH teams and must be recomputed after merging home+away form.
    _opponent_cols = {
        "Elo_Away", "Elo_Diff", "Glicko_Diff",
        "TablePosDiff", "SeasonPtsDiff",
        "BothTopSix", "RelegationClash",
    }
    home_feat_cols = [c for c in base.columns
                      if c not in _meta and c not in _opponent_cols
                      and not c.startswith("y_")
                      and not c.startswith("Away_")
                      and pd.api.types.is_numeric_dtype(base[c])]
    # Away form: Away_* cols + Elo_Away (away team's own Elo from their away games)
    away_feat_cols = [c for c in base.columns
                      if c.startswith("Away_") and not c.startswith("y_")
                      and pd.api.types.is_numeric_dtype(base[c])]
    if "Elo_Away" in base.columns:
        away_feat_cols.append("Elo_Away")

    # ----------------------------------------------------------------
    # Vectorized time-weighted form: last-5 appearances per fixture team
    # ----------------------------------------------------------------
    def _weighted_form(fx_df, base_df, team_col, feat_cols, n_games=5):
        """Return one time-weighted-average row per (League, Date, team_col).
        All feat_cols must be numeric.
        """
        hist = base_df[["League", "Date", team_col] + feat_cols].rename(
            columns={"Date": "HistDate"})

        merged = fx_df[["League", "Date", team_col]].merge(
            hist, on=["League", team_col], how="inner"
        )
        merged = merged[merged["HistDate"] < merged["Date"]]

        # Keep last n_games per (League, team, fixture-date)
        merged = merged.sort_values(["League", team_col, "Date", "HistDate"])
        merged["_rank"] = (merged.groupby(["League", team_col, "Date"])
                           .cumcount(ascending=False) + 1)
        merged = merged[merged["_rank"] <= n_games].copy()

        # Time weights
        merged["_days"] = (merged["Date"] - merged["HistDate"]).dt.days.clip(lower=0)
        merged["_w"]    = np.exp(-merged["_days"] / thl)

        grp_keys = ["League", "Date", team_col]
        wsum = merged.groupby(grp_keys)["_w"].sum()

        # Multiply each col by weight then sum, then divide by wsum
        for col in feat_cols:
            merged[col] = merged[col].fillna(0) * merged["_w"]
        return (merged.groupby(grp_keys)[feat_cols].sum()
                .div(wsum, axis=0).reset_index())

    home_form = _weighted_form(fx, base, "HomeTeam", home_feat_cols)
    away_form  = _weighted_form(fx, base, "AwayTeam",  away_feat_cols)

    # Merge into fixture frame
    result = fx[["League", "Date", "HomeTeam", "AwayTeam"]].copy()
    # Also carry referee/venue if present in fixture CSV
    for extra in ["referee", "venue_name"]:
        if extra in fx.columns:
            result[extra] = fx[extra]

    result = result.merge(home_form, on=["League", "Date", "HomeTeam"], how="left")
    result = result.merge(away_form,  on=["League", "Date", "AwayTeam"],  how="left")

    # Add contextual flags AFTER merges to avoid _x/_y suffix collisions
    # (home_form/away_form may carry is_cup/is_european from base rolling stats)
    result["is_cup"] = result["League"].isin(ALL_CUPS).astype(int)
    result["is_european"] = result["League"].isin(EUROPEAN_CUPS).astype(int)

    # Recompute cross-team derived features now that both home and away form are merged
    if "Elo_Home" in result.columns and "Elo_Away" in result.columns:
        result["Elo_Diff"] = result["Elo_Home"] - result["Elo_Away"]
    if "Home_Glicko" in result.columns and "Away_Glicko" in result.columns:
        result["Glicko_Diff"] = result["Home_Glicko"] - result["Away_Glicko"]
    if "Home_TablePos" in result.columns and "Away_TablePos" in result.columns:
        result["TablePosDiff"] = result["Home_TablePos"] - result["Away_TablePos"]
    if "Home_SeasonPts" in result.columns and "Away_SeasonPts" in result.columns:
        result["SeasonPtsDiff"] = result["Home_SeasonPts"] - result["Away_SeasonPts"]
    if "IsTopSix_Home" in result.columns and "IsTopSix_Away" in result.columns:
        result["BothTopSix"]      = ((result["IsTopSix_Home"] == 1) & (result["IsTopSix_Away"] == 1)).astype(float)
    if "IsBottom3_Home" in result.columns and "IsBottom3_Away" in result.columns:
        result["RelegationClash"] = ((result["IsBottom3_Home"] == 1) | (result["IsBottom3_Away"] == 1)).astype(float)

    # ----------------------------------------------------------------
    # H2H features (vectorised per pair)
    # ----------------------------------------------------------------
    h2h_cols = {
        "H2H_HomeWinRate": np.nan, "H2H_AwayWinRate": np.nan, "H2H_DrawRate": np.nan,
        "H2H_AvgGoals": np.nan, "H2H_BTTSRate": np.nan,
        "H2H_HomeGoalsAvg": np.nan, "H2H_AwayGoalsAvg": np.nan, "H2H_Count": 0,
    }
    for col, default in h2h_cols.items():
        result[col] = default

    if "FTHG" in base.columns and "FTR" in base.columns:
        base_h2h = base[["League", "Date", "HomeTeam", "AwayTeam",
                          "FTHG", "FTAG", "FTR"]].copy()
        for _, r in fx.iterrows():
            lg, dt, ht, at = r["League"], r["Date"], r["HomeTeam"], r["AwayTeam"]
            h2h = base_h2h[
                (base_h2h["Date"] < dt) & (base_h2h["League"] == lg) &
                (((base_h2h["HomeTeam"] == ht) & (base_h2h["AwayTeam"] == at)) |
                 ((base_h2h["HomeTeam"] == at) & (base_h2h["AwayTeam"] == ht)))
            ].tail(5)
            if len(h2h) == 0:
                continue
            mask = (result["League"] == lg) & (result["Date"] == dt) & \
                   (result["HomeTeam"] == ht) & (result["AwayTeam"] == at)
            hah = h2h[h2h["HomeTeam"] == ht]
            haa = h2h[h2h["HomeTeam"] == at]
            n = len(h2h)
            wins   = int((hah["FTR"] == "H").sum()) + int((haa["FTR"] == "A").sum())
            losses = int((hah["FTR"] == "A").sum()) + int((haa["FTR"] == "H").sum())
            result.loc[mask, "H2H_HomeWinRate"] = wins / n
            result.loc[mask, "H2H_AwayWinRate"] = losses / n
            result.loc[mask, "H2H_DrawRate"]    = (n - wins - losses) / n
            result.loc[mask, "H2H_AvgGoals"]    = (h2h["FTHG"] + h2h["FTAG"]).mean()
            result.loc[mask, "H2H_Count"]        = n
            btts = ((h2h["FTHG"] > 0) & (h2h["FTAG"] > 0))
            result.loc[mask, "H2H_BTTSRate"]     = btts.mean()
            hg = pd.concat([hah["FTHG"], haa["FTAG"]])
            ag = pd.concat([hah["FTAG"], haa["FTHG"]])
            result.loc[mask, "H2H_HomeGoalsAvg"] = hg.mean() if len(hg) > 0 else np.nan
            result.loc[mask, "H2H_AwayGoalsAvg"] = ag.mean() if len(ag) > 0 else np.nan

    # ----------------------------------------------------------------
    # Referee features (vectorised lookup)
    # ----------------------------------------------------------------
    ref_cols = ["Ref_AvgGoals", "Ref_HomeWinRate", "Ref_BTTSRate",
                "Ref_AvgCards", "Ref_AvgFouls", "Ref_Count"]
    for col in ref_cols:
        result[col] = np.nan

    if "referee" in base.columns and "referee" in result.columns:
        has_cy = "Home_CardsY" in base.columns and "Away_CardsY" in base.columns
        has_foul = "Home_Fouls" in base.columns and "Away_Fouls" in base.columns
        ref_base = base[["Date", "referee", "FTHG", "FTAG", "FTR"] +
                        (["Home_CardsY", "Away_CardsY"] if has_cy else []) +
                        (["Home_Fouls", "Away_Fouls"] if has_foul else [])].dropna(subset=["referee"])
        for _, r in fx.iterrows():
            ref = r.get("referee")
            if pd.isna(ref):
                continue
            rp = ref_base[(ref_base["referee"] == ref) & (ref_base["Date"] < r["Date"])]
            if len(rp) < 3:
                continue
            mask = (result["HomeTeam"] == r["HomeTeam"]) & \
                   (result["Date"] == r["Date"]) & (result["League"] == r["League"])
            result.loc[mask, "Ref_AvgGoals"]   = (rp["FTHG"] + rp["FTAG"]).mean()
            result.loc[mask, "Ref_HomeWinRate"] = (rp["FTR"] == "H").mean()
            result.loc[mask, "Ref_BTTSRate"]    = ((rp["FTHG"] > 0) & (rp["FTAG"] > 0)).mean()
            result.loc[mask, "Ref_Count"]       = len(rp)
            if has_cy:
                result.loc[mask, "Ref_AvgCards"] = (rp["Home_CardsY"] + rp["Away_CardsY"]).mean()
            if has_foul:
                result.loc[mask, "Ref_AvgFouls"] = (rp["Home_Fouls"] + rp["Away_Fouls"]).mean()

    # Blank y_ target columns so models ignore them at prediction time
    for c in base.columns:
        if c.startswith("y_"):
            result[c] = pd.NA

    # Save to disk cache for reuse across auto_tune trials
    try:
        _FUTURE_FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
    except Exception:
        pass

    return result

def _collect_market_columns() -> List[str]:
    """All expected probability column names - COMPREHENSIVE VERSION"""
    cols = []

    # Core Markets
    cols += ["P_1X2_H", "P_1X2_D", "P_1X2_A"]
    cols += ["P_BTTS_Y", "P_BTTS_N"]

    # Over/Under Total Goals (Extended)
    for l in ["0_5", "1_5", "2_5", "3_5", "4_5", "5_5"]:
        cols += [f"P_OU_{l}_O", f"P_OU_{l}_U"]

    # Goal Range
    cols += [f"P_GR_{k}" for k in ["0","1","2","3","4","5+"]]

    # Exact Total Goals
    for i in ["0", "1", "2", "3", "4", "5", "6+"]:
        cols += [f"P_ExactTotal_{i}_Y", f"P_ExactTotal_{i}_N"]

    # Correct Score
    for i in range(6):
        for j in range(6):
            cols.append(f"P_CS_{i}_{j}")
    cols.append("P_CS_Other")

    # Draw No Bet
    cols += ["P_DNB_H_Y", "P_DNB_H_N", "P_DNB_A_Y", "P_DNB_A_N"]

    # To Score
    cols += ["P_HomeToScore_Y", "P_HomeToScore_N"]
    cols += ["P_AwayToScore_Y", "P_AwayToScore_N"]

    # Half-time Markets
    cols += ["P_HT_H", "P_HT_D", "P_HT_A"]
    cols += [f"P_HTFT_{a}_{b}" for a in ["H","D","A"] for b in ["H","D","A"]]
    cols += ["P_HT_OU_0_5_O", "P_HT_OU_0_5_U", "P_HT_OU_1_5_O", "P_HT_OU_1_5_U", "P_HT_OU_2_5_O", "P_HT_OU_2_5_U"]
    cols += ["P_HT_BTTS_Y", "P_HT_BTTS_N"]

    # Second Half Markets
    cols += ["P_2H_OU_0_5_O", "P_2H_OU_0_5_U", "P_2H_OU_1_5_O", "P_2H_OU_1_5_U", "P_2H_OU_2_5_O", "P_2H_OU_2_5_U"]
    cols += ["P_2H_BTTS_Y", "P_2H_BTTS_N"]

    # Half Comparison
    cols += ["P_HigherHalf_1H", "P_HigherHalf_2H", "P_HigherHalf_EQ"]
    cols += ["P_GoalsBothHalves_Y", "P_GoalsBothHalves_N"]
    cols += ["P_HomeScoresBothHalves_Y", "P_HomeScoresBothHalves_N"]
    cols += ["P_AwayScoresBothHalves_Y", "P_AwayScoresBothHalves_N"]

    # Win Half Markets
    cols += ["P_HomeWinEitherHalf_Y", "P_HomeWinEitherHalf_N"]
    cols += ["P_AwayWinEitherHalf_Y", "P_AwayWinEitherHalf_N"]
    cols += ["P_HomeWinBothHalves_Y", "P_HomeWinBothHalves_N"]
    cols += ["P_AwayWinBothHalves_Y", "P_AwayWinBothHalves_N"]

    # First to Score
    cols += ["P_FirstToScore_H", "P_FirstToScore_A", "P_FirstToScore_None"]

    # Team Goals Over/Under
    for l in ["0_5","1_5","2_5","3_5"]:
        cols += [f"P_HomeTG_{l}_O", f"P_HomeTG_{l}_U"]
        cols += [f"P_AwayTG_{l}_O", f"P_AwayTG_{l}_U"]

    # Exact Team Goals
    for i in ["0", "1", "2", "3+"]:
        cols += [f"P_HomeExact_{i}_Y", f"P_HomeExact_{i}_N"]
        cols += [f"P_AwayExact_{i}_Y", f"P_AwayExact_{i}_N"]

    # Asian Handicap (Extended)
    for l in ["-2_0", "-1_5", "-1_0", "-0_5", "0_0", "+0_5", "+1_0", "+1_5", "+2_0"]:
        cols += [f"P_AH_{l}_H", f"P_AH_{l}_A", f"P_AH_{l}_P"]

    # European Handicap
    for l in ["m1", "m2", "p1", "p2"]:
        cols += [f"P_EH_{l}_H_Y", f"P_EH_{l}_H_N"]
        cols += [f"P_EH_{l}_D_Y", f"P_EH_{l}_D_N"]
        cols += [f"P_EH_{l}_A_Y", f"P_EH_{l}_A_N"]

    # Double Chance
    cols += ["P_DC_1X_Y", "P_DC_1X_N"]
    cols += ["P_DC_X2_Y", "P_DC_X2_N"]
    cols += ["P_DC_12_Y", "P_DC_12_N"]

    # Win to Nil
    cols += ["P_HomeWTN_Y", "P_HomeWTN_N"]
    cols += ["P_AwayWTN_Y", "P_AwayWTN_N"]

    # Clean Sheets
    cols += ["P_HomeCS_Y", "P_HomeCS_N"]
    cols += ["P_AwayCS_Y", "P_AwayCS_N"]

    # No Goal
    cols += ["P_NoGoal_Y", "P_NoGoal_N"]

    # Win by Margin
    cols += ["P_HomeWinBy1_Y", "P_HomeWinBy1_N"]
    cols += ["P_HomeWinBy2_Y", "P_HomeWinBy2_N"]
    cols += ["P_HomeWinBy3+_Y", "P_HomeWinBy3+_N"]
    cols += ["P_AwayWinBy1_Y", "P_AwayWinBy1_N"]
    cols += ["P_AwayWinBy2_Y", "P_AwayWinBy2_N"]
    cols += ["P_AwayWinBy3+_Y", "P_AwayWinBy3+_N"]
    cols += ["P_HomeWin2+_Y", "P_HomeWin2+_N"]
    cols += ["P_AwayWin2+_Y", "P_AwayWin2+_N"]

    # Odd/Even
    cols += ["P_TotalOddEven_Odd", "P_TotalOddEven_Even"]
    cols += ["P_HomeOddEven_Odd", "P_HomeOddEven_Even"]
    cols += ["P_AwayOddEven_Odd", "P_AwayOddEven_Even"]

    # Multi-Goal
    cols += ["P_Match2+Goals_Y", "P_Match2+Goals_N"]
    cols += ["P_Match3+Goals_Y", "P_Match3+Goals_N"]
    cols += ["P_Match4+Goals_Y", "P_Match4+Goals_N"]
    cols += ["P_Match5+Goals_Y", "P_Match5+Goals_N"]

    # Result & BTTS Combos
    cols += ["P_HomeWin_BTTS_Y_Y", "P_HomeWin_BTTS_Y_N"]
    cols += ["P_HomeWin_BTTS_N_Y", "P_HomeWin_BTTS_N_N"]
    cols += ["P_AwayWin_BTTS_Y_Y", "P_AwayWin_BTTS_Y_N"]
    cols += ["P_AwayWin_BTTS_N_Y", "P_AwayWin_BTTS_N_N"]
    cols += ["P_Draw_BTTS_Y_Y", "P_Draw_BTTS_Y_N"]
    cols += ["P_Draw_BTTS_N_Y", "P_Draw_BTTS_N_N"]

    # Result & O/U Combos
    cols += ["P_HomeWin_O25_Y", "P_HomeWin_O25_N"]
    cols += ["P_HomeWin_U25_Y", "P_HomeWin_U25_N"]
    cols += ["P_AwayWin_O25_Y", "P_AwayWin_O25_N"]
    cols += ["P_AwayWin_U25_Y", "P_AwayWin_U25_N"]
    cols += ["P_Draw_O25_Y", "P_Draw_O25_N"]
    cols += ["P_Draw_U25_Y", "P_Draw_U25_N"]

    # Double Chance + O/U Combos
    cols += ["P_DC1X_O25_Y", "P_DC1X_O25_N"]
    cols += ["P_DC1X_U25_Y", "P_DC1X_U25_N"]
    cols += ["P_DCX2_O25_Y", "P_DCX2_O25_N"]
    cols += ["P_DCX2_U25_Y", "P_DCX2_U25_N"]
    cols += ["P_DC12_O25_Y", "P_DC12_O25_N"]
    cols += ["P_DC12_U25_Y", "P_DC12_U25_N"]

    # Double Chance + BTTS Combos
    cols += ["P_DC1X_BTTS_Y_Y", "P_DC1X_BTTS_Y_N"]
    cols += ["P_DC1X_BTTS_N_Y", "P_DC1X_BTTS_N_N"]
    cols += ["P_DCX2_BTTS_Y_Y", "P_DCX2_BTTS_Y_N"]
    cols += ["P_DCX2_BTTS_N_Y", "P_DCX2_BTTS_N_N"]

    # Corners O/U binary markets
    for sfx in ["O6_5","O7_5","O8_5","O9_5","O10_5","O11_5","O12_5","O13_5"]:
        cols += [f"P_TotalCorners_{sfx}_Y", f"P_TotalCorners_{sfx}_N"]
    for sfx in ["O3_5","O4_5","O5_5","O6_5"]:
        cols += [f"P_HomeCorners_{sfx}_Y", f"P_HomeCorners_{sfx}_N"]
        cols += [f"P_AwayCorners_{sfx}_Y", f"P_AwayCorners_{sfx}_N"]
    cols += ["P_HomeCorners_Win_Y", "P_HomeCorners_Win_N"]

    # Yellow cards / booking points binary markets
    for sfx in ["O1_5","O2_5","O3_5","O4_5","O5_5","O6_5"]:
        cols += [f"P_TotalYC_{sfx}_Y", f"P_TotalYC_{sfx}_N"]
    for sfx in ["O20_5","O30_5","O40_5","O50_5"]:
        cols += [f"P_BookingPts_{sfx}_Y", f"P_BookingPts_{sfx}_N"]
    cols += ["P_HomeTeam_Card_Y", "P_HomeTeam_Card_N"]
    cols += ["P_AwayTeam_Card_Y", "P_AwayTeam_Card_N"]

    return cols

def _map_preds_to_columns(models, preds: dict, fixtures_df: pd.DataFrame = None) -> Tuple[List[dict], List[str]]:
    """Enhanced mapping with all improvements"""
    out_cols = _collect_market_columns()
    n_rows = next(iter(preds.values())).shape[0] if preds else 0
    rows = []
    
    # Calculate league profiles
    base_features = _load_base_features()
    league_profiles = calculate_league_profiles(base_features)
    
    class_maps = {t: list(m.classes_) for t, m in models.items()}
    
    def labmap(t):
        return {lab: i for i, lab in enumerate(class_maps.get(t, []))}
    
    def pick(p, t, label):
        if t not in class_maps:
            return 0.0
        m = labmap(t)
        return float(p[m[label]]) if label in m else 0.0
    
    for i in range(n_rows):
        row = {}

        # Get fixture info
        league = fixtures_df.iloc[i]['League'] if fixtures_df is not None and 'League' in fixtures_df.columns else None

        # Extract rolling xG for Poisson blending — use None if missing/NaN
        _home_xg = _away_xg = None
        if fixtures_df is not None:
            _r = fixtures_df.iloc[i]
            _hx = _r.get('Home_xG_ewm') or _r.get('xG_ewm')
            _ax = _r.get('Away_xG_ewm')
            _home_xg = float(_hx) if _hx is not None and not pd.isna(_hx) else None
            _away_xg = float(_ax) if _ax is not None and not pd.isna(_ax) else None

        # Initialize
        for col in out_cols:
            row[col] = 0.0
        
        # Map predictions (same as predict2.py)
        if "y_1X2" in preds:
            p = preds["y_1X2"][i]
            row["P_1X2_H"] = pick(p, "y_1X2", "H")
            row["P_1X2_D"] = pick(p, "y_1X2", "D")
            row["P_1X2_A"] = pick(p, "y_1X2", "A")
        
        if "y_BTTS" in preds:
            p = preds["y_BTTS"][i]
            row["P_BTTS_Y"] = pick(p, "y_BTTS", "Y")
            row["P_BTTS_N"] = pick(p, "y_BTTS", "N")
        
        for l in OU_LINES:
            key = f"y_OU_{l}"
            if key in preds:
                p = preds[key][i]
                row[f"P_OU_{l}_O"] = pick(p, key, "O")
                row[f"P_OU_{l}_U"] = pick(p, key, "U")
        
        for l in AH_LINES:
            key = f"y_AH_{l}"
            if key in preds:
                p = preds[key][i]
                row[f"P_AH_{l}_H"] = pick(p, key, "H")
                row[f"P_AH_{l}_A"] = pick(p, key, "A")
                row[f"P_AH_{l}_P"] = pick(p, key, "P")
        
        if "y_GOAL_RANGE" in preds:
            p = preds["y_GOAL_RANGE"][i]
            for k in ["0","1","2","3","4","5+"]:
                row[f"P_GR_{k}"] = pick(p, "y_GOAL_RANGE", k)
        
        if "y_HT" in preds:
            p = preds["y_HT"][i]
            row["P_HT_H"] = pick(p, "y_HT", "H")
            row["P_HT_D"] = pick(p, "y_HT", "D")
            row["P_HT_A"] = pick(p, "y_HT", "A")
        
        if "y_HTFT" in preds:
            p = preds["y_HTFT"][i]
            for a in ["H","D","A"]:
                for b in ["H","D","A"]:
                    row[f"P_HTFT_{a}_{b}"] = pick(p, "y_HTFT", f"{a}-{b}")
        
        for l in ["0_5","1_5","2_5","3_5"]:
            hk = f"y_HomeTG_{l}"
            ak = f"y_AwayTG_{l}"
            if hk in preds:
                p = preds[hk][i]
                row[f"P_HomeTG_{l}_O"] = pick(p, hk, "O")
                row[f"P_HomeTG_{l}_U"] = pick(p, hk, "U")
            if ak in preds:
                p = preds[ak][i]
                row[f"P_AwayTG_{l}_O"] = pick(p, ak, "O")
                row[f"P_AwayTG_{l}_U"] = pick(p, ak, "U")
        
        if "y_HomeCardsY_BAND" in preds:
            p = preds["y_HomeCardsY_BAND"][i]
            for b in ["0-2","3","4-5","6+"]:
                row[f"P_HomeCardsY_{b}"] = pick(p, "y_HomeCardsY_BAND", b)
        
        if "y_AwayCardsY_BAND" in preds:
            p = preds["y_AwayCardsY_BAND"][i]
            for b in ["0-2","3","4-5","6+"]:
                row[f"P_AwayCardsY_{b}"] = pick(p, "y_AwayCardsY_BAND", b)
        
        if "y_HomeCorners_BAND" in preds:
            p = preds["y_HomeCorners_BAND"][i]
            for b in ["0-3","4-5","6-7","8-9","10+"]:
                row[f"P_HomeCorners_{b}"] = pick(p, "y_HomeCorners_BAND", b)
        
        if "y_AwayCorners_BAND" in preds:
            p = preds["y_AwayCorners_BAND"][i]
            for b in ["0-3","4-5","6-7","8-9","10+"]:
                row[f"P_AwayCorners_{b}"] = pick(p, "y_AwayCorners_BAND", b)

        # Binary corners O/U targets (Y=Over, N=Under)
        for sfx in ["O6_5","O7_5","O8_5","O9_5","O10_5","O11_5","O12_5","O13_5"]:
            key = f"y_TotalCorners_{sfx}"
            if key in preds:
                p = preds[key][i]
                row[f"P_TotalCorners_{sfx}_Y"] = pick(p, key, "Y")
                row[f"P_TotalCorners_{sfx}_N"] = pick(p, key, "N")
        for sfx in ["O3_5","O4_5","O5_5","O6_5"]:
            hk = f"y_HomeCorners_{sfx}"
            ak = f"y_AwayCorners_{sfx}"
            if hk in preds:
                p = preds[hk][i]
                row[f"P_HomeCorners_{sfx}_Y"] = pick(p, hk, "Y")
                row[f"P_HomeCorners_{sfx}_N"] = pick(p, hk, "N")
            if ak in preds:
                p = preds[ak][i]
                row[f"P_AwayCorners_{sfx}_Y"] = pick(p, ak, "Y")
                row[f"P_AwayCorners_{sfx}_N"] = pick(p, ak, "N")
        if "y_HomeCorners_Win" in preds:
            p = preds["y_HomeCorners_Win"][i]
            row["P_HomeCorners_Win_Y"] = pick(p, "y_HomeCorners_Win", "Y")
            row["P_HomeCorners_Win_N"] = pick(p, "y_HomeCorners_Win", "N")

        # Yellow cards / booking points targets (Y=Over, N=Under)
        for sfx in ["O1_5","O2_5","O3_5","O4_5","O5_5","O6_5"]:
            key = f"y_TotalYC_{sfx}"
            if key in preds:
                p = preds[key][i]
                row[f"P_TotalYC_{sfx}_Y"] = pick(p, key, "Y")
                row[f"P_TotalYC_{sfx}_N"] = pick(p, key, "N")
        for sfx in ["O20_5","O30_5","O40_5","O50_5"]:
            key = f"y_BookingPts_{sfx}"
            if key in preds:
                p = preds[key][i]
                row[f"P_BookingPts_{sfx}_Y"] = pick(p, key, "Y")
                row[f"P_BookingPts_{sfx}_N"] = pick(p, key, "N")
        for side in ["HomeTeam","AwayTeam"]:
            key = f"y_{side}_Card"
            if key in preds:
                p = preds[key][i]
                row[f"P_{side}_Card_Y"] = pick(p, key, "Y")
                row[f"P_{side}_Card_N"] = pick(p, key, "N")

        if "y_CS" in preds:
            p = preds["y_CS"][i]
            for a in range(6):
                for b in range(6):
                    row[f"P_CS_{a}_{b}"] = pick(p, "y_CS", f"{a}-{b}")
            row["P_CS_Other"] = pick(p, "y_CS", "Other")
        
        # Apply league calibration
        if league and league_profiles:
            for market in row.keys():
                if market.startswith('P_') and pd.notna(row[market]):
                    row[market] = apply_league_calibration(
                        row[market], market, league, league_profiles
                    )
        
        # Convert to series
        row_series = pd.Series(row)
        
        # Apply Poisson adjustments using per-team rolling xG when available
        row_series = apply_poisson_adjustment(row_series, home_xg=_home_xg, away_xg=_away_xg,
                                              league=league, league_profiles=league_profiles)
        
        # Enforce cross-market constraints
        row_series = enforce_cross_market_constraints(row_series)
        
        rows.append(row_series.to_dict())
    
    return rows, out_cols


def apply_injury_adjustments(df: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply injury-based adjustments to predictions.

    Injuries affect:
    - Team strength (more injuries = weaker)
    - Goals markets (key attacker out = fewer goals)
    - Clean sheet probability

    Args:
        df: DataFrame with predictions
        fixtures_df: DataFrame with fixture info including injury data

    Returns:
        DataFrame with adjusted predictions
    """
    if 'home_injuries' not in fixtures_df.columns or 'away_injuries' not in fixtures_df.columns:
        return df

    print("[INJURY] Applying injury impact adjustments...")

    df = df.copy()

    # Maximum injury impact (per player out)
    INJURY_IMPACT_PER_PLAYER = 0.02  # 2% adjustment per injury
    MAX_INJURY_IMPACT = 0.10  # Maximum 10% total adjustment

    for idx in range(len(df)):
        if idx >= len(fixtures_df):
            break

        home_inj = fixtures_df.iloc[idx].get('home_injuries', 0)
        away_inj = fixtures_df.iloc[idx].get('away_injuries', 0)

        if home_inj == 0 and away_inj == 0:
            continue

        # Calculate injury impact (capped)
        home_impact = min(home_inj * INJURY_IMPACT_PER_PLAYER, MAX_INJURY_IMPACT)
        away_impact = min(away_inj * INJURY_IMPACT_PER_PLAYER, MAX_INJURY_IMPACT)

        # Adjust 1X2 probabilities
        # More home injuries = reduce home win, increase away win
        # More away injuries = reduce away win, increase home win
        for prefix in ['P_', 'BLEND_']:
            h_col = f'{prefix}1X2_H'
            d_col = f'{prefix}1X2_D'
            a_col = f'{prefix}1X2_A'

            if h_col in df.columns and a_col in df.columns:
                # Adjust based on injury differential
                h_adj = away_impact - home_impact  # Home benefits from away injuries
                a_adj = home_impact - away_impact  # Away benefits from home injuries

                df.at[idx, h_col] = max(0.01, min(0.99, df.at[idx, h_col] + h_adj))
                df.at[idx, a_col] = max(0.01, min(0.99, df.at[idx, a_col] + a_adj))

                # Normalize to ensure sum <= 1
                total = df.at[idx, h_col] + df.at[idx, d_col] + df.at[idx, a_col]
                if total > 1:
                    df.at[idx, h_col] /= total
                    df.at[idx, d_col] /= total
                    df.at[idx, a_col] /= total

        # Adjust over/under based on total injuries (more injuries typically = fewer goals)
        total_inj = home_inj + away_inj
        if total_inj > 0:
            goal_reduction = min(total_inj * 0.01, 0.05)  # Max 5% reduction in over probability

            for line in ['0_5', '1_5', '2_5', '3_5', '4_5']:
                for prefix in ['P_', 'BLEND_']:
                    o_col = f'{prefix}OU_{line}_O'
                    u_col = f'{prefix}OU_{line}_U'

                    if o_col in df.columns and u_col in df.columns:
                        df.at[idx, o_col] = max(0.01, df.at[idx, o_col] - goal_reduction)
                        df.at[idx, u_col] = min(0.99, df.at[idx, u_col] + goal_reduction)

    adjusted_count = len(fixtures_df[(fixtures_df['home_injuries'] > 0) | (fixtures_df['away_injuries'] > 0)])
    print(f"[INJURY] Adjusted {adjusted_count} fixtures based on injury data")

    return df


def _apply_blend(out: pd.DataFrame) -> pd.DataFrame:
    """Enhanced blending with dynamic weights by league quality"""
    try:
        if not BLEND_WEIGHTS_JSON.exists():
            heartbeat("Blend weights file missing; skipping BLEND_* columns.")
            return out

        weights = json.loads(BLEND_WEIGHTS_JSON.read_text())
        if not weights:
            heartbeat("No blend weights found; skipping BLEND_* columns.")
            return out
    except Exception as e:
        heartbeat(f"Error loading blend weights: {e}; skipping BLEND_* columns.")
        return out

    # Dynamic league profiles for quality-based ML weight adjustment
    try:
        _base = _load_base_features()
        league_profiles = calculate_league_profiles(_base)
    except Exception:
        league_profiles = {}

    def pair_cols_for_target(target: str) -> Tuple[List[str], List[str]]:
        # Normalise: blend_weights.json may store keys with or without "y_" prefix
        t = target if target.startswith("y_") else f"y_{target}"
        if t == "y_1X2":
            ml_cols = ["P_1X2_H","P_1X2_D","P_1X2_A"]
            dc_cols = ["DC_1X2_H","DC_1X2_D","DC_1X2_A"]
        elif t == "y_BTTS":
            ml_cols = ["P_BTTS_N","P_BTTS_Y"]
            dc_cols = ["DC_BTTS_N","DC_BTTS_Y"]
        elif t == "y_GOAL_RANGE":
            ml_cols = [f"P_GR_{k}" for k in ["0","1","2","3","4","5+"]]
            dc_cols = [f"DC_GR_{k}" for k in ["0","1","2","3","4","5+"]]
        elif t == "y_CS":
            ml_cols = [f"P_CS_{a}_{b}" for a in range(6) for b in range(6)] + ["P_CS_Other"]
            dc_cols = [f"DC_CS_{a}_{b}" for a in range(6) for b in range(6)] + ["DC_CS_Other"]
        elif t.startswith("y_OU_"):
            line_part = t.replace("y_OU_", "")
            ml_cols = [f"P_OU_{line_part}_U", f"P_OU_{line_part}_O"]
            dc_cols = [f"DC_OU_{line_part}_U", f"DC_OU_{line_part}_O"]
        elif t.startswith("y_AH_"):
            line_part = t.replace("y_AH_", "")
            ml_cols = [f"P_AH_{line_part}_A", f"P_AH_{line_part}_P", f"P_AH_{line_part}_H"]
            dc_cols = [f"DC_AH_{line_part}_A", f"DC_AH_{line_part}_P", f"DC_AH_{line_part}_H"]
        elif t.startswith("y_HomeTG_"):
            line_part = t.replace("y_HomeTG_", "")
            ml_cols = [f"P_HomeTG_{line_part}_U", f"P_HomeTG_{line_part}_O"]
            dc_cols = [f"DC_HomeTG_{line_part}_U", f"DC_HomeTG_{line_part}_O"]
        elif t.startswith("y_AwayTG_"):
            line_part = t.replace("y_AwayTG_", "")
            ml_cols = [f"P_AwayTG_{line_part}_U", f"P_AwayTG_{line_part}_O"]
            dc_cols = [f"DC_AwayTG_{line_part}_U", f"DC_AwayTG_{line_part}_O"]
        # Cards/corners markets: blend against the NB (negative-binomial count
        # model) prediction when available — a genuine second signal, unlike
        # DC which has no notion of cards/corners. Falls back to ML-only
        # (dc_cols == ml_cols) if NB predictions weren't generated for some
        # reason (e.g. insufficient training data for that count family).
        elif t.startswith("y_TotalYC_"):
            sfx = t.replace("y_TotalYC_", "")
            ml_cols = [f"P_TotalYC_{sfx}_Y", f"P_TotalYC_{sfx}_N"]
            nb_cols = [f"NB_TotalYC_{sfx}_Y", f"NB_TotalYC_{sfx}_N"]
            dc_cols = nb_cols
        elif t.startswith("y_BookingPts_"):
            sfx = t.replace("y_BookingPts_", "")
            ml_cols = [f"P_BookingPts_{sfx}_Y", f"P_BookingPts_{sfx}_N"]
            dc_cols = ml_cols  # no NB support (booking points, not a raw count family)
        elif t in ("y_HomeTeam_Card", "y_AwayTeam_Card"):
            name = t.replace("y_", "")
            ml_cols = [f"P_{name}_Y", f"P_{name}_N"]
            dc_cols = [f"NB_{name}_Y", f"NB_{name}_N"]
        elif t.startswith("y_TotalCorners_"):
            sfx = t.replace("y_TotalCorners_", "")
            ml_cols = [f"P_TotalCorners_{sfx}_Y", f"P_TotalCorners_{sfx}_N"]
            dc_cols = [f"NB_TotalCorners_{sfx}_Y", f"NB_TotalCorners_{sfx}_N"]
        elif t.startswith("y_HomeCorners_"):
            sfx = t.replace("y_HomeCorners_", "")
            ml_cols = [f"P_HomeCorners_{sfx}_Y", f"P_HomeCorners_{sfx}_N"]
            dc_cols = [f"NB_HomeCorners_{sfx}_Y", f"NB_HomeCorners_{sfx}_N"]
        elif t.startswith("y_AwayCorners_"):
            sfx = t.replace("y_AwayCorners_", "")
            ml_cols = [f"P_AwayCorners_{sfx}_Y", f"P_AwayCorners_{sfx}_N"]
            dc_cols = [f"NB_AwayCorners_{sfx}_Y", f"NB_AwayCorners_{sfx}_N"]
        elif t == "y_HTFT":
            ml_cols = [f"P_HTFT_{a}_{b}" for a in ["H","D","A"] for b in ["H","D","A"]]
            dc_cols = ml_cols  # no DC model for HTFT
        else:
            return [], []
        return ml_cols, dc_cols

    print("Creating enhanced BLEND predictions with dynamic weights...")
    
    # Apply blending row by row with league-specific adjustments
    for idx in range(len(out)):
        league = out.iloc[idx]['League'] if 'League' in out.columns else None
        
        # Determine ML weight adjustment based on league quality (dynamic profiles)
        ml_weight_boost = 0.0
        _lp = league_profiles.get(league) if league_profiles else None
        if _lp is None and league in LEAGUE_PROFILES:
            _lp = LEAGUE_PROFILES[league]
        if _lp:
            quality = _lp.get('quality', 'medium')
            if quality == 'elite':
                ml_weight_boost = 0.15  # Trust ML more in top leagues
            elif quality == 'high':
                ml_weight_boost = 0.10
            elif quality == 'medium':
                ml_weight_boost = 0.05
        
        for target, base_alpha in weights.items():
            ml_cols, dc_cols = pair_cols_for_target(target)
            if not ml_cols:
                continue

            missing_ml = [c for c in ml_cols if c not in out.columns]
            missing_dc = [c for c in dc_cols if c not in out.columns]

            if missing_ml:
                continue
            if missing_dc:
                # Second signal (DC/NB) unavailable for this target — fall back
                # to the ML-only passthrough rather than dropping the market
                # from BLEND_* entirely.
                dc_cols = ml_cols

            # ML-only markets (cards/corners/HTFT) have dc_cols == ml_cols:
            # there is nothing to blend, and applying dc_temperature to the ML
            # probs here would double-compress them on top of the
            # temperature_binary scaling already applied during calibration.
            if dc_cols == ml_cols:
                blend_cols = [c.replace("P_", "BLEND_") for c in ml_cols]
                out.loc[idx, blend_cols] = out.loc[idx, ml_cols].values.astype(np.float64)
                continue

            # Adjust alpha based on league quality, with per-market ML weight cap
            global_cap = TUNING_OVERRIDES.get('ml_weight_cap', 0.85)
            market_cap_key = f'ml_weight_cap_{target.replace("y_", "").lower()}'
            ml_cap = float(TUNING_OVERRIDES.get(market_cap_key, global_cap))
            alpha = min(float(base_alpha) + ml_weight_boost, ml_cap)

            # Get probabilities (force float64 — DataFrame loc can return object dtype)
            M = out.loc[idx, ml_cols].values.astype(np.float64)
            D = out.loc[idx, dc_cols].values.astype(np.float64)

            # Apply DC temperature scaling to reduce overconfidence (T > 1 softens, T < 1 sharpens)
            # Per-market overrides take precedence over the global dc_temperature
            global_temp = float(TUNING_OVERRIDES.get('dc_temperature', 1.0))
            market_temp_key = f'dc_temperature_{target.replace("y_", "").lower()}'
            dc_temp = float(TUNING_OVERRIDES.get(market_temp_key, global_temp))
            if dc_temp != 1.0 and len(D) > 0 and D.sum() > 0:
                log_D = np.log(np.clip(D, 1e-10, None)) / dc_temp
                D = np.exp(log_D - log_D.max())
                D = D / D.sum()

            # Blend: alpha * ML + (1-alpha) * DC
            B = alpha * M + (1.0 - alpha) * D
            
            # Renormalize
            s = B.sum()
            if s > 0:
                B = B / s
            
            # Create BLEND columns
            blend_cols = [c.replace("P_","BLEND_") for c in ml_cols]
            out.loc[idx, blend_cols] = B
    
    # Apply final cross-market constraints to BLEND columns
    print("Applying cross-market constraints to BLEND predictions...")
    for idx in range(len(out)):
        blend_cols = [c for c in out.columns if c.startswith('BLEND_')]
        if blend_cols:
            blend_row = out.loc[idx, blend_cols]
            renamed = {col: col.replace('BLEND_', 'P_') for col in blend_cols}
            blend_row = blend_row.rename(renamed)
            blend_row = enforce_cross_market_constraints(blend_row)
            
            for old_name, new_name in renamed.items():
                out.at[idx, old_name] = blend_row[new_name]
    
    return out

def calculate_confidence_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate confidence based on model agreement"""
    print("Calculating confidence scores with model agreement...")
    
    for idx in range(len(df)):
        # Key markets to check
        for market in ['1X2_H', '1X2_D', '1X2_A', 'BTTS_Y', 'BTTS_N', 'OU_2_5_O', 'OU_2_5_U']:
            predictions = []
            
            for prefix in ['P_', 'DC_', 'BLEND_']:
                col = f'{prefix}{market}'
                if col in df.columns and pd.notna(df.at[idx, col]):
                    predictions.append(df.at[idx, col])
            
            if len(predictions) >= 2:
                # Confidence = 1 - (standard deviation * 2)
                std_dev = np.std(predictions)
                confidence = max(0, 1 - min(std_dev * 2, 1))
                df.at[idx, f'CONF_{market}'] = confidence
                
                # Also store agreement score (0-100%)
                mean_pred = np.mean(predictions)
                max_deviation = max(abs(p - mean_pred) for p in predictions)
                agreement = max(0, 1 - (max_deviation * 2))
                df.at[idx, f'AGREE_{market}'] = agreement * 100
    
    return df

def _write_combined_high_confidence(df: pd.DataFrame, path: Path):
    """
    Create combined output for 1X2, OU2.5, OU1.5, and BTTS markets above 90% confidence.
    Shows all four markets side-by-side for each match where any qualifies.
    """
    print("\n[COMBINED] Creating high-confidence combined output (1X2 + OU2.5 + OU1.5 + BTTS >= 90%)...")

    # One source per market, in preference order BLEND > DC > P. Taking the
    # max across ALL sources (as before) cherry-picks whichever model happens
    # to be most confident per row, systematically inflating "90%+" picks.
    def _pick_source(df_cols, candidates_by_source):
        for source_cols in candidates_by_source:
            if all(c in df_cols for c in source_cols):
                return source_cols
        return []

    markets = {
        '1X2': _pick_source(df.columns, [
            ['BLEND_1X2_H', 'BLEND_1X2_D', 'BLEND_1X2_A'],
            ['DC_1X2_H', 'DC_1X2_D', 'DC_1X2_A'],
            ['P_1X2_H', 'P_1X2_D', 'P_1X2_A'],
        ]),
        'OU_2_5': _pick_source(df.columns, [
            ['BLEND_OU_2_5_O', 'BLEND_OU_2_5_U'],
            ['DC_OU_2_5_O', 'DC_OU_2_5_U'],
            ['P_OU_2_5_O', 'P_OU_2_5_U'],
        ]),
        'OU_1_5': _pick_source(df.columns, [
            ['BLEND_OU_1_5_O', 'BLEND_OU_1_5_U'],
            ['DC_OU_1_5_O', 'DC_OU_1_5_U'],
            ['P_OU_1_5_O', 'P_OU_1_5_U'],
        ]),
        'BTTS': _pick_source(df.columns, [
            ['BLEND_BTTS_Y', 'BLEND_BTTS_N'],
            ['DC_BTTS_Y', 'DC_BTTS_N'],
            ['P_BTTS_Y', 'P_BTTS_N'],
        ]),
    }

    rows = []
    threshold = 0.90

    for idx, row in df.iterrows():
        match_info = {
            'Date': row.get('Date', ''),
            'League': row.get('League', ''),
            'Home': row.get('Home', row.get('HomeTeam', '')),
            'Away': row.get('Away', row.get('AwayTeam', '')),
        }

        # Check each market
        has_high_conf = False

        # 1X2 Market
        x1x2_cols = [c for c in markets['1X2'] if c in df.columns]
        if x1x2_cols:
            x1x2_probs = row[x1x2_cols]
            best_1x2 = x1x2_probs.max()
            best_1x2_col = x1x2_probs.idxmax() if best_1x2 > 0 else ''
            if best_1x2 >= threshold:
                has_high_conf = True
                if '_H' in best_1x2_col:
                    match_info['1X2_Pick'] = 'Home'
                elif '_D' in best_1x2_col:
                    match_info['1X2_Pick'] = 'Draw'
                elif '_A' in best_1x2_col:
                    match_info['1X2_Pick'] = 'Away'
                match_info['1X2_Prob'] = f"{best_1x2:.1%}"
            else:
                match_info['1X2_Pick'] = '-'
                match_info['1X2_Prob'] = f"{best_1x2:.1%}" if best_1x2 > 0 else '-'
        else:
            match_info['1X2_Pick'] = '-'
            match_info['1X2_Prob'] = '-'

        # OU 2.5 Market
        ou25_cols = [c for c in markets['OU_2_5'] if c in df.columns]
        if ou25_cols:
            ou25_probs = row[ou25_cols]
            best_ou25 = ou25_probs.max()
            best_ou25_col = ou25_probs.idxmax() if best_ou25 > 0 else ''
            if best_ou25 >= threshold:
                has_high_conf = True
                match_info['OU25_Pick'] = 'Over' if '_O' in best_ou25_col else 'Under'
                match_info['OU25_Prob'] = f"{best_ou25:.1%}"
            else:
                match_info['OU25_Pick'] = '-'
                match_info['OU25_Prob'] = f"{best_ou25:.1%}" if best_ou25 > 0 else '-'
        else:
            match_info['OU25_Pick'] = '-'
            match_info['OU25_Prob'] = '-'

        # OU 1.5 Market
        ou15_cols = [c for c in markets['OU_1_5'] if c in df.columns]
        if ou15_cols:
            ou15_probs = row[ou15_cols]
            best_ou15 = ou15_probs.max()
            best_ou15_col = ou15_probs.idxmax() if best_ou15 > 0 else ''
            if best_ou15 >= threshold:
                has_high_conf = True
                match_info['OU15_Pick'] = 'Over' if '_O' in best_ou15_col else 'Under'
                match_info['OU15_Prob'] = f"{best_ou15:.1%}"
            else:
                match_info['OU15_Pick'] = '-'
                match_info['OU15_Prob'] = f"{best_ou15:.1%}" if best_ou15 > 0 else '-'
        else:
            match_info['OU15_Pick'] = '-'
            match_info['OU15_Prob'] = '-'

        # BTTS Market
        btts_cols = [c for c in markets['BTTS'] if c in df.columns]
        if btts_cols:
            btts_probs = row[btts_cols]
            best_btts = btts_probs.max()
            best_btts_col = btts_probs.idxmax() if best_btts > 0 else ''
            if best_btts >= threshold:
                has_high_conf = True
                match_info['BTTS_Pick'] = 'Yes' if '_Y' in best_btts_col else 'No'
                match_info['BTTS_Prob'] = f"{best_btts:.1%}"
            else:
                match_info['BTTS_Pick'] = '-'
                match_info['BTTS_Prob'] = f"{best_btts:.1%}" if best_btts > 0 else '-'
        else:
            match_info['BTTS_Pick'] = '-'
            match_info['BTTS_Prob'] = '-'

        # Only include matches with at least one high-confidence market
        if has_high_conf:
            rows.append(match_info)

    if not rows:
        print("[COMBINED] No matches with 90%+ confidence in 1X2, OU2.5, OU1.5, or BTTS")
        return

    # Create DataFrame and save
    combined_df = pd.DataFrame(rows)
    combined_df = combined_df.sort_values(['Date', 'League'])

    # Save CSV
    csv_path = path / "high_confidence_combined.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"[OK] Saved combined CSV: {csv_path} ({len(combined_df)} matches)")

    # Generate HTML report
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>High Confidence Picks (90%+) - 1X2, OU2.5, OU1.5, BTTS</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ color: #00d4ff; text-align: center; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #16213e; }}
        th {{ background: #0f3460; color: #00d4ff; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #0f3460; }}
        tr:hover {{ background: #1f4068; }}
        .pick {{ font-weight: bold; color: #00ff88; }}
        .no-pick {{ color: #666; }}
        .prob {{ font-size: 0.9em; color: #aaa; }}
        .high-prob {{ color: #00ff88; font-weight: bold; }}
        .summary {{ text-align: center; margin: 20px 0; color: #aaa; }}
    </style>
</head>
<body>
    <h1>High Confidence Combined Picks (90%+)</h1>
    <p class='summary'>{len(combined_df)} matches with at least one market above 90% confidence</p>
    <table>
        <tr>
            <th>Date</th>
            <th>League</th>
            <th>Match</th>
            <th>1X2</th>
            <th>O/U 2.5</th>
            <th>O/U 1.5</th>
            <th>BTTS</th>
        </tr>
"""

    for _, row in combined_df.iterrows():
        # Format each cell
        x1x2_class = 'pick high-prob' if row['1X2_Pick'] != '-' else 'no-pick'
        ou25_class = 'pick high-prob' if row['OU25_Pick'] != '-' else 'no-pick'
        ou15_class = 'pick high-prob' if row['OU15_Pick'] != '-' else 'no-pick'
        btts_class = 'pick high-prob' if row['BTTS_Pick'] != '-' else 'no-pick'

        html_content += f"""        <tr>
            <td>{row['Date']}</td>
            <td>{row['League']}</td>
            <td>{row['Home']} vs {row['Away']}</td>
            <td class='{x1x2_class}'>{row['1X2_Pick']} <span class='prob'>({row['1X2_Prob']})</span></td>
            <td class='{ou25_class}'>{row['OU25_Pick']} <span class='prob'>({row['OU25_Prob']})</span></td>
            <td class='{ou15_class}'>{row['OU15_Pick']} <span class='prob'>({row['OU15_Prob']})</span></td>
            <td class='{btts_class}'>{row['BTTS_Pick']} <span class='prob'>({row['BTTS_Prob']})</span></td>
        </tr>
"""

    html_content += """    </table>
</body>
</html>"""

    html_path = path / "high_confidence_combined.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[OK] Saved combined HTML: {html_path}")


def _write_enhanced_html(df: pd.DataFrame, path: Path, secondary_path: Path = None):
    """Enhanced HTML report - Elite picks sorted by date and league.

    Only considers core betting markets (1X2, O/U 1.5-4.5, BTTS) to avoid
    trivial markets like O/U 0.5 or niche exact scores dominating the output.
    """
    # Core market suffixes we actually care about
    _CORE_SUFFIXES = {
        '1X2_H', '1X2_D', '1X2_A',
        'BTTS_Y', 'BTTS_N',
        'OU_1_5_O', 'OU_1_5_U',
        'OU_2_5_O', 'OU_2_5_U',
        'OU_3_5_O', 'OU_3_5_U',
        'OU_4_5_O', 'OU_4_5_U',
    }

    def _is_core_col(col):
        for suffix in _CORE_SUFFIXES:
            if col.endswith(suffix):
                return True
        return False

    prob_cols = [c for c in df.columns
                 if (c.startswith("BLEND_") or c.startswith("P_") or c.startswith("DC_"))
                 and _is_core_col(c)]
    if not prob_cols:
        print("Warning: No core probability columns found")
        return

    df2 = df.copy()
    df2["BestProb"] = df2[prob_cols].max(axis=1)
    df2["BestMarket"] = df2[prob_cols].idxmax(axis=1)

    # Add confidence if available
    conf_cols = [c for c in df2.columns if c.startswith("CONF_")]
    if conf_cols:
        df2["AvgConfidence"] = df2[conf_cols].mean(axis=1)
    else:
        df2["AvgConfidence"] = 0.5

    # Show all predictions with meaningful confidence (>70% on core markets)
    elite_threshold = 0.70
    elite = df2[df2["BestProb"] >= elite_threshold].copy()

    # Sort by Date, then League
    if 'Date' in elite.columns:
        elite['Date'] = pd.to_datetime(elite['Date'], errors='coerce')
        elite = elite.sort_values(['Date', 'League'], ascending=[True, True])

    # Count stats
    total_fixtures = len(df2)
    elite_count = len(elite)
    very_high = len(df2[df2["BestProb"] >= 0.90])
    high_conf = len(df2[(df2["BestProb"] >= 0.75) & (df2["AvgConfidence"] >= 0.70)])
    
    def parse_market(market_name):
        market_name = market_name.replace("BLEND_", "").replace("P_", "").replace("DC_", "")
        
        if "1X2" in market_name:
            if market_name.endswith("_H"): return "1X2", "Home Win"
            elif market_name.endswith("_D"): return "1X2", "Draw"
            elif market_name.endswith("_A"): return "1X2", "Away Win"
        elif "BTTS" in market_name:
            if market_name.endswith("_Y"): return "BTTS", "Yes"
            elif market_name.endswith("_N"): return "BTTS", "No"
        elif "OU_" in market_name:
            parts = market_name.split("_")
            if len(parts) >= 3:
                line = parts[1] + "." + parts[2]
                if market_name.endswith("_O"): return f"O/U {line}", "Over"
                elif market_name.endswith("_U"): return f"O/U {line}", "Under"
        elif "GR_" in market_name:
            goal_range = market_name.split("_")[-1]
            return "Goals", f"{goal_range}"
        elif "CS_" in market_name:
            if market_name.endswith("_Other"): return "Score", "Other"
            parts = market_name.split("_")
            if len(parts) >= 3:
                return "Score", f"{parts[-2]}-{parts[-1]}"
        
        return market_name, ""
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <title>[TARGET] ULTIMATE Predictions - {elite_count} Elite Picks</title>
    <style>
        * {{box-sizing: border-box; margin: 0; padding: 0;}}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        .stats {{
            background: #f8f9fa;
            padding: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            border-bottom: 3px solid #667eea;
        }}
        .stat-box {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-box .number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .stat-box .label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #667eea;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        tr:hover {{
            background: #f8f9fa;
            transform: scale(1.01);
            transition: all 0.2s;
        }}
        .rank {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.2em;
        }}
        .elite {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
        }}
        .high {{
            background: #d4edda;
        }}
        .medium {{
            background: #fff3cd;
        }}
        .prob {{
            font-weight: bold;
            font-size: 1.3em;
            color: #dc3545;
        }}
        .confidence {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .conf-high {{
            background: #28a745;
            color: white;
        }}
        .conf-med {{
            background: #ffc107;
            color: #333;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: bold;
            margin-left: 5px;
        }}
        .badge-blend {{
            background: #667eea;
            color: white;
        }}
        @media print {{
            body {{background: white; padding: 0;}}
            .container {{box-shadow: none;}}
        }}
    </style>
</head>
<body>
    <div class='container'>
        <div class='header'>
            <h1>[TARGET] ULTIMATE PREDICTIONS</h1>
            <p style='font-size: 1.2em; opacity: 0.9;'>Maximum Accuracy System - Top {elite_count} Elite Picks</p>
        </div>

        <div class='stats'>
            <div class='stat-box'>
                <div class='number'>{total_fixtures}</div>
                <div class='label'>Total Fixtures</div>
            </div>
            <div class='stat-box'>
                <div class='number'>{elite_count}</div>
                <div class='label'>Elite (85%+)</div>
            </div>
            <div class='stat-box'>
                <div class='number'>{very_high}</div>
                <div class='label'>Very High (90%+)</div>
            </div>
            <div class='stat-box'>
                <div class='number'>{high_conf}</div>
                <div class='label'>High Confidence</div>
            </div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Date</th>
                    <th>League</th>
                    <th>Fixture</th>
                    <th>Market</th>
                    <th>Pick</th>
                    <th>Probability</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>"""
    
    for i, (_, r) in enumerate(elite.iterrows(), 1):
        market, selection = parse_market(r['BestMarket'])
        prob = r['BestProb']
        conf = r.get('AvgConfidence', 0.5)

        # Row styling - all are elite (85%+), differentiate by 90%+ and confidence
        if prob >= 0.95:
            row_class = "elite"  # 95%+ = gold tier
        elif prob >= 0.90:
            row_class = "high"   # 90-95% = green tier
        else:
            row_class = "medium" # 85-90% = yellow tier

        # Confidence badge
        if conf >= 0.7:
            conf_class = "conf-high"
        else:
            conf_class = "conf-med"

        # Source badge
        source = "BLEND" if r['BestMarket'].startswith("BLEND_") else ("DC" if r['BestMarket'].startswith("DC_") else "ML")

        html += f"""
                <tr class='{row_class}'>
                    <td class='rank'>{i}</td>
                    <td>{str(r['Date']).split()[0]}</td>
                    <td><strong>{r['League']}</strong></td>
                    <td>{r['HomeTeam']}<br><small>vs {r['AwayTeam']}</small></td>
                    <td>{market}</td>
                    <td>{selection} <span class='badge badge-blend'>{source}</span></td>
                    <td class='prob'>{prob:.1%}</td>
                    <td><span class='confidence {conf_class}'>{conf:.0%}</span></td>
                </tr>"""

    # Handle empty elite list
    if elite_count == 0:
        html += """
                <tr>
                    <td colspan='8' style='text-align: center; padding: 40px; color: #666;'>
                        No predictions above 85% threshold for these fixtures.
                        Check weekly_bets_full.csv for all predictions.
                    </td>
                </tr>"""

    html += """
            </tbody>
        </table>
    </div>
</body>
</html>"""

    out_path = path / "elite_picks.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote ULTIMATE HTML -> {out_path}")
    
    if secondary_path:
        secondary_path.mkdir(parents=True, exist_ok=True)
        secondary_out = secondary_path / "elite_picks.html"
        secondary_out.write_text(html, encoding="utf-8")
        print(f"[OK] Wrote ULTIMATE HTML (copy) -> {secondary_out}")

def predict_week(fixtures_csv: Path) -> Path:
    """ULTIMATE prediction pipeline"""

    log_header("[TARGET] ULTIMATE WEEKLY PREDICTIONS")
    print("Maximum Accuracy Features:")
    print("  * League-specific calibration")
    print("  * Cross-market constraints")
    print("  * Poisson adjustments")
    print("  * Time-weighted form")
    print("  * Dynamic blend weights")
    print("  * Confidence scoring")
    print("  * Injury impact adjustments\n")

    # Load models
    models = load_trained_targets()
    if not models:
        raise RuntimeError("No trained models found!")

    # Load fixtures
    fx = pd.read_csv(fixtures_csv)
    fx["Date"] = pd.to_datetime(fx["Date"])

    # Check for injury data
    has_injury_data = 'home_injuries' in fx.columns and 'away_injuries' in fx.columns
    if has_injury_data:
        total_injuries = fx['home_injuries'].sum() + fx['away_injuries'].sum()
        print(f"[LIVE] Injury data available: {total_injuries} total injuries across fixtures")
    else:
        print("[INFO] No injury data - run closer to match time for live data")
    
    # Build features
    log_header("BUILD ENHANCED FEATURES")
    df_future = _build_future_frame(fixtures_csv)
    
    # Generate ML predictions
    log_header("GENERATE ML PREDICTIONS")
    preds = model_predict(models, df_future)
    
    # Map predictions with all enhancements
    log_header("APPLY ENHANCEMENTS")
    # Pass df_future (sorted) to _map_preds_to_columns so league/xG lookups align
    # with prediction row order (df_future is sorted by League/Date/HomeTeam)
    rows, out_cols = _map_preds_to_columns(models, preds, df_future)

    # Create output — use df_future for ID columns (same sort order as predictions)
    df_out = pd.DataFrame(rows, columns=out_cols)
    for col in ID_COLS:
        if col in df_future.columns:
            df_out[col] = df_future[col].values[:len(df_out)]

    # Carry real kickoff time through to outputs — the betting layer needs it
    # for in-play timing instead of assuming a default kickoff hour.
    if "Time" in fx.columns:
        df_out = df_out.merge(
            fx[ID_COLS + ["Time"]].drop_duplicates(subset=ID_COLS),
            on=ID_COLS, how="left",
        )
    # Carry fixture_id (odds joins) and Glicko RDs (unknown-team bet gate)
    if "fixture_id" in fx.columns:
        df_out = df_out.merge(
            fx[ID_COLS + ["fixture_id"]].drop_duplicates(subset=ID_COLS),
            on=ID_COLS, how="left",
        )
    for _rd_col in ("Home_GlickoRD", "Away_GlickoRD"):
        if _rd_col in df_future.columns:
            df_out[_rd_col] = df_future[_rd_col].values[:len(df_out)]

    # Games played this season (season cold-start gate for cards/corners — see
    # tools/best_bets.py's _season_too_thin)
    for _sg_col in ("Home_SeasonGames", "Away_SeasonGames"):
        if _sg_col in df_future.columns:
            df_out[_sg_col] = df_future[_sg_col].values[:len(df_out)]
    
    # Add DC predictions
    log_header("GENERATE DC PREDICTIONS")
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fixtures_copy = OUTPUT_DIR / "upcoming_fixtures.csv"
        fx.to_csv(fixtures_copy, index=False)
        
        dc_path = build_dc_for_fixtures(fixtures_copy)
        dc_df = pd.read_csv(dc_path)
        dc_df["Date"] = pd.to_datetime(dc_df["Date"])

        dc_cols = [c for c in dc_df.columns if c.startswith("DC_")]
        merge_keys = [k for k in ["League", "Date", "HomeTeam", "AwayTeam"] if k in dc_df.columns]
        df_out["Date"] = pd.to_datetime(df_out["Date"])
        df_out = df_out.merge(
            dc_df[merge_keys + dc_cols], on=merge_keys, how="left", suffixes=("", "_dc")
        )
        print(f"[OK] Merged {len(dc_cols)} DC predictions")
    except Exception as e:
        print(f"[WARN] DC predictions failed: {e}")

    # Add NB (negative-binomial count model) predictions for corners/cards —
    # the genuine second signal for these markets, replacing the old alpha=1.0
    # ML-only passthrough.
    log_header("GENERATE NB PREDICTIONS")
    try:
        nb_targets = [t for t in models.keys() if nb_supported(t)]
        nb_df = build_nb_for_frame(df_future, nb_targets)
        for col in nb_df.columns:
            df_out[col] = nb_df[col].values[:len(df_out)]
        print(f"[OK] Merged {len(nb_df.columns)} NB predictions")
    except Exception as e:
        print(f"[WARN] NB predictions failed: {e}")

    # Apply enhanced blending
    log_header("APPLY DYNAMIC BLENDING")
    df_out = _apply_blend(df_out)

    # Market anchoring: merge live odds where available, record EDGE_* columns
    # (model minus market probability) and optionally shrink toward the market
    # (TUNING_OVERRIDES['market_anchor_weight'], default 0 = diagnostics only).
    log_header("APPLY MARKET ANCHOR")
    try:
        from odds_utils import merge_odds_into_df
        from market_anchor import apply_market_anchor
        df_out = merge_odds_into_df(df_out)
        df_out = apply_market_anchor(df_out)
    except Exception as e:
        print(f"[WARN] Market anchor skipped: {e}")

    # Apply injury adjustments (if injury data available)
    log_header("APPLY INJURY ADJUSTMENTS")
    df_out = apply_injury_adjustments(df_out, fx)

    # Calculate confidence scores
    log_header("CALCULATE CONFIDENCE")
    df_out = calculate_confidence_scores(df_out)

    # Max probability across MEANINGFUL markets for filtering.
    # Excludes OU_0_5 (Over 0.5 is ~97% in every match, which previously made
    # MaxConfidence ≈ 0.97 for all rows and useless as a filter) and untrained
    # markets (whose columns are all-zero placeholders).
    p_cols = [col for col in df_out.columns
              if col.startswith('P_') and not col.startswith('P_OU_0_5')]
    if p_cols:
        nonzero = [c for c in p_cols if pd.to_numeric(df_out[c], errors='coerce').fillna(0).abs().sum() > 0]
        if nonzero:
            df_out['MaxConfidence'] = df_out[nonzero].max(axis=1)

    # Sort by Date, then League for better readability
    if 'Date' in df_out.columns:
        df_out['Date'] = pd.to_datetime(df_out['Date'], errors='coerce')
        df_out = df_out.sort_values(['Date', 'League'], ascending=[True, True])
        print("[OK] Sorted output by Date and League")

    # Save full version with all columns (NO FILTER - keep everything)
    output_path_full = OUTPUT_DIR / "weekly_bets_full.csv"
    df_out.to_csv(output_path_full, index=False)
    print(f"\n[OK] Saved full predictions: {output_path_full} ({len(df_out)} matches)")

    # Save weekly_bets.csv with all matches (no confidence filter)
    output_path = OUTPUT_DIR / "weekly_bets.csv"
    df_out.to_csv(output_path, index=False)
    print(f"[OK] Saved predictions: {output_path} ({len(df_out)} matches)")

    # Create combined high-confidence output for 1X2, OU2.5, and OU1.5 markets
    _write_combined_high_confidence(df_out, OUTPUT_DIR)
    
    # Generate HTML
    log_header("GENERATE REPORTS")
    onedrive_path = None  # Optional: set to custom path if needed
    _write_enhanced_html(df_out, OUTPUT_DIR, onedrive_path)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"[CHART] ULTIMATE PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total matches: {len(df_out)}")
    print(f"Leagues: {df_out['League'].unique().tolist() if 'League' in df_out.columns else 'N/A'}")
    
    if 'AvgConfidence' in df_out.columns:
        high_conf = df_out[df_out['AvgConfidence'] > 0.7]
        print(f"High confidence (>70%): {len(high_conf)}")
        print(f"Average confidence: {df_out['AvgConfidence'].mean():.1%}")
    
    blend_cols = [c for c in df_out.columns if c.startswith('BLEND_')]
    print(f"BLEND predictions: {len(blend_cols)}")
    print(f"{'='*60}\n")
    
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures_csv", type=str, default="outputs/upcoming_fixtures.csv")
    args = parser.parse_args()
    
    predict_week(Path(args.fixtures_csv))