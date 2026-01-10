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

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from scipy.stats import poisson
from datetime import datetime, timedelta

from config import FEATURES_PARQUET, OUTPUT_DIR, MODEL_ARTIFACTS_DIR, log_header
from models import load_trained_targets, predict_proba as model_predict
from dc_predict import build_dc_for_fixtures
from progress_utils import heartbeat
from blending import BLEND_WEIGHTS_JSON

ID_COLS = ["League","Date","HomeTeam","AwayTeam"]
OU_LINES = ["0_5","1_5","2_5","3_5","4_5"]
AH_LINES = ["-1_0","-0_5","0_0","+0_5","+1_0"]

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
    """Load historical features with preprocessing"""
    df = pd.read_parquet(FEATURES_PARQUET)
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["League","Date"])

def calculate_league_profiles(df: pd.DataFrame) -> Dict:
    """Calculate actual league profiles from historical data"""
    profiles = {}
    
    for league in df['League'].unique():
        league_data = df[df['League'] == league]
        if len(league_data) < 50:
            continue
            
        total_goals = league_data['FTHG'].fillna(0) + league_data['FTAG'].fillna(0)
        home_wins = (league_data['FTR'] == 'H').mean()
        away_wins = (league_data['FTR'] == 'A').mean()
        
        profiles[league] = {
            'avg_goals': total_goals.mean(),
            'home_adv': home_wins - away_wins,
            'btts_rate': ((league_data['FTHG'] > 0) & (league_data['FTAG'] > 0)).mean(),
            'over25_rate': (total_goals > 2.5).mean(),
            'over15_rate': (total_goals > 1.5).mean(),
            'over35_rate': (total_goals > 3.5).mean(),
            'over45_rate': (total_goals > 4.5).mean(),
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
    calibration_weight = 0.3 * (1 - confidence)  # More calibration when less confident

    # Style-based adjustments
    style_boost = {'attacking': 0.08, 'balanced': 0.0, 'defensive': -0.08}
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
        return min(prob * (1 + home_adv * 0.3), 0.95)

    elif '1X2_A' in market:
        return max(prob * (1 - home_adv * 0.3), 0.05)

    elif '1X2_D' in market:
        # Draw less likely in leagues with high home advantage
        return max(0.05, min(0.45, prob * (1 - home_adv * 0.15)))

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
    
    # 2. BTTS and O/U 0.5 logical consistency
    if 'P_BTTS_Y' in row and 'P_OU_0_5_U' in row:
        if pd.notna(row['P_BTTS_Y']) and pd.notna(row['P_OU_0_5_U']):
            # BTTS=Yes implies Over 0.5 must be very high
            if row['P_BTTS_Y'] > 0.7:
                row['P_OU_0_5_U'] = min(row['P_OU_0_5_U'], 0.02)
                row['P_OU_0_5_O'] = max(row['P_OU_0_5_O'], 0.98)
            
            # Under 0.5 high means BTTS=Yes must be zero
            if row['P_OU_0_5_U'] > 0.5:
                row['P_BTTS_Y'] = 0.0
                row['P_BTTS_N'] = 1.0
    
    # 3. BTTS and O/U 1.5 consistency
    if 'P_BTTS_Y' in row and 'P_OU_1_5_O' in row:
        if pd.notna(row['P_BTTS_Y']) and pd.notna(row['P_OU_1_5_O']):
            # BTTS=Yes requires at least 2 goals (Over 1.5)
            if row['P_BTTS_Y'] > 0.6:
                row['P_OU_1_5_O'] = max(row['P_OU_1_5_O'], row['P_BTTS_Y'] * 0.9)
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
    
    # 6. Team goals and BTTS consistency
    if all(col in row for col in ['P_BTTS_Y', 'P_HomeTG_0_5_O', 'P_AwayTG_0_5_O']):
        if all(pd.notna(row[col]) for col in ['P_BTTS_Y', 'P_HomeTG_0_5_O', 'P_AwayTG_0_5_O']):
            # BTTS requires both teams to score
            min_btts = row['P_HomeTG_0_5_O'] * row['P_AwayTG_0_5_O']
            row['P_BTTS_Y'] = max(row['P_BTTS_Y'], min_btts * 0.85)
            row['P_BTTS_N'] = 1 - row['P_BTTS_Y']
    
    return row

def apply_poisson_adjustment(row: pd.Series, home_xg: float = None, away_xg: float = None, league: str = None) -> pd.Series:
    """Apply Poisson distribution for goal-based markets"""
    row = row.copy()
    
    # Use league-specific or default xG
    if league and league in LEAGUE_PROFILES:
        profile = LEAGUE_PROFILES[league]
        total_expected = profile['avg_goals']
        home_share = 0.54  # Home advantage ~54% of goals
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
        if line == '0_5':
            blend_weight = 0.3  # Less Poisson influence (nearly always over)
        elif line == '1_5':
            blend_weight = 0.4
        elif line == '2_5':
            blend_weight = 0.5  # Equal blend
        elif line == '3_5':
            blend_weight = 0.4
        else:  # 4_5
            blend_weight = 0.3
        
        if f'P_OU_{line}_O' in row and pd.notna(row[f'P_OU_{line}_O']):
            row[f'P_OU_{line}_O'] = row[f'P_OU_{line}_O'] * (1 - blend_weight) + poisson_over * blend_weight
            row[f'P_OU_{line}_U'] = 1 - row[f'P_OU_{line}_O']
    
    # BTTS using Poisson
    prob_home_scores = 1 - poisson.pmf(0, home_xg)
    prob_away_scores = 1 - poisson.pmf(0, away_xg)
    poisson_btts = prob_home_scores * prob_away_scores
    
    if 'P_BTTS_Y' in row and pd.notna(row['P_BTTS_Y']):
        row['P_BTTS_Y'] = row['P_BTTS_Y'] * 0.65 + poisson_btts * 0.35
        row['P_BTTS_N'] = 1 - row['P_BTTS_Y']
    
    return row

def _build_future_frame(fixtures_csv: Path) -> pd.DataFrame:
    """Enhanced feature building with time weighting"""
    base = _load_base_features()
    fx = pd.read_csv(fixtures_csv)
    fx["Date"] = pd.to_datetime(fx["Date"])
    
    # Add time weights for recent form emphasis
    current_date = datetime.now()
    base['days_ago'] = (current_date - pd.to_datetime(base['Date'])).dt.days
    base['time_weight'] = np.exp(-base['days_ago'] / 180)  # 180-day half-life
    
    rows = []
    for _, r in fx.iterrows():
        lg, dt, ht, at = r["League"], r["Date"], r["HomeTeam"], r["AwayTeam"]
        hist_lg = base[base["League"] == lg]
        
        # Get last 10 games for each team with time weighting
        hrow = hist_lg[(hist_lg["HomeTeam"]==ht) | (hist_lg["AwayTeam"]==ht)]
        hrow = hrow[hrow["Date"]<dt].sort_values("Date").tail(10)
        
        arow = hist_lg[(hist_lg["HomeTeam"]==at) | (hist_lg["AwayTeam"]==at)]
        arow = arow[arow["Date"]<dt].sort_values("Date").tail(10)
        
        if hrow.empty or arow.empty:
            continue
        
        feat_cols = [c for c in base.columns if not c.startswith("y_") 
                     and c not in ["FTHG","FTAG","FTR","HTHG","HTAG","HTR","days_ago","time_weight"]]
        
        fused = pd.DataFrame()
        
        # Calculate weighted features for home team
        for col in feat_cols:
            if col in hrow.columns and hrow[col].dtype in ['float64', 'int64']:
                weights = hrow['time_weight'].values[-5:]
                values = hrow[col].fillna(0).values[-5:]
                if weights.sum() > 0:
                    weighted_avg = np.average(values, weights=weights)
                    fused.at[0, col] = weighted_avg
                else:
                    fused.at[0, col] = hrow[col].iloc[-1] if len(hrow) > 0 else 0
            else:
                fused.at[0, col] = hrow[col].iloc[-1] if len(hrow) > 0 else 0
        
        # Update away team features
        for c in fused.columns:
            if c.startswith("Away_") and c in arow.columns:
                if arow[c].dtype in ['float64', 'int64']:
                    weights = arow['time_weight'].values[-5:]
                    values = arow[c].fillna(0).values[-5:]
                    if weights.sum() > 0:
                        weighted_avg = np.average(values, weights=weights)
                        fused.at[0, c] = weighted_avg
                else:
                    fused.at[0, c] = arow[c].iloc[-1] if len(arow) > 0 else 0
        
        fused["League"] = lg
        fused["Date"] = dt
        fused["HomeTeam"] = ht
        fused["AwayTeam"] = at
        
        for c in base.columns:
            if c.startswith("y_"):
                fused[c] = pd.NA
        
        rows.append(fused)
    
    if not rows:
        raise RuntimeError("No fixtures matched with historical features.")
    
    return pd.concat(rows, ignore_index=True).sort_values(["League","Date","HomeTeam"])

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
        
        # Apply Poisson adjustments
        row_series = apply_poisson_adjustment(row_series, league=league)
        
        # Enforce cross-market constraints
        row_series = enforce_cross_market_constraints(row_series)
        
        rows.append(row_series.to_dict())
    
    return rows, out_cols

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

    def pair_cols_for_target(target: str) -> Tuple[List[str], List[str]]:
        if target == "y_1X2":
            ml_cols = ["P_1X2_H","P_1X2_D","P_1X2_A"]
            dc_cols = ["DC_1X2_H","DC_1X2_D","DC_1X2_A"]
        elif target == "y_BTTS":
            ml_cols = ["P_BTTS_N","P_BTTS_Y"]
            dc_cols = ["DC_BTTS_N","DC_BTTS_Y"]
        elif target == "y_GOAL_RANGE":
            ml_cols = [f"P_GR_{k}" for k in ["0","1","2","3","4","5+"]]
            dc_cols = [f"DC_GR_{k}" for k in ["0","1","2","3","4","5+"]]
        elif target == "y_CS":
            ml_cols = [f"P_CS_{a}_{b}" for a in range(6) for b in range(6)] + ["P_CS_Other"]
            dc_cols = [f"DC_CS_{a}_{b}" for a in range(6) for b in range(6)] + ["DC_CS_Other"]
        elif target.startswith("y_OU_"):
            line_part = target.replace("y_OU_", "")
            ml_cols = [f"P_OU_{line_part}_U", f"P_OU_{line_part}_O"]
            dc_cols = [f"DC_OU_{line_part}_U", f"DC_OU_{line_part}_O"]
        elif target.startswith("y_AH_"):
            line_part = target.replace("y_AH_", "")
            ml_cols = [f"P_AH_{line_part}_A", f"P_AH_{line_part}_P", f"P_AH_{line_part}_H"]
            dc_cols = [f"DC_AH_{line_part}_A", f"DC_AH_{line_part}_P", f"DC_AH_{line_part}_H"]
        else:
            return [], []
        return ml_cols, dc_cols

    print("Creating enhanced BLEND predictions with dynamic weights...")
    
    # Apply blending row by row with league-specific adjustments
    for idx in range(len(out)):
        league = out.iloc[idx]['League'] if 'League' in out.columns else None
        
        # Determine ML weight adjustment based on league quality
        ml_weight_boost = 0.0
        if league in LEAGUE_PROFILES:
            quality = LEAGUE_PROFILES[league].get('quality', 'medium')
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
            
            if missing_ml or missing_dc:
                continue
            
            # Adjust alpha based on league quality
            alpha = min(float(base_alpha) + ml_weight_boost, 0.85)
            
            # Get probabilities
            M = out.loc[idx, ml_cols].values
            D = out.loc[idx, dc_cols].values
            
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

def _write_enhanced_html(df: pd.DataFrame, path: Path, secondary_path: Path = None):
    """Enhanced HTML report - Elite picks (>85% probability) sorted by date and league"""
    prob_cols = [c for c in df.columns if c.startswith("BLEND_") or c.startswith("P_") or c.startswith("DC_")]
    if not prob_cols:
        print("Warning: No probability columns found")
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

    # Filter for ELITE predictions only (>85% probability)
    elite_threshold = 0.85
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
    <title>[TARGET] ULTIMATE Predictions - {len(top)} Elite Picks</title>
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
            <p style='font-size: 1.2em; opacity: 0.9;'>Maximum Accuracy System - Top {len(top)} Elite Picks</p>
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
    print("  * Confidence scoring\n")
    
    # Load models
    models = load_trained_targets()
    if not models:
        raise RuntimeError("No trained models found!")
    
    # Load fixtures
    fx = pd.read_csv(fixtures_csv)
    fx["Date"] = pd.to_datetime(fx["Date"])
    
    # Build features
    log_header("BUILD ENHANCED FEATURES")
    df_future = _build_future_frame(fixtures_csv)
    
    # Generate ML predictions
    log_header("GENERATE ML PREDICTIONS")
    preds = model_predict(models, df_future)
    
    # Map predictions with all enhancements
    log_header("APPLY ENHANCEMENTS")
    rows, out_cols = _map_preds_to_columns(models, preds, fx)
    
    # Create output
    df_out = pd.DataFrame(rows, columns=out_cols)
    for col in ID_COLS:
        if col in fx.columns:
            df_out[col] = fx[col].values[:len(df_out)]
    
    # Add DC predictions
    log_header("GENERATE DC PREDICTIONS")
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fixtures_copy = OUTPUT_DIR / "upcoming_fixtures.csv"
        fx.to_csv(fixtures_copy, index=False)
        
        dc_path = build_dc_for_fixtures(fixtures_copy)
        dc_df = pd.read_csv(dc_path)
        
        dc_cols = [c for c in dc_df.columns if c.startswith("DC_")]
        for col in dc_cols:
            if col in dc_df.columns:
                df_out[col] = dc_df[col].values[:len(df_out)]
        
        print(f"[OK] Merged {len(dc_cols)} DC predictions")
    except Exception as e:
        print(f"[WARN] DC predictions failed: {e}")
    
    # Apply enhanced blending
    log_header("APPLY DYNAMIC BLENDING")
    df_out = _apply_blend(df_out)
    
    # Calculate confidence scores
    log_header("CALCULATE CONFIDENCE")
    df_out = calculate_confidence_scores(df_out)

    # Calculate max probability across all P_ columns for filtering
    p_cols = [col for col in df_out.columns if col.startswith('P_')]
    if p_cols:
        df_out['MaxConfidence'] = df_out[p_cols].max(axis=1)

    # Sort by Date, then League for better readability
    if 'Date' in df_out.columns:
        df_out['Date'] = pd.to_datetime(df_out['Date'], errors='coerce')
        df_out = df_out.sort_values(['Date', 'League'], ascending=[True, True])
        print("[OK] Sorted output by Date and League")

    # Save full version with all columns (NO FILTER - keep everything)
    output_path_full = OUTPUT_DIR / "weekly_bets_full.csv"
    df_out.to_csv(output_path_full, index=False)
    print(f"\n[OK] Saved full predictions: {output_path_full} ({len(df_out)} matches)")

    # Filter for weekly_bets.csv: Keep only predictions with >= 60% confidence
    if 'MaxConfidence' in df_out.columns:
        df_filtered = df_out[df_out['MaxConfidence'] >= 0.60].copy()
        filtered_count = len(df_out) - len(df_filtered)
        print(f"[FILTER] Removed {filtered_count} matches with confidence < 60%")
    else:
        df_filtered = df_out.copy()

    # Save filtered version as weekly_bets.csv for compatibility with other scripts
    output_path = OUTPUT_DIR / "weekly_bets.csv"
    df_filtered.to_csv(output_path, index=False)
    print(f"[OK] Saved filtered predictions: {output_path} ({len(df_filtered)} matches >= 60% confidence)")
    
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