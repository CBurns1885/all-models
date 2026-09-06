# features.py - ENHANCED VERSION
# Leak-free historical feature engineering with API-Football advanced statistics
# Creates 200+ features for maximum prediction accuracy

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    PROCESSED_DIR, FEATURES_PARQUET, HISTORICAL_PARQUET,
    TRAIN_SEASONS_BACK, USE_ELO, USE_ROLLING_FORM, USE_MARKET_FEATURES,
    USE_XG_FEATURES, USE_ADVANCED_STATS, FORM_WINDOWS, EWM_SPAN,
    ALL_CUPS, EUROPEAN_CUPS, log_header
)
from api_football_adapter import get_injury_counts_from_db

# -----------------------------
# Utilities
# -----------------------------

RESULT_MAP = {"H": 1, "D": 0, "A": -1}

def _points_from_ftr(ftr: pd.Series) -> pd.Series:
    return ftr.map({"H": 3, "D": 1, "A": 0}).fillna(0).astype(int)

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# -----------------------------
# Elo rating (enhanced with momentum)
# -----------------------------

@dataclass
class EloConfig:
    base_rating: float = 1500.0
    k_base: float = 20.0
    home_adv: float = 65.0
    margin_factor: float = 0.5  # Goal margin influence
    momentum_decay: float = 0.95  # Recent form weight

def _expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(ra - rb) / 400.0))

def _elo_by_league(df: pd.DataFrame, cfg: EloConfig) -> pd.DataFrame:
    """Compute league-specific Elo ratings with momentum adjustment.

    For cup competitions, uses league Elo as fallback when cup history is limited.
    """
    from config import ALL_CUPS, CUP_TO_LEAGUE

    df = df.sort_values(["League","Date"]).copy()

    state: Dict[Tuple[str,str], float] = {}
    momentum: Dict[Tuple[str,str], float] = {}
    match_count: Dict[Tuple[str,str], int] = {}  # Track matches per team/league

    home_elos, away_elos, home_mom, away_mom = [], [], [], []

    for idx, row in df.iterrows():
        lg = row["League"]
        ht, at = row["HomeTeam"], row["AwayTeam"]
        key_h, key_a = (lg, ht), (lg, at)

        # Get base Elo for this competition
        ra = state.get(key_h, cfg.base_rating)
        rb = state.get(key_a, cfg.base_rating)

        # For cup games: fallback to league Elo if cup history is limited
        if lg in ALL_CUPS:
            primary_league = CUP_TO_LEAGUE.get(lg)
            if primary_league:
                # Check if team has limited cup history (< 5 matches)
                if match_count.get(key_h, 0) < 5:
                    league_elo_h = state.get((primary_league, ht), cfg.base_rating)
                    if league_elo_h != cfg.base_rating:
                        # Blend: 70% league Elo, 30% cup Elo (if any)
                        cup_weight = min(match_count.get(key_h, 0) / 5.0, 1.0) * 0.3
                        ra = league_elo_h * (1 - cup_weight) + ra * cup_weight

                if match_count.get(key_a, 0) < 5:
                    league_elo_a = state.get((primary_league, at), cfg.base_rating)
                    if league_elo_a != cfg.base_rating:
                        cup_weight = min(match_count.get(key_a, 0) / 5.0, 1.0) * 0.3
                        rb = league_elo_a * (1 - cup_weight) + rb * cup_weight

        ma = momentum.get(key_h, 0.0)
        mb = momentum.get(key_a, 0.0)

        home_elos.append(ra)
        away_elos.append(rb)
        home_mom.append(ma)
        away_mom.append(mb)

        ftr = row.get("FTR")
        if pd.isna(ftr):
            continue

        # Increment match count
        match_count[key_h] = match_count.get(key_h, 0) + 1
        match_count[key_a] = match_count.get(key_a, 0) + 1

        # Calculate scores
        if ftr == "H":
            score_home = 1.0
            mom_change_h, mom_change_a = 1.0, -1.0
        elif ftr == "D":
            score_home = 0.5
            mom_change_h, mom_change_a = 0.0, 0.0
        else:
            score_home = 0.0
            mom_change_h, mom_change_a = -1.0, 1.0

        # Goal margin adjustment
        goal_diff = abs((row.get('FTHG', 0) or 0) - (row.get('FTAG', 0) or 0))
        margin_mult = 1 + cfg.margin_factor * np.log1p(goal_diff)

        ra_eff = ra + cfg.home_adv + ma * 10  # Add momentum
        rb_eff = rb + mb * 10

        k = cfg.k_base * margin_mult
        ea = _expected_score(ra_eff, rb_eff)

        ra_new = ra + k * (score_home - ea)
        rb_new = rb + k * ((1.0 - score_home) - (1.0 - ea))

        # Store new ratings (home_adv only affects expected score, not stored rating)
        state[key_h] = ra_new
        state[key_a] = rb_new

        # Update momentum
        momentum[key_h] = ma * cfg.momentum_decay + mom_change_h * (1 - cfg.momentum_decay)
        momentum[key_a] = mb * cfg.momentum_decay + mom_change_a * (1 - cfg.momentum_decay)

    out = df.copy()
    out["Elo_Home"] = home_elos
    out["Elo_Away"] = away_elos
    out["Elo_Diff"] = out["Elo_Home"] - out["Elo_Away"]
    out["Elo_Mom_Home"] = home_mom
    out["Elo_Mom_Away"] = away_mom
    out["Elo_Mom_Diff"] = out["Elo_Mom_Home"] - out["Elo_Mom_Away"]

    return out

# -----------------------------
# Rolling team form & stats (enhanced)
# -----------------------------

def _add_team_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Create unified columns for home/away perspective"""
    out = df.copy()
    
    if side == "Home":
        out["Team"] = out["HomeTeam"]
        out["Opp"] = out["AwayTeam"]
        out["GoalsFor"] = out["FTHG"]
        out["GoalsAgainst"] = out["FTAG"]
        out["Win"] = (out["FTR"] == "H").astype(int)
        out["Draw"] = (out["FTR"] == "D").astype(int)
        out["Loss"] = (out["FTR"] == "A").astype(int)
        
        # Standard stats
        out = _ensure_cols(out, ["HS","HST","HC","HY","HR"])
        out["Shots"] = out["HS"]
        out["ShotsT"] = out["HST"]
        out["Corners"] = out["HC"]
        out["CardsY"] = out["HY"]
        out["CardsR"] = out["HR"]
        out["Fouls"] = out.get("Home_Fouls", np.nan)

        # Advanced stats (from API-Football)
        out = _ensure_cols(out, ["Home_xG", "Home_Possession", "Home_Shots_Inside_Box",
                                  "Home_Big_Chances", "Home_Pass_Accuracy"])
        out["xG"] = out.get("Home_xG", np.nan)
        out["Possession"] = out.get("Home_Possession", np.nan)
        out["ShotsInBox"] = out.get("Home_Shots_Inside_Box", np.nan)
        out["BigChances"] = out.get("Home_Big_Chances", np.nan)
        out["PassAcc"] = out.get("Home_Pass_Accuracy", np.nan)

    else:  # Away
        out["Team"] = out["AwayTeam"]
        out["Opp"] = out["HomeTeam"]
        out["GoalsFor"] = out["FTAG"]
        out["GoalsAgainst"] = out["FTHG"]
        out["Win"] = (out["FTR"] == "A").astype(int)
        out["Draw"] = (out["FTR"] == "D").astype(int)
        out["Loss"] = (out["FTR"] == "H").astype(int)

        out = _ensure_cols(out, ["AS","AST","AC","AY","AR"])
        out["Shots"] = out["AS"]
        out["ShotsT"] = out["AST"]
        out["Corners"] = out["AC"]
        out["CardsY"] = out["AY"]
        out["CardsR"] = out["AR"]
        out["Fouls"] = out.get("Away_Fouls", np.nan)

        out = _ensure_cols(out, ["Away_xG", "Away_Possession", "Away_Shots_Inside_Box",
                                  "Away_Big_Chances", "Away_Pass_Accuracy"])
        out["xG"] = out.get("Away_xG", np.nan)
        out["Possession"] = out.get("Away_Possession", np.nan)
        out["ShotsInBox"] = out.get("Away_Shots_Inside_Box", np.nan)
        out["BigChances"] = out.get("Away_Big_Chances", np.nan)
        out["PassAcc"] = out.get("Away_Pass_Accuracy", np.nan)

    out["Side"] = side
    out["CleanSheet"] = (out["GoalsAgainst"] == 0).astype(int)
    out["FailedToScore"] = (out["GoalsFor"] == 0).astype(int)
    out["BTTS"] = ((out["GoalsFor"] > 0) & (out["GoalsAgainst"] > 0)).astype(int)

    cols = ["League","Date","Team","Opp","Side","GoalsFor","GoalsAgainst",
            "Win","Draw","Loss","Shots","ShotsT","Corners","CardsY","CardsR","Fouls",
            "CleanSheet","FailedToScore","BTTS",
            "xG","Possession","ShotsInBox","BigChances","PassAcc"]

    return out[[c for c in cols if c in out.columns]]

def _rolling_stats(team_df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    """Calculate comprehensive rolling statistics"""
    if windows is None:
        windows = FORM_WINDOWS
    
    team_df = team_df.sort_values("Date").copy()
    
    for w in windows:
        shifted = team_df.shift(1)
        rolled = shifted.rolling(window=w, min_periods=1)

        # Core stats
        team_df[f"GF_ma{w}"] = rolled["GoalsFor"].mean()
        team_df[f"GA_ma{w}"] = rolled["GoalsAgainst"].mean()
        team_df[f"GD_ma{w}"] = team_df[f"GF_ma{w}"] - team_df[f"GA_ma{w}"]
        # PPG: use rolling mean of points (not sum/w, which underestimates
        # when fewer than w matches are available due to min_periods=1)
        points = shifted["Win"] * 3 + shifted["Draw"]
        team_df[f"PPG_ma{w}"] = points.rolling(window=w, min_periods=1).mean()

        # Shot/discipline stats
        for col in ["Shots","ShotsT","Corners","CardsY","CardsR","Fouls"]:
            if col in team_df.columns and team_df[col].notna().any():
                team_df[f"{col}_ma{w}"] = rolled[col].mean()

        # Derived rates
        team_df[f"CleanSheet_rate{w}"] = rolled["CleanSheet"].mean()
        team_df[f"FTS_rate{w}"] = rolled["FailedToScore"].mean()
        team_df[f"BTTS_rate{w}"] = rolled["BTTS"].mean()

        # Advanced stats (if available)
        for col in ["xG", "Possession", "ShotsInBox", "BigChances", "PassAcc"]:
            if col in team_df.columns and team_df[col].notna().any():
                team_df[f"{col}_ma{w}"] = rolled[col].mean()
    
    # EWMA features — two spans to capture both burst form and stable trend
    shifted = team_df.shift(1)
    for span, tag in [(3, "ewm3"), (EWM_SPAN, "ewm")]:
        ew = shifted.ewm(span=span, adjust=False)
        team_df[f"GF_{tag}"]  = ew["GoalsFor"].mean()
        team_df[f"GA_{tag}"]  = ew["GoalsAgainst"].mean()
        team_df[f"PPG_{tag}"] = ew["Win"].mean() * 3 + ew["Draw"].mean()
        team_df[f"CleanSheet_rate_{tag}"] = ew["CleanSheet"].mean()
        team_df[f"FTS_rate_{tag}"]        = ew["FailedToScore"].mean()
        team_df[f"BTTS_rate_{tag}"]       = ew["BTTS"].mean()
        for col in ["Shots", "ShotsT", "Corners", "CardsY", "CardsR", "Fouls", "xG"]:
            if col in team_df.columns and team_df[col].notna().any():
                team_df[f"{col}_{tag}"] = ew[col].mean()

    return team_df

def _build_side_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build rolling features for both home and away perspectives"""
    home = _add_team_side(df, "Home")
    away = _add_team_side(df, "Away")
    long = pd.concat([home, away], ignore_index=True)
    
    parts = []
    for (lg, tm), g in long.groupby(["League","Team"], sort=False):
        parts.append(_rolling_stats(g))
    
    return pd.concat(parts, ignore_index=True)

def _pivot_back(match_df: pd.DataFrame, side_feats: pd.DataFrame) -> pd.DataFrame:
    """Join side features back to match rows as Home_* and Away_*"""
    key_cols = ["League","Date","HomeTeam","AwayTeam"]
    
    # Get all potential columns from original
    base_cols = key_cols + ["FTHG","FTAG","FTR","HTHG","HTAG","HTR","Season"]
    odds_cols = ["B365H","B365D","B365A","PSCH","PSCD","PSCA","AvgH","AvgD","AvgA",
                 "MaxH","MaxD","MaxA","Odds_O25","Odds_U25","Odds_BTTS_Y","Odds_BTTS_N"]
    
    available_base = [c for c in base_cols + odds_cols if c in match_df.columns]
    out = match_df[available_base].copy()
    
    # Feature columns to join (exclude identifiers)
    exclude = {"League","Date","Team","Opp","Side","GoalsFor","GoalsAgainst","Win","Draw","Loss",
               "CleanSheet","FailedToScore","BTTS"}
    feat_cols = [c for c in side_feats.columns if c not in exclude]
    
    # Home join
    hf = side_feats.query("Side == 'Home'")[["League","Date","Team"] + feat_cols].copy()
    hf = hf.rename(columns={c: f"Home_{c}" for c in feat_cols})
    out = out.merge(hf, left_on=["League","Date","HomeTeam"], right_on=["League","Date","Team"], how="left")
    if "Team" in out.columns:
        out = out.drop(columns=["Team"])
    
    # Away join
    af = side_feats.query("Side == 'Away'")[["League","Date","Team"] + feat_cols].copy()
    af = af.rename(columns={c: f"Away_{c}" for c in af.columns if c not in ["League","Date","Team"]})
    out = out.merge(af, left_on=["League","Date","AwayTeam"], right_on=["League","Date","Team"], how="left")
    if "Team" in out.columns:
        out = out.drop(columns=["Team"])
    
    return out

# -----------------------------
# Contextual features (NEW)
# -----------------------------

def _add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add contextual match features"""
    out = df.copy()

    # Cup flag — draws are half as common in cups, away wins ~40% vs 31% in leagues
    out['is_cup'] = out['League'].isin(ALL_CUPS).astype(int)
    # European flag — UCL/UEL/UECL have elite-only teams, two-leg ties, neutral finals
    out['is_european'] = out['League'].isin(EUROPEAN_CUPS).astype(int)

    # Day of week
    out['Date'] = pd.to_datetime(out['Date'])
    out['DayOfWeek'] = out['Date'].dt.dayofweek
    out['IsWeekend'] = out['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Season progress (approximate)
    out['Month'] = out['Date'].dt.month
    out['SeasonProgress'] = out['Month'].apply(lambda m: 
        (m - 8) / 10 if m >= 8 else (m + 4) / 10
    ).clip(0, 1)
    
    # Calculate rest days
    team_dates: Dict[str, pd.Timestamp] = {}
    home_rest, away_rest = [], []
    
    for idx, row in df.sort_values('Date').iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        match_date = pd.to_datetime(row['Date'])
        
        # Home team rest
        if ht in team_dates:
            days = (match_date - team_dates[ht]).days
            home_rest.append(min(days, 21))
        else:
            home_rest.append(7)  # Default
        
        # Away team rest
        if at in team_dates:
            days = (match_date - team_dates[at]).days
            away_rest.append(min(days, 21))
        else:
            away_rest.append(7)
        
        # Update last match dates
        team_dates[ht] = match_date
        team_dates[at] = match_date
    
    rest_df = pd.DataFrame({
        'Home_RestDays': home_rest,
        'Away_RestDays': away_rest
    }, index=df.sort_values('Date').index)
    
    out = out.merge(rest_df, left_index=True, right_index=True, how='left')
    out['RestDiff'] = out['Home_RestDays'] - out['Away_RestDays']
    
    return out

# -----------------------------
# Market features (from odds)
# -----------------------------

def _add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market-derived features from betting odds"""
    out = df.copy()
    
    # 1X2 implied probabilities
    for prefix in ['B365', 'PS', 'Avg', 'Max']:
        h_col, d_col, a_col = f'{prefix}H', f'{prefix}D', f'{prefix}A'
        
        if all(c in out.columns for c in [h_col, d_col, a_col]):
            total = (1/out[h_col]) + (1/out[d_col]) + (1/out[a_col])
            total = total.replace([np.inf, -np.inf], np.nan)
            
            out[f'{prefix}_Impl_H'] = (1/out[h_col]) / total
            out[f'{prefix}_Impl_D'] = (1/out[d_col]) / total
            out[f'{prefix}_Impl_A'] = (1/out[a_col]) / total
            out[f'{prefix}_Overround'] = total - 1
    
    # O/U 2.5 implied
    if 'Odds_O25' in out.columns and 'Odds_U25' in out.columns:
        total_ou = (1/out['Odds_O25']) + (1/out['Odds_U25'])
        total_ou = total_ou.replace([np.inf, -np.inf], np.nan)
        out['Impl_O25'] = (1/out['Odds_O25']) / total_ou
        out['Impl_U25'] = (1/out['Odds_U25']) / total_ou
    
    # BTTS implied
    if 'Odds_BTTS_Y' in out.columns and 'Odds_BTTS_N' in out.columns:
        total_btts = (1/out['Odds_BTTS_Y']) + (1/out['Odds_BTTS_N'])
        total_btts = total_btts.replace([np.inf, -np.inf], np.nan)
        out['Impl_BTTS_Y'] = (1/out['Odds_BTTS_Y']) / total_btts
        out['Impl_BTTS_N'] = (1/out['Odds_BTTS_N']) / total_btts
    
    return out

# -----------------------------
# Targets for multiple markets
# -----------------------------

OU_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
AH_LINES = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
TEAM_GOAL_LINES = [0.5, 1.5, 2.5, 3.5]

def _add_all_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add all target variables - COMPREHENSIVE VERSION with all betting markets"""
    out = df.copy()

    # Drop rows without a valid result — these are abandoned/postponed matches
    # that would create "nan" string classes and pollute the model
    out = out.dropna(subset=["FTR"]).copy()
    out = out[out["FTR"].isin(["H", "D", "A"])].copy()

    # Ensure numeric columns
    out["FTHG"] = pd.to_numeric(out["FTHG"], errors='coerce').fillna(0).astype(int)
    out["FTAG"] = pd.to_numeric(out["FTAG"], errors='coerce').fillna(0).astype(int)
    total = out["FTHG"] + out["FTAG"]
    goal_diff = out["FTHG"] - out["FTAG"]

    # ===========================================================================
    # CORE MARKETS
    # ===========================================================================

    # 1X2 Match Result (guaranteed to be H, D, or A after filter above)
    out["y_1X2"] = out["FTR"].astype(str)

    # BTTS (Both Teams To Score)
    btts = (out["FTHG"] > 0) & (out["FTAG"] > 0)
    out["y_BTTS"] = np.where(btts, "Y", "N")

    # Over/Under Total Goals
    for line in OU_LINES:
        line_str = str(line).replace('.', '_')
        out[f"y_OU_{line_str}"] = np.where(total > line, "O", "U")

    # Goal Range (0, 1, 2, 3, 4, 5+)
    bins = pd.cut(total, bins=[-1,0,1,2,3,4,100], labels=["0","1","2","3","4","5+"])
    out["y_GOAL_RANGE"] = bins.astype(str)

    # Exact Total Goals (0, 1, 2, 3, 4, 5, 6+)
    out["y_ExactTotal_0"] = np.where(total == 0, "Y", "N")
    out["y_ExactTotal_1"] = np.where(total == 1, "Y", "N")
    out["y_ExactTotal_2"] = np.where(total == 2, "Y", "N")
    out["y_ExactTotal_3"] = np.where(total == 3, "Y", "N")
    out["y_ExactTotal_4"] = np.where(total == 4, "Y", "N")
    out["y_ExactTotal_5"] = np.where(total == 5, "Y", "N")
    out["y_ExactTotal_6+"] = np.where(total >= 6, "Y", "N")

    # Correct Score (0-0 to 5-5 + Other)
    def get_cs(row):
        h = int(row['FTHG']) if pd.notna(row['FTHG']) else -1
        a = int(row['FTAG']) if pd.notna(row['FTAG']) else -1
        if h < 0 or a < 0:
            return 'Other'
        if h <= 5 and a <= 5:
            return f'{h}-{a}'
        return 'Other'

    out["y_CS"] = out.apply(get_cs, axis=1)

    # ===========================================================================
    # DRAW NO BET (2-way market)
    # ===========================================================================

    out["y_DNB_H"] = np.where(out["FTR"] == "H", "Y", "N")
    out["y_DNB_A"] = np.where(out["FTR"] == "A", "Y", "N")

    # ===========================================================================
    # TO SCORE MARKETS
    # ===========================================================================

    out["y_HomeToScore"] = np.where(out["FTHG"] > 0, "Y", "N")
    out["y_AwayToScore"] = np.where(out["FTAG"] > 0, "Y", "N")

    # ===========================================================================
    # HALF-TIME MARKETS
    # ===========================================================================

    # Half-time result (if available)
    if "HTHG" in out.columns and "HTAG" in out.columns:
        out["HTHG"] = pd.to_numeric(out["HTHG"], errors='coerce').fillna(0).astype(int)
        out["HTAG"] = pd.to_numeric(out["HTAG"], errors='coerce').fillna(0).astype(int)

        ht_total = out["HTHG"] + out["HTAG"]

        # HT Result
        def get_ht_result(row):
            if row['HTHG'] > row['HTAG']:
                return 'H'
            elif row['HTHG'] < row['HTAG']:
                return 'A'
            return 'D'

        out["y_HT"] = out.apply(get_ht_result, axis=1)

        # HT/FT combo
        out["y_HTFT"] = out["y_HT"].astype(str) + "-" + out["FTR"].astype(str)

        # HT Over/Under
        out["y_HT_OU_0_5"] = np.where(ht_total > 0.5, "O", "U")
        out["y_HT_OU_1_5"] = np.where(ht_total > 1.5, "O", "U")
        out["y_HT_OU_2_5"] = np.where(ht_total > 2.5, "O", "U")

        # HT BTTS
        ht_btts = (out["HTHG"] > 0) & (out["HTAG"] > 0)
        out["y_HT_BTTS"] = np.where(ht_btts, "Y", "N")

        # Second Half Goals
        sh_home = out["FTHG"] - out["HTHG"]
        sh_away = out["FTAG"] - out["HTAG"]
        sh_total = sh_home + sh_away

        out["y_2H_OU_0_5"] = np.where(sh_total > 0.5, "O", "U")
        out["y_2H_OU_1_5"] = np.where(sh_total > 1.5, "O", "U")
        out["y_2H_OU_2_5"] = np.where(sh_total > 2.5, "O", "U")

        # 2H BTTS
        sh_btts = (sh_home > 0) & (sh_away > 0)
        out["y_2H_BTTS"] = np.where(sh_btts, "Y", "N")

        # Highest Scoring Half
        out["y_HigherHalf"] = np.where(ht_total > sh_total, "1H",
                                       np.where(ht_total < sh_total, "2H", "EQ"))

        # Goals in Both Halves
        out["y_GoalsBothHalves"] = np.where((ht_total > 0) & (sh_total > 0), "Y", "N")

        # Home Scores Both Halves
        out["y_HomeScoresBothHalves"] = np.where((out["HTHG"] > 0) & (sh_home > 0), "Y", "N")

        # Away Scores Both Halves
        out["y_AwayScoresBothHalves"] = np.where((out["HTAG"] > 0) & (sh_away > 0), "Y", "N")

        # Win Either Half
        ht_winner_home = out["HTHG"] > out["HTAG"]
        sh_winner_home = sh_home > sh_away
        ht_winner_away = out["HTHG"] < out["HTAG"]
        sh_winner_away = sh_home < sh_away

        out["y_HomeWinEitherHalf"] = np.where(ht_winner_home | sh_winner_home, "Y", "N")
        out["y_AwayWinEitherHalf"] = np.where(ht_winner_away | sh_winner_away, "Y", "N")

        # Win Both Halves
        out["y_HomeWinBothHalves"] = np.where(ht_winner_home & sh_winner_home, "Y", "N")
        out["y_AwayWinBothHalves"] = np.where(ht_winner_away & sh_winner_away, "Y", "N")

        # First Team to Score (estimated: whoever leads at HT or if 0-0, who wins)
        def first_scorer(row):
            if row['HTHG'] > 0 and row['HTAG'] == 0:
                return 'H'
            elif row['HTAG'] > 0 and row['HTHG'] == 0:
                return 'A'
            elif row['HTHG'] > 0 and row['HTAG'] > 0:
                return 'Unknown'  # Both scored in 1H
            elif row['FTHG'] > row['FTAG']:
                return 'H'
            elif row['FTAG'] > row['FTHG']:
                return 'A'
            return 'None'  # 0-0 draw

        out["y_FirstToScore"] = out.apply(first_scorer, axis=1)

    elif "HTR" in out.columns:
        # Legacy format
        out["y_HT"] = out["HTR"].astype(str)
        out["y_HTFT"] = out["HTR"].astype(str) + "-" + out["FTR"].astype(str)

    # ===========================================================================
    # TEAM GOALS MARKETS
    # ===========================================================================

    # Home Team Goals Over/Under
    for line in TEAM_GOAL_LINES:
        line_str = str(line).replace('.', '_')
        out[f"y_HomeTG_{line_str}"] = np.where(out["FTHG"] > line, "O", "U")

    # Away Team Goals Over/Under
    for line in TEAM_GOAL_LINES:
        line_str = str(line).replace('.', '_')
        out[f"y_AwayTG_{line_str}"] = np.where(out["FTAG"] > line, "O", "U")

    # Exact Home Goals
    out["y_HomeExact_0"] = np.where(out["FTHG"] == 0, "Y", "N")
    out["y_HomeExact_1"] = np.where(out["FTHG"] == 1, "Y", "N")
    out["y_HomeExact_2"] = np.where(out["FTHG"] == 2, "Y", "N")
    out["y_HomeExact_3+"] = np.where(out["FTHG"] >= 3, "Y", "N")

    # Exact Away Goals
    out["y_AwayExact_0"] = np.where(out["FTAG"] == 0, "Y", "N")
    out["y_AwayExact_1"] = np.where(out["FTAG"] == 1, "Y", "N")
    out["y_AwayExact_2"] = np.where(out["FTAG"] == 2, "Y", "N")
    out["y_AwayExact_3+"] = np.where(out["FTAG"] >= 3, "Y", "N")

    # ===========================================================================
    # ASIAN HANDICAP MARKETS (Extended)
    # ===========================================================================

    for line in AH_LINES:
        if line < 0:
            line_str = f"-{abs(line)}".replace('.', '_')
        elif line > 0:
            line_str = f"+{line}".replace('.', '_')
        else:
            line_str = "0_0"

        adjusted = goal_diff - line

        # H = Home covers, A = Away covers, P = Push
        def ah_result(adj):
            if adj > 0:
                return "H"
            elif adj < 0:
                return "A"
            else:
                return "P"

        out[f"y_AH_{line_str}"] = adjusted.apply(ah_result)

    # ===========================================================================
    # EUROPEAN HANDICAP (3-way)
    # ===========================================================================

    for line in [-1, -2, 1, 2]:
        line_str = f"{line:+d}".replace('+', 'p').replace('-', 'm')
        adj_diff = goal_diff + line  # Home gets the handicap

        out[f"y_EH_{line_str}_H"] = np.where(adj_diff > 0, "Y", "N")
        out[f"y_EH_{line_str}_D"] = np.where(adj_diff == 0, "Y", "N")
        out[f"y_EH_{line_str}_A"] = np.where(adj_diff < 0, "Y", "N")

    # ===========================================================================
    # DOUBLE CHANCE MARKETS
    # ===========================================================================

    out["y_DC_1X"] = np.where(out["FTR"].isin(["H", "D"]), "Y", "N")
    out["y_DC_X2"] = np.where(out["FTR"].isin(["D", "A"]), "Y", "N")
    out["y_DC_12"] = np.where(out["FTR"].isin(["H", "A"]), "Y", "N")

    # ===========================================================================
    # MARGIN OF VICTORY
    # ===========================================================================

    # Win to nil
    out["y_HomeWTN"] = np.where((out["FTR"] == "H") & (out["FTAG"] == 0), "Y", "N")
    out["y_AwayWTN"] = np.where((out["FTR"] == "A") & (out["FTHG"] == 0), "Y", "N")

    # Win by exactly 1, 2, 3+
    out["y_HomeWinBy1"] = np.where((out["FTR"] == "H") & (goal_diff == 1), "Y", "N")
    out["y_HomeWinBy2"] = np.where((out["FTR"] == "H") & (goal_diff == 2), "Y", "N")
    out["y_HomeWinBy3+"] = np.where((out["FTR"] == "H") & (goal_diff >= 3), "Y", "N")

    out["y_AwayWinBy1"] = np.where((out["FTR"] == "A") & (goal_diff == -1), "Y", "N")
    out["y_AwayWinBy2"] = np.where((out["FTR"] == "A") & (goal_diff == -2), "Y", "N")
    out["y_AwayWinBy3+"] = np.where((out["FTR"] == "A") & (goal_diff <= -3), "Y", "N")

    # Win by 2+ (legacy)
    out["y_HomeWin2+"] = np.where((out["FTR"] == "H") & (goal_diff >= 2), "Y", "N")
    out["y_AwayWin2+"] = np.where((out["FTR"] == "A") & (goal_diff <= -2), "Y", "N")

    # ===========================================================================
    # CLEAN SHEETS
    # ===========================================================================

    out["y_HomeCS"] = np.where(out["FTAG"] == 0, "Y", "N")
    out["y_AwayCS"] = np.where(out["FTHG"] == 0, "Y", "N")

    # No Goal (0-0)
    out["y_NoGoal"] = np.where(total == 0, "Y", "N")

    # ===========================================================================
    # ODD/EVEN GOALS
    # ===========================================================================

    out["y_TotalOddEven"] = np.where(total % 2 == 0, "Even", "Odd")
    out["y_HomeOddEven"] = np.where(out["FTHG"] % 2 == 0, "Even", "Odd")
    out["y_AwayOddEven"] = np.where(out["FTAG"] % 2 == 0, "Even", "Odd")

    # ===========================================================================
    # MULTI-GOAL MARKET
    # ===========================================================================

    out["y_Match2+Goals"] = np.where(total >= 2, "Y", "N")
    out["y_Match3+Goals"] = np.where(total >= 3, "Y", "N")
    out["y_Match4+Goals"] = np.where(total >= 4, "Y", "N")
    out["y_Match5+Goals"] = np.where(total >= 5, "Y", "N")

    # ===========================================================================
    # RESULT AND BTTS COMBOS
    # ===========================================================================

    out["y_HomeWin_BTTS_Y"] = np.where((out["FTR"] == "H") & btts, "Y", "N")
    out["y_HomeWin_BTTS_N"] = np.where((out["FTR"] == "H") & ~btts, "Y", "N")
    out["y_AwayWin_BTTS_Y"] = np.where((out["FTR"] == "A") & btts, "Y", "N")
    out["y_AwayWin_BTTS_N"] = np.where((out["FTR"] == "A") & ~btts, "Y", "N")
    out["y_Draw_BTTS_Y"] = np.where((out["FTR"] == "D") & btts, "Y", "N")
    out["y_Draw_BTTS_N"] = np.where((out["FTR"] == "D") & ~btts, "Y", "N")

    # ===========================================================================
    # RESULT AND OVER/UNDER COMBOS
    # ===========================================================================

    over25 = total > 2.5
    under25 = total <= 2.5

    out["y_HomeWin_O25"] = np.where((out["FTR"] == "H") & over25, "Y", "N")
    out["y_HomeWin_U25"] = np.where((out["FTR"] == "H") & under25, "Y", "N")
    out["y_AwayWin_O25"] = np.where((out["FTR"] == "A") & over25, "Y", "N")
    out["y_AwayWin_U25"] = np.where((out["FTR"] == "A") & under25, "Y", "N")
    out["y_Draw_O25"] = np.where((out["FTR"] == "D") & over25, "Y", "N")
    out["y_Draw_U25"] = np.where((out["FTR"] == "D") & under25, "Y", "N")

    # ===========================================================================
    # DOUBLE CHANCE + O/U COMBOS
    # ===========================================================================

    out["y_DC1X_O25"] = np.where(out["FTR"].isin(["H", "D"]) & over25, "Y", "N")
    out["y_DC1X_U25"] = np.where(out["FTR"].isin(["H", "D"]) & under25, "Y", "N")
    out["y_DCX2_O25"] = np.where(out["FTR"].isin(["D", "A"]) & over25, "Y", "N")
    out["y_DCX2_U25"] = np.where(out["FTR"].isin(["D", "A"]) & under25, "Y", "N")
    out["y_DC12_O25"] = np.where(out["FTR"].isin(["H", "A"]) & over25, "Y", "N")
    out["y_DC12_U25"] = np.where(out["FTR"].isin(["H", "A"]) & under25, "Y", "N")

    # ===========================================================================
    # DOUBLE CHANCE + BTTS COMBOS
    # ===========================================================================

    out["y_DC1X_BTTS_Y"] = np.where(out["FTR"].isin(["H", "D"]) & btts, "Y", "N")
    out["y_DC1X_BTTS_N"] = np.where(out["FTR"].isin(["H", "D"]) & ~btts, "Y", "N")
    out["y_DCX2_BTTS_Y"] = np.where(out["FTR"].isin(["D", "A"]) & btts, "Y", "N")
    out["y_DCX2_BTTS_N"] = np.where(out["FTR"].isin(["D", "A"]) & ~btts, "Y", "N")

    # ===========================================================================
    # CARDS MARKETS (requires Home_CardsY, Away_CardsY, Home_CardsR, Away_CardsR)
    # ===========================================================================
    home_cy = "Home_CardsY" if "Home_CardsY" in out.columns else "HY" if "HY" in out.columns else None
    away_cy = "Away_CardsY" if "Away_CardsY" in out.columns else "AY" if "AY" in out.columns else None
    home_cr = "Home_CardsR" if "Home_CardsR" in out.columns else "HR" if "HR" in out.columns else None
    away_cr = "Away_CardsR" if "Away_CardsR" in out.columns else "AR" if "AR" in out.columns else None

    if home_cy and away_cy:
        # Matches without card data must get NaN targets, not "no cards".
        # fillna(0) here mislabelled every stats-less fixture as Under/No,
        # poisoning the classifiers for ~half the dataset.
        hcy = pd.to_numeric(out[home_cy], errors='coerce')
        acy = pd.to_numeric(out[away_cy], errors='coerce')
        cards_valid = hcy.notna() & acy.notna()
        total_cy = hcy + acy

        # Total yellow cards O/U lines
        for line in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
            tag = str(line).replace('.', '_')
            out[f"y_TotalYC_O{tag}"] = np.where(
                cards_valid, np.where(total_cy > line, "Y", "N"), None)
        # Booking points O/U (common on UK exchanges: 10=yellow, 25=red)
        bp = hcy * 10
        bp_away = acy * 10
        if home_cr and away_cr:
            bp = bp + pd.to_numeric(out[home_cr], errors='coerce').fillna(0) * 25
            bp_away = bp_away + pd.to_numeric(out[away_cr], errors='coerce').fillna(0) * 25
        total_bp = bp + bp_away
        for line in [20.5, 30.5, 40.5, 50.5]:
            tag = str(line).replace('.', '_')
            out[f"y_BookingPts_O{tag}"] = np.where(
                cards_valid, np.where(total_bp > line, "Y", "N"), None)
        # Home/Away team to receive a card
        out["y_HomeTeam_Card"] = np.where(
            cards_valid, np.where(hcy > 0, "Y", "N"), None)
        out["y_AwayTeam_Card"] = np.where(
            cards_valid, np.where(acy > 0, "Y", "N"), None)

    # ===========================================================================
    # CORNERS MARKETS (requires Home_Corners, Away_Corners)
    # ===========================================================================
    home_cor = "Home_Corners" if "Home_Corners" in out.columns else None
    away_cor = "Away_Corners" if "Away_Corners" in out.columns else None

    if home_cor and away_cor:
        # Same NaN-target rule as cards: no data -> no label, not "Under".
        hcor = pd.to_numeric(out[home_cor], errors='coerce')
        acor = pd.to_numeric(out[away_cor], errors='coerce')
        corners_valid = hcor.notna() & acor.notna()
        total_corners = hcor + acor
        for line in [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]:
            tag = str(line).replace('.', '_')
            out[f"y_TotalCorners_O{tag}"] = np.where(
                corners_valid, np.where(total_corners > line, "Y", "N"), None)
        for line in [3.5, 4.5, 5.5, 6.5]:
            tag = str(line).replace('.', '_')
            out[f"y_HomeCorners_O{tag}"] = np.where(
                corners_valid, np.where(hcor > line, "Y", "N"), None)
            out[f"y_AwayCorners_O{tag}"] = np.where(
                corners_valid, np.where(acor > line, "Y", "N"), None)
        # Corner handicap (home - away diff)
        out["y_HomeCorners_Win"] = np.where(
            corners_valid, np.where(hcor > acor, "Y", "N"), None)

    return out

# -----------------------------
# Referee features
# -----------------------------

def _add_referee_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add referee historical stats features.

    Uses cumulative expanding stats for each referee from past games only
    (shift-by-1 before expanding mean), so no data leakage.

    Requires: Date, FTHG, FTAG, FTR, and either 'referee' or 'Referee' column.
    """
    ref_col = None
    if 'referee' in df.columns:
        ref_col = 'referee'
    elif 'Referee' in df.columns:
        ref_col = 'Referee'
    else:
        print("   [REF] No referee column — skipping")
        return df

    needed = {ref_col, 'Date', 'FTHG', 'FTAG', 'FTR'}
    if not needed.issubset(df.columns):
        print("   [REF] Skipping — required columns missing")
        return df

    # Work on rows with known referee AND known result to build stats
    df = df.sort_values('Date').reset_index(drop=True)

    has_ref = df[ref_col].notna() & (df[ref_col] != '')
    has_result = df['FTHG'].notna() & df['FTAG'].notna()
    valid = has_ref & has_result

    # Accept both raw card col names (HY/AY from CSV) and processed names (Home_CardsY/Away_CardsY)
    card_col_candidates = ['HY', 'AY', 'Home_CardsY', 'Away_CardsY']
    card_cols = [c for c in card_col_candidates if c in df.columns]
    # Pick one home and one away card column each
    home_card = next((c for c in ['HY', 'Home_CardsY'] if c in df.columns), None)
    away_card = next((c for c in ['AY', 'Away_CardsY'] if c in df.columns), None)
    card_cols = [c for c in [home_card, away_card] if c is not None]
    # Pick fouls columns if available (Home_Fouls and Away_Fouls are per-match raw values)
    home_fouls = 'Home_Fouls' if 'Home_Fouls' in df.columns else None
    away_fouls = 'Away_Fouls' if 'Away_Fouls' in df.columns else None
    foul_cols = [c for c in [home_fouls, away_fouls] if c is not None]

    extra_cols = card_cols + foul_cols
    base_cols = [ref_col, 'FTHG', 'FTAG', 'FTR'] + extra_cols
    tmp = df.loc[valid, base_cols].copy()
    tmp['_goals'] = tmp['FTHG'] + tmp['FTAG']
    tmp['_hw'] = (tmp['FTR'] == 'H').astype(float)
    tmp['_btts'] = ((tmp['FTHG'] > 0) & (tmp['FTAG'] > 0)).astype(float)
    if card_cols:
        tmp['_cards'] = tmp[card_cols].sum(axis=1)
    if foul_cols:
        tmp['_fouls'] = tmp[foul_cols].sum(axis=1)

    def _expanding_lag_mean(series):
        return series.shift(1).expanding().mean()

    stat_map = [
        ('Ref_AvgGoals', '_goals'),
        ('Ref_HomeWinRate', '_hw'),
        ('Ref_BTTSRate', '_btts'),
    ]
    if card_cols:
        stat_map.append(('Ref_AvgCards', '_cards'))
    if foul_cols:
        stat_map.append(('Ref_AvgFouls', '_fouls'))

    for out_col, src_col in stat_map:
        df.loc[valid, out_col] = (
            tmp.groupby(ref_col)[src_col]
            .transform(_expanding_lag_mean)
            .values
        )

    # Ref match count — how much history is available
    df.loc[valid, 'Ref_Count'] = (
        tmp.groupby(ref_col)['_goals']
        .transform(lambda s: s.shift(1).expanding().count())
        .values
    )

    return df


# -----------------------------
# League table position features
# -----------------------------

def _add_table_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add live league table position at the time of each match.

    For each match computes cumulative season points/GD for home and away teams
    using only results BEFORE that match date (no leakage). Then derives:
      Home_SeasonPts, Away_SeasonPts
      Home_SeasonGF, Home_SeasonGA, Home_SeasonGD
      Away_SeasonGF, Away_SeasonGA, Away_SeasonGD
      Home_TablePos, Away_TablePos   (1 = top; only within same league+season)
      TablePosDiff                   (Home - Away position; negative = home ranked higher)
      Home_PPG_Season, Away_PPG_Season
      IsTopSix_Home, IsTopSix_Away, IsBottom3_Home, IsBottom3_Away
    """
    needed = {'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'League', 'Season'}
    if not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        print(f"   [TABLE] Skipping — missing cols: {missing}")
        return df

    df = df.sort_values(['League', 'Season', 'Date']).reset_index(drop=True)

    # Build cumulative season points per team per league+season (points BEFORE each game)
    # One row per appearance (home + away)
    rows_h = df[['League', 'Season', 'Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR']].rename(
        columns={'HomeTeam': 'Team', 'FTHG': 'GF', 'FTAG': 'GA'})
    rows_h['Pts'] = rows_h['FTR'].map({'H': 3, 'D': 1, 'A': 0}).fillna(0)

    rows_a = df[['League', 'Season', 'Date', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].rename(
        columns={'AwayTeam': 'Team', 'FTAG': 'GF', 'FTHG': 'GA'})
    rows_a['Pts'] = rows_a['FTR'].map({'A': 3, 'D': 1, 'H': 0}).fillna(0)

    apps = pd.concat([rows_h, rows_a], ignore_index=True).sort_values(['League', 'Season', 'Date'])

    # Cumulative season stats up to (but not including) each game
    apps['CumPts'] = apps.groupby(['League', 'Season', 'Team'])['Pts'].transform(
        lambda s: s.shift(1).expanding().sum().fillna(0))
    apps['CumGF'] = apps.groupby(['League', 'Season', 'Team'])['GF'].transform(
        lambda s: s.shift(1).expanding().sum().fillna(0))
    apps['CumGA'] = apps.groupby(['League', 'Season', 'Team'])['GA'].transform(
        lambda s: s.shift(1).expanding().sum().fillna(0))
    apps['CumGames'] = apps.groupby(['League', 'Season', 'Team'])['Pts'].transform(
        lambda s: s.shift(1).expanding().count().fillna(0))

    # For each match date: compute the league table snapshot by taking each team's latest row
    # up to (not including) the match date, then rank by Pts desc, GD desc, GF desc.
    # We join pre-match cumulative stats onto the main df.
    latest = apps.copy()
    latest['CumGD'] = latest['CumGF'] - latest['CumGA']

    # Index-based lookup — avoids copying the already-wide df (merge copies, this doesn't)
    cum_idx = latest.set_index(['League', 'Season', 'Date', 'Team'])[
        ['CumPts', 'CumGF', 'CumGA', 'CumGD', 'CumGames']]
    cum_idx = cum_idx[~cum_idx.index.duplicated(keep='first')]

    h_keys = pd.MultiIndex.from_arrays([df['League'], df['Season'], df['Date'], df['HomeTeam']])
    a_keys = pd.MultiIndex.from_arrays([df['League'], df['Season'], df['Date'], df['AwayTeam']])

    for src, h_dst, a_dst in [
        ('CumPts',   'Home_SeasonPts',   'Away_SeasonPts'),
        ('CumGF',    'Home_SeasonGF',    'Away_SeasonGF'),
        ('CumGA',    'Home_SeasonGA',    'Away_SeasonGA'),
        ('CumGD',    'Home_SeasonGD',    'Away_SeasonGD'),
        ('CumGames', 'Home_SeasonGames', 'Away_SeasonGames'),
    ]:
        df[h_dst] = cum_idx[src].reindex(h_keys).values
        df[a_dst] = cum_idx[src].reindex(a_keys).values

    # Points per game this season
    df['Home_PPG_Season'] = np.where(df['Home_SeasonGames'] > 0,
                                      df['Home_SeasonPts'] / df['Home_SeasonGames'], 0.0)
    df['Away_PPG_Season'] = np.where(df['Away_SeasonGames'] > 0,
                                      df['Away_SeasonPts'] / df['Away_SeasonGames'], 0.0)

    # Table position: rank the FULL league table as of strictly before each
    # match date. The previous implementation grouped by League+Season+Date,
    # which ranked only the teams that happened to PLAY on that exact date —
    # e.g. "2nd of the 4 teams playing on Tuesday" — so Home_TablePos,
    # IsTopSix, IsBottom3 etc. were noise rather than real table positions.
    #
    # Approach per league-season: pivot each team's post-match cumulative
    # stats by date, forward-fill (teams keep their standing on days they
    # don't play), shift one match-date back (state BEFORE that date), then
    # rank across all teams by Pts desc, GD desc, GF desc.
    latest['PostPts'] = latest['CumPts'] + latest['Pts']
    latest['PostGF']  = latest['CumGF'] + latest['GF']
    latest['PostGA']  = latest['CumGA'] + latest['GA']

    snap_frames = []
    for (lg, season), grp in latest.groupby(['League', 'Season'], sort=False):
        pts = grp.pivot_table(index='Date', columns='Team', values='PostPts', aggfunc='last')
        gf  = grp.pivot_table(index='Date', columns='Team', values='PostGF',  aggfunc='last')
        ga  = grp.pivot_table(index='Date', columns='Team', values='PostGA',  aggfunc='last')
        if pts.empty:
            continue
        pts = pts.sort_index().ffill().shift(1).fillna(0.0)
        gf  = gf.sort_index().ffill().shift(1).fillna(0.0)
        ga  = ga.sort_index().ffill().shift(1).fillna(0.0)
        gd  = gf - ga
        # Composite ranking key (Pts >> GD >> GF); offsets keep GD positive
        score = pts * 1e8 + (gd + 1000.0) * 1e4 + gf
        pos = score.rank(axis=1, method='first', ascending=False)
        n_teams = pts.shape[1]
        long_pos = pos.stack().rename('TablePos').reset_index()
        long_pos.columns = ['Date', 'Team', 'TablePos']
        long_pos['League'] = lg
        long_pos['Season'] = season
        long_pos['NumTeams'] = float(n_teams)
        snap_frames.append(long_pos)

    if not snap_frames:
        print("   [TABLE] No league-season groups to rank — skipping table positions")
        return df

    snapshot = pd.concat(snap_frames, ignore_index=True)
    snapshot['TablePosPct'] = snapshot['TablePos'] / snapshot['NumTeams']

    snap_idx = snapshot.set_index(['League', 'Season', 'Date', 'Team'])[
        ['TablePos', 'TablePosPct', 'NumTeams']]
    snap_idx = snap_idx[~snap_idx.index.duplicated(keep='first')]

    df['Home_TablePos']   = snap_idx['TablePos'].reindex(h_keys).values
    df['Home_TablePosPct']= snap_idx['TablePosPct'].reindex(h_keys).values
    df['Home_NumTeams']   = snap_idx['NumTeams'].reindex(h_keys).values
    df['Away_TablePos']   = snap_idx['TablePos'].reindex(a_keys).values
    df['Away_TablePosPct']= snap_idx['TablePosPct'].reindex(a_keys).values

    df['TablePosDiff'] = df['Home_TablePos'] - df['Away_TablePos']  # negative = home higher
    df['SeasonPtsDiff'] = df['Home_SeasonPts'] - df['Away_SeasonPts']

    # Binary pressure flags based on table % position (bottom 15% = relegation zone, top 15% = title)
    df['IsBottom3_Home'] = (df['Home_TablePosPct'] >= 0.85).astype(float)
    df['IsBottom3_Away'] = (df['Away_TablePosPct'] >= 0.85).astype(float)
    df['IsTopSix_Home']  = (df['Home_TablePosPct'] <= 0.30).astype(float)
    df['IsTopSix_Away']  = (df['Away_TablePosPct'] <= 0.30).astype(float)
    df['BothTopSix']     = ((df['IsTopSix_Home'] == 1) & (df['IsTopSix_Away'] == 1)).astype(float)
    df['RelegationClash'] = ((df['IsBottom3_Home'] == 1) | (df['IsBottom3_Away'] == 1)).astype(float)

    new_cols = ['Home_SeasonPts', 'Away_SeasonPts', 'Home_SeasonGF', 'Home_SeasonGA',
                'Home_SeasonGD', 'Away_SeasonGF', 'Away_SeasonGA', 'Away_SeasonGD',
                'Home_PPG_Season', 'Away_PPG_Season', 'Home_TablePos', 'Away_TablePos',
                'TablePosDiff', 'SeasonPtsDiff', 'IsTopSix_Home', 'IsTopSix_Away',
                'IsBottom3_Home', 'IsBottom3_Away', 'BothTopSix', 'RelegationClash']
    added = [c for c in new_cols if c in df.columns]
    print(f"   Added table position features: {added}")
    return df


# -----------------------------
# H2H features
# -----------------------------

def _add_h2h_features(df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
    """Add head-to-head history features for each fixture.

    For each match computes stats from the last n_matches between the same two
    teams (either direction) BEFORE the match date, so there is no data leakage.

    Uses a vectorised merge approach rather than row-by-row iteration.
    """
    needed = {'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'}
    if not needed.issubset(df.columns):
        print("   [H2H] Skipping — required columns missing")
        return df
    # Derive FTR if not already present (happens when called before _add_all_targets)
    if 'FTR' not in df.columns:
        df = df.copy()
        df['FTR'] = np.where(df['FTHG'] > df['FTAG'], 'H',
                             np.where(df['FTHG'] < df['FTAG'], 'A', 'D'))

    df = df.sort_values('Date').reset_index(drop=True)

    # Build a long-form table with a canonical (team_a, team_b) pair key (sorted)
    # to look up all past meetings regardless of home/away direction.
    ref = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].copy()
    ref['pair'] = [tuple(sorted([h, a])) for h, a in zip(ref['HomeTeam'], ref['AwayTeam'])]
    ref['_idx'] = ref.index

    # For each fixture, find all past meetings for that pair
    out_cols = {
        'H2H_HomeWinRate': [], 'H2H_AwayWinRate': [], 'H2H_DrawRate': [],
        'H2H_AvgGoals': [], 'H2H_BTTSRate': [],
        'H2H_HomeGoalsAvg': [], 'H2H_AwayGoalsAvg': [],
        'H2H_Count': [],
    }

    # Group past matches by pair for fast lookup
    pair_groups = ref.groupby('pair')

    for _, row in df[['Date', 'HomeTeam', 'AwayTeam']].iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        date = row['Date']
        pair = tuple(sorted([home, away]))

        if pair not in pair_groups.groups:
            for k in out_cols:
                out_cols[k].append(np.nan if k != 'H2H_Count' else 0)
            continue

        past = ref.loc[pair_groups.groups[pair]]
        past = past[past['Date'] < date].tail(n_matches)

        if len(past) == 0:
            for k in out_cols:
                out_cols[k].append(np.nan if k != 'H2H_Count' else 0)
            continue

        # Wins from current home-team perspective
        home_as_home = past[(past['HomeTeam'] == home)]
        home_as_away = past[(past['AwayTeam'] == home)]

        wins = (home_as_home['FTR'] == 'H').sum() + (home_as_away['FTR'] == 'A').sum()
        losses = (home_as_home['FTR'] == 'A').sum() + (home_as_away['FTR'] == 'H').sum()
        total = len(past)

        out_cols['H2H_HomeWinRate'].append(wins / total)
        out_cols['H2H_AwayWinRate'].append(losses / total)
        out_cols['H2H_DrawRate'].append((total - wins - losses) / total)

        goals = pd.concat([
            home_as_home['FTHG'] + home_as_home['FTAG'],
            home_as_away['FTHG'] + home_as_away['FTAG'],
        ])
        out_cols['H2H_AvgGoals'].append(goals.mean())

        btts = pd.concat([
            (home_as_home['FTHG'] > 0) & (home_as_home['FTAG'] > 0),
            (home_as_away['FTHG'] > 0) & (home_as_away['FTAG'] > 0),
        ])
        out_cols['H2H_BTTSRate'].append(btts.mean())

        hgoals = pd.concat([home_as_home['FTHG'], home_as_away['FTAG']])
        agoals = pd.concat([home_as_home['FTAG'], home_as_away['FTHG']])
        out_cols['H2H_HomeGoalsAvg'].append(hgoals.mean())
        out_cols['H2H_AwayGoalsAvg'].append(agoals.mean())
        out_cols['H2H_Count'].append(total)

    for col, vals in out_cols.items():
        df[col] = vals

    return df


# -----------------------------
# -----------------------------
# Previous-season standings features
# -----------------------------

def _add_prev_season_standings(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-season final standings as features (no lookahead bias).

    For each match in season S, looks up the home/away team's final rank,
    points, and GD from season S-1 in the same league.
    Falls back to API-Football standings table for teams with no prior history.
    """
    needed = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "League", "Season"}
    if not needed.issubset(df.columns):
        print("   [PREV_STANDINGS] Skipping — missing required columns")
        return df

    # --- Build end-of-season totals from historical match results ---
    rows_h = df[["League", "Season", "HomeTeam", "FTHG", "FTAG", "FTR"]].copy()
    rows_h = rows_h.rename(columns={"HomeTeam": "Team", "FTHG": "GF", "FTAG": "GA"})
    rows_h["Pts"] = rows_h["FTR"].map({"H": 3, "D": 1, "A": 0}).fillna(0)

    rows_a = df[["League", "Season", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    rows_a = rows_a.rename(columns={"AwayTeam": "Team", "FTAG": "GF", "FTHG": "GA"})
    rows_a["Pts"] = rows_a["FTR"].map({"A": 3, "D": 1, "H": 0}).fillna(0)

    apps = pd.concat([rows_h, rows_a], ignore_index=True)
    season_totals = apps.groupby(["League", "Season", "Team"], as_index=False).agg(
        SeasonPts=("Pts", "sum"),
        SeasonGF=("GF", "sum"),
        SeasonGA=("GA", "sum"),
    )
    season_totals["SeasonGD"] = season_totals["SeasonGF"] - season_totals["SeasonGA"]
    season_totals["SeasonRank"] = (
        season_totals.groupby(["League", "Season"])["SeasonPts"]
        .rank(ascending=False, method="min")
        .astype(float)
    )
    n_teams = season_totals.groupby(["League", "Season"])["Team"].transform("count")
    season_totals["SeasonRankPct"] = season_totals["SeasonRank"] / n_teams

    # --- Try to supplement with API standings for seasons not in historical data ---
    try:
        import sqlite3
        from config import API_FOOTBALL_DB
        conn = sqlite3.connect(str(API_FOOTBALL_DB))
        api_st = pd.read_sql_query(
            "SELECT league_code, season, team_name, rank, points, goals_diff FROM standings",
            conn
        )
        conn.close()
        if not api_st.empty:
            api_st = api_st.rename(columns={
                "league_code": "League", "season": "Season",
                "team_name": "Team", "rank": "SeasonRank",
                "points": "SeasonPts", "goals_diff": "SeasonGD"
            })
            api_st["SeasonRankPct"] = np.nan
            api_st["SeasonGF"] = np.nan
            api_st["SeasonGA"] = np.nan
            # Only use API rows for (League, Season) combos missing from historical
            hist_keys = set(zip(season_totals["League"], season_totals["Season"].astype(str)))
            api_st["_key"] = list(zip(api_st["League"], api_st["Season"].astype(str)))
            api_extra = api_st[~api_st["_key"].isin(hist_keys)].drop(columns=["_key"])
            season_totals = pd.concat([season_totals, api_extra], ignore_index=True)
    except Exception:
        pass

    # --- Create lookup: (League, Season, Team) -> stats ---
    season_totals = season_totals.drop_duplicates(subset=["League", "Season", "Team"], keep="first")
    lut = season_totals.set_index(["League", "Season", "Team"])

    prev = df["Season"] - 1
    h_keys = list(zip(df["League"], prev, df["HomeTeam"]))
    a_keys = list(zip(df["League"], prev, df["AwayTeam"]))

    def _lookup(keys, col):
        idx = pd.MultiIndex.from_tuples(keys)
        return lut[col].reindex(idx).values

    df["Home_PrevSeasonRank"]    = _lookup(h_keys, "SeasonRank")
    df["Away_PrevSeasonRank"]    = _lookup(a_keys, "SeasonRank")
    df["Home_PrevSeasonPts"]     = _lookup(h_keys, "SeasonPts")
    df["Away_PrevSeasonPts"]     = _lookup(a_keys, "SeasonPts")
    df["Home_PrevSeasonGD"]      = _lookup(h_keys, "SeasonGD")
    df["Away_PrevSeasonGD"]      = _lookup(a_keys, "SeasonGD")
    df["Home_PrevSeasonRankPct"] = _lookup(h_keys, "SeasonRankPct")
    df["Away_PrevSeasonRankPct"] = _lookup(a_keys, "SeasonRankPct")
    df["PrevSeasonRankDiff"]     = df["Home_PrevSeasonRank"] - df["Away_PrevSeasonRank"]
    df["PrevSeasonPtsDiff"]      = df["Home_PrevSeasonPts"]  - df["Away_PrevSeasonPts"]
    df["PrevSeasonGDDiff"]       = df["Home_PrevSeasonGD"]   - df["Away_PrevSeasonGD"]

    filled = df["Home_PrevSeasonRank"].notna().sum()
    print(f"   Added prev-season standings: {filled:,}/{len(df):,} rows have data")
    return df


# Main build function
# -----------------------------

def build_features(force: bool = False) -> Path:
    """Build complete feature set for model training"""
    out_path = FEATURES_PARQUET
    
    if out_path.exists() and not force:
        log_header(f"Features exist at {out_path}. Use force=True to rebuild.")
        return out_path

    hist_path = HISTORICAL_PARQUET
    if not hist_path.exists():
        raise FileNotFoundError(f"Historical parquet not found at {hist_path}")

    log_header("BUILDING FEATURES")
    
    df = pd.read_parquet(hist_path)
    df = df.dropna(subset=["Date","HomeTeam","AwayTeam"]).copy()
    df = df.sort_values(["League","Date"]).reset_index(drop=True)

    # Normalise xG column names (historical_matches uses lowercase home_xG/away_xG)
    if 'home_xG' in df.columns:
        df = df.rename(columns={'home_xG': 'Home_xG', 'away_xG': 'Away_xG'})

    print(f"Loaded {len(df):,} matches")

    # 1. Elo ratings
    print("1. Calculating Elo ratings...")
    if USE_ELO:
        # Check if Elo columns already exist in historical_matches.parquet
        existing_elo = [c for c in df.columns if 'elo' in c.lower()]

        if existing_elo:
            # Use existing Elo columns and rename them
            print(f"   Using existing Elo columns: {existing_elo}")
            df = df.rename(columns={
                'EloHome_pre': 'Elo_Home',
                'EloAway_pre': 'Elo_Away',
                'EloDiff_pre': 'Elo_Diff'
            })
        else:
            # Calculate Elo from scratch
            df = _elo_by_league(df, EloConfig())

        print(f"   Added Elo features")

    # 1a. Glicko-2 ratings — Elo with explicit uncertainty (RD). RD is both a
    # feature (data sparsity signal) and the betting layer's unknown-team gate.
    print("1a. Calculating Glicko-2 ratings...")
    try:
        from ratings_glicko import add_glicko_features
        df = add_glicko_features(df)
    except Exception as e:
        print(f"   [GLICKO] Skipping: {e}")

    # 1b. League table position — run BEFORE rolling features to keep df narrow for merge
    print("1b. Adding league table position features...")
    df = _add_table_position_features(df)

    # 1c. Previous-season standings — needs FTHG/FTAG/FTR which survive only before pivot
    print("1c. Adding previous-season standings features...")
    df = _add_prev_season_standings(df)

    # 2. Rolling form/stats
    print("2. Calculating rolling form...")
    if USE_ROLLING_FORM:
        # _pivot_back keeps only its own base_cols, so EVERY column built in
        # steps 1/1a/1b/1c would otherwise be silently dropped. Preserving by
        # an explicit prefix list has now failed three times (the 31-feature
        # loss of 2026-07-13, then Glicko, then BothTopSix/RelegationClash/
        # Home_NumTeams) because each new feature has to remember to add
        # itself. Snapshot the column set instead and re-attach whatever the
        # pivot drops — new features are then preserved automatically.
        _keys = ['League', 'Date', 'HomeTeam', 'AwayTeam']
        _pre_pivot_cols = list(df.columns)
        preserve_data = df[_pre_pivot_cols].copy()

        side_feats = _build_side_features(df)
        df = _pivot_back(df, side_feats)

        # Anything present before the pivot but absent after it, minus the
        # join keys. 'Season' survives the pivot via base_cols, so excluding
        # it here avoids the Season_x/Season_y collision.
        preserve_cols = [c for c in _pre_pivot_cols
                         if c not in df.columns and c not in _keys]
        if preserve_cols:
            preserve_data = preserve_data[_keys + preserve_cols]
            print(f"   Re-attaching {len(preserve_cols)} pre-pivot column(s) dropped by the pivot")
        else:
            preserve_data = None

        # Re-merge preserved columns
        if preserve_cols:
            _n = len(df)
            df = df.merge(preserve_data.drop_duplicates(subset=_keys),
                          on=_keys, how='left')
            if len(df) != _n:
                raise RuntimeError(
                    f"Preserve-merge changed row count ({_n} -> {len(df)}) — "
                    "duplicate fixture keys in the feature frame")
        # _pivot_back already keeps Season; if a collision created Season_x/Season_y, clean it up
        if 'Season_x' in df.columns:
            df = df.rename(columns={'Season_x': 'Season'})
        if 'Season_y' in df.columns:
            df = df.drop(columns=['Season_y'])

        print(f"   Added rolling features")

    # 3. Contextual features
    print("3. Adding contextual features...")
    df = _add_contextual_features(df)
    print(f"   Added context features")

    # 4. Market features
    print("4. Adding market features...")
    if USE_MARKET_FEATURES:
        df = _add_market_features(df)
        print(f"   Added market features")

    # 5. Injury features
    print("5. Adding injury features...")
    try:
        injuries = get_injury_counts_from_db()
        if not injuries.empty:
            home_inj = injuries.rename(columns={'Team': 'HomeTeam', 'InjuryCount': 'Home_InjuryCount'})
            away_inj = injuries.rename(columns={'Team': 'AwayTeam', 'InjuryCount': 'Away_InjuryCount'})
            df = df.merge(home_inj[['League','Date','HomeTeam','Home_InjuryCount']],
                          on=['League','Date','HomeTeam'], how='left')
            df = df.merge(away_inj[['League','Date','AwayTeam','Away_InjuryCount']],
                          on=['League','Date','AwayTeam'], how='left')
            df['Home_InjuryCount'] = df['Home_InjuryCount'].fillna(0)
            df['Away_InjuryCount'] = df['Away_InjuryCount'].fillna(0)
            df['InjuryDiff'] = df['Home_InjuryCount'] - df['Away_InjuryCount']
            n_with = (df['Home_InjuryCount'] + df['Away_InjuryCount'] > 0).sum()
            print(f"   Injury data for {n_with:,} / {len(df):,} matches")
        else:
            print("   [INJ] No injury data in DB — skipping")
    except Exception as e:
        print(f"   [INJ] Skipping: {e}")

    # 6. Referee features
    print("6. Adding referee features...")
    df = _add_referee_features(df)
    h2h_ref_cols = [c for c in df.columns if c.startswith('Ref_')]
    print(f"   Added referee features: {h2h_ref_cols}")

    # 7. H2H features
    print("7. Adding H2H features...")
    df = _add_h2h_features(df, n_matches=5)
    h2h_cols = [c for c in df.columns if c.startswith('H2H_')]
    print(f"   Added H2H features: {h2h_cols}")

    # 8. Targets (note: _add_all_targets requires FTR in df)
    print("8. Creating target variables...")
    df = _add_all_targets(df)
    print(f"   Added targets")

    # 9. Handle NaN values
    print("9. Handling missing values...")
    # Drop metadata-only cols that serve no training purpose (venue_name is not predictive)
    df.drop(columns=[c for c in ['venue_name', 'fixture_id'] if c in df.columns], inplace=True)
    # Ensure referee is purely string — DB can store integer IDs for some entries and
    # pyarrow rejects mixed int/str object columns when writing parquet.
    if 'referee' in df.columns:
        df['referee'] = df['referee'].where(df['referee'].isna(), df['referee'].astype(str))
    # Note: 'referee' string col kept for predict.py lookup; models.py excludes it from features

    # Missingness indicators BEFORE any filling — median-fill erases the
    # difference between "average team" and "no data", which is exactly how
    # unknown teams ended up with confident garbage predictions.
    if 'Home_Corners' in df.columns:
        df['Has_MatchStats'] = df['Home_Corners'].notna().astype(float)
    if 'Home_xG' in df.columns:
        df['Has_xG'] = df['Home_xG'].notna().astype(float)
    if 'Home_PrevSeasonRank' in df.columns:
        df['Has_PrevSeason'] = df['Home_PrevSeasonRank'].notna().astype(float)
    # Count columns: NaN means "no history", which IS zero occurrences
    for cnt_col in ('H2H_Count', 'Ref_Count'):
        if cnt_col in df.columns:
            df[cnt_col] = df[cnt_col].fillna(0)

    # Median-fill ONLY genuine feature columns. Raw current-match stats
    # (Home_Corners, Away_CardsY, ...) are OUTCOMES used as count-model
    # targets — imputing them would fabricate results for ~half the fixtures
    # (matches without stats coverage). Result columns likewise stay NaN.
    from feature_rules import is_feature_col
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col.startswith('y_') or not is_feature_col(col):
            continue
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    
    log_header(f"FEATURES COMPLETE")
    print(f"Saved: {out_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    
    # Count feature types
    target_cols = [c for c in df.columns if c.startswith('y_')]
    feature_cols = [c for c in df.columns if not c.startswith('y_')]
    print(f"Features: {len(feature_cols)}, Targets: {len(target_cols)}")
    
    return out_path


def get_feature_columns() -> List[str]:
    """Return list of feature columns (not targets, metadata, or raw match stats).

    Exclusions come from feature_rules.py — the same single source of truth
    used by models.py:_feature_columns(), so the lists cannot drift apart.
    Raw CURRENT-match stats (both pre-pivot HS/HC/HY names and post-pivot
    Home_Corners/Away_CardsY/... names) leak the result; only rolling/EWM
    versions computed from prior matches are valid features.
    """
    from feature_rules import is_feature_col

    df = pd.read_parquet(FEATURES_PARQUET)
    return [col for col in df.columns if is_feature_col(col)]


if __name__ == "__main__":
    build_features(force=True)
