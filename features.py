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
    log_header
)

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
        
        # Advanced stats (from API-Football match_stats)
        # xG comes as home_xG (lowercase) from the adapter
        out = _ensure_cols(out, ["home_xG", "Home_Possession", "Home_Shots_Inside_Box",
                                  "Home_Pass_Accuracy", "Home_GKSaves",
                                  "Home_TotalPasses", "Home_Fouls", "Home_Offsides",
                                  "Home_BlockedShots", "Home_ShotsOffGoal",
                                  "Home_ShotsOutsideBox"])
        out["xG"] = out["home_xG"]
        out["Possession"] = out["Home_Possession"]
        out["ShotsInBox"] = out["Home_Shots_Inside_Box"]
        out["PassAcc"] = out["Home_Pass_Accuracy"]
        out["GKSaves"] = out["Home_GKSaves"]
        out["TotalPasses"] = out["Home_TotalPasses"]
        out["Fouls"] = out["Home_Fouls"]
        out["Offsides"] = out["Home_Offsides"]
        out["BlockedShots"] = out["Home_BlockedShots"]
        out["ShotsOffGoal"] = out["Home_ShotsOffGoal"]
        out["ShotsOutBox"] = out["Home_ShotsOutsideBox"]
        
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
        
        # xG comes as away_xG (lowercase) from the adapter
        out = _ensure_cols(out, ["away_xG", "Away_Possession", "Away_Shots_Inside_Box",
                                  "Away_Pass_Accuracy", "Away_GKSaves",
                                  "Away_TotalPasses", "Away_Fouls", "Away_Offsides",
                                  "Away_BlockedShots", "Away_ShotsOffGoal",
                                  "Away_ShotsOutsideBox"])
        out["xG"] = out["away_xG"]
        out["Possession"] = out["Away_Possession"]
        out["ShotsInBox"] = out["Away_Shots_Inside_Box"]
        out["PassAcc"] = out["Away_Pass_Accuracy"]
        out["GKSaves"] = out["Away_GKSaves"]
        out["TotalPasses"] = out["Away_TotalPasses"]
        out["Fouls"] = out["Away_Fouls"]
        out["Offsides"] = out["Away_Offsides"]
        out["BlockedShots"] = out["Away_BlockedShots"]
        out["ShotsOffGoal"] = out["Away_ShotsOffGoal"]
        out["ShotsOutBox"] = out["Away_ShotsOutsideBox"]
    
    out["Side"] = side
    out["CleanSheet"] = (out["GoalsAgainst"] == 0).astype(int)
    out["FailedToScore"] = (out["GoalsFor"] == 0).astype(int)
    out["BTTS"] = ((out["GoalsFor"] > 0) & (out["GoalsAgainst"] > 0)).astype(int)
    
    cols = ["League","Date","Team","Opp","Side","GoalsFor","GoalsAgainst",
            "Win","Draw","Loss","Shots","ShotsT","Corners","CardsY","CardsR",
            "CleanSheet","FailedToScore","BTTS",
            "xG","Possession","ShotsInBox","PassAcc",
            "GKSaves","TotalPasses","Fouls","Offsides","BlockedShots",
            "ShotsOffGoal","ShotsOutBox"]
    
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
        team_df[f"PPG_ma{w}"] = (rolled["Win"].sum() * 3 + rolled["Draw"].sum()) / rolled["Win"].count().clip(lower=1)
        
        # Shot stats
        for col in ["Shots","ShotsT","Corners","CardsY","CardsR"]:
            if col in team_df.columns:
                team_df[f"{col}_ma{w}"] = rolled[col].mean()
        
        # Derived rates
        team_df[f"CleanSheet_rate{w}"] = rolled["CleanSheet"].mean()
        team_df[f"FTS_rate{w}"] = rolled["FailedToScore"].mean()
        team_df[f"BTTS_rate{w}"] = rolled["BTTS"].mean()
        
        # Advanced stats (if available from API-Football match_stats)
        for col in ["xG", "Possession", "ShotsInBox", "PassAcc",
                     "GKSaves", "TotalPasses", "Fouls", "Offsides", "BlockedShots",
                     "ShotsOffGoal", "ShotsOutBox"]:
            if col in team_df.columns and team_df[col].notna().any():
                team_df[f"{col}_ma{w}"] = rolled[col].mean()
    
    # EWMA features (recency-weighted)
    ew = team_df.shift(1).ewm(span=EWM_SPAN, adjust=False)
    team_df["GF_ewm"] = ew["GoalsFor"].mean()
    team_df["GA_ewm"] = ew["GoalsAgainst"].mean()
    # Compute EWM of actual points sequence (not of Win/Draw separately)
    shifted = team_df.shift(1)
    points = shifted["Win"] * 3 + shifted["Draw"]
    team_df["PPG_ewm"] = points.ewm(span=EWM_SPAN, adjust=False).mean()
    
    if "xG" in team_df.columns and team_df["xG"].notna().any():
        team_df["xG_ewm"] = ew["xG"].mean()
    
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
    base_cols = key_cols + ["FTHG","FTAG","FTR"]
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
    
    # Day of week
    out['Date'] = pd.to_datetime(out['Date'])
    out['DayOfWeek'] = out['Date'].dt.dayofweek
    out['IsWeekend'] = out['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Season progress (approximate)
    out['Month'] = out['Date'].dt.month
    out['SeasonProgress'] = out['Month'].apply(lambda m: 
        (m - 8) / 10 if m >= 8 else (m + 4) / 10
    ).clip(0, 1)
    
    # Calculate rest days (keyed by original index to avoid misalignment)
    team_dates: Dict[str, pd.Timestamp] = {}
    rest_by_idx: Dict[int, Tuple[int, int]] = {}

    for idx, row in out.sort_values('Date').iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        match_date = row['Date']

        # Home team rest
        if ht in team_dates:
            h_rest = min((match_date - team_dates[ht]).days, 21)
        else:
            h_rest = 7  # Default

        # Away team rest
        if at in team_dates:
            a_rest = min((match_date - team_dates[at]).days, 21)
        else:
            a_rest = 7

        rest_by_idx[idx] = (h_rest, a_rest)

        # Update last match dates for both teams
        team_dates[ht] = match_date
        team_dates[at] = match_date

    out['Home_RestDays'] = out.index.map(lambda i: rest_by_idx.get(i, (7, 7))[0])
    out['Away_RestDays'] = out.index.map(lambda i: rest_by_idx.get(i, (7, 7))[1])
    out['RestDiff'] = out['Home_RestDays'] - out['Away_RestDays']

    # League standings features (if available from API)
    try:
        from api_football_adapter import get_standings_from_db
        standings = get_standings_from_db()
        if not standings.empty:
            # Join home team standings
            home_standings = standings.rename(columns={
                col: f'Home_{col}' for col in standings.columns
                if col not in ['League', 'Season', 'Team']
            })
            out = out.merge(
                home_standings, left_on=['League', 'HomeTeam'],
                right_on=['League', 'Team'], how='left'
            )
            if 'Team' in out.columns:
                out = out.drop(columns=['Team'])
            if 'Season_y' in out.columns:
                out = out.drop(columns=['Season_y'])
                out = out.rename(columns={'Season_x': 'Season'})

            # Join away team standings
            away_standings = standings.rename(columns={
                col: f'Away_{col}' for col in standings.columns
                if col not in ['League', 'Season', 'Team']
            })
            out = out.merge(
                away_standings, left_on=['League', 'AwayTeam'],
                right_on=['League', 'Team'], how='left'
            )
            if 'Team' in out.columns:
                out = out.drop(columns=['Team'])
            if 'Season_y' in out.columns:
                out = out.drop(columns=['Season_y'])
                out = out.rename(columns={'Season_x': 'Season'})

            # Derived features: position diff, points diff
            if 'Home_LeaguePosition' in out.columns and 'Away_LeaguePosition' in out.columns:
                out['PositionDiff'] = out['Away_LeaguePosition'] - out['Home_LeaguePosition']
                out['PointsDiff'] = out.get('Home_LeaguePoints', 0) - out.get('Away_LeaguePoints', 0)
                out['PPGDiff'] = out.get('Home_LeaguePPG', 0) - out.get('Away_LeaguePPG', 0)
                out['FormScoreDiff'] = out.get('Home_LeagueFormScore', 0) - out.get('Away_LeagueFormScore', 0)
                print(f"   Added league standings features for {standings['League'].nunique()} leagues")
    except (ImportError, Exception) as e:
        print(f"   [INFO] League standings not available: {e}")

    # Referee tendencies (cards/fouls per game from historical data)
    if 'referee' in out.columns and out['referee'].notna().any():
        try:
            ref_cols_needed = ['HY', 'AY', 'HR', 'AR', 'Home_Fouls', 'Away_Fouls']
            ref_cols_avail = [c for c in ref_cols_needed if c in out.columns]
            if ref_cols_avail:
                out_sorted = out.sort_values('Date')
                # Total cards per match by referee
                if 'HY' in out.columns and 'AY' in out.columns:
                    out_sorted['_TotalCards'] = (
                        out_sorted['HY'].fillna(0) + out_sorted['AY'].fillna(0) +
                        out_sorted.get('HR', pd.Series(0, index=out_sorted.index)).fillna(0) +
                        out_sorted.get('AR', pd.Series(0, index=out_sorted.index)).fillna(0)
                    )
                if 'Home_Fouls' in out.columns and 'Away_Fouls' in out.columns:
                    out_sorted['_TotalFouls'] = (
                        out_sorted['Home_Fouls'].fillna(0) + out_sorted['Away_Fouls'].fillna(0)
                    )
                # Total goals per match (for referee goal tendency)
                out_sorted['_TotalGoals'] = out_sorted['FTHG'].fillna(0) + out_sorted['FTAG'].fillna(0)

                # Expanding mean per referee (only uses past matches)
                for stat, out_col in [('_TotalCards', 'Ref_AvgCards'),
                                       ('_TotalFouls', 'Ref_AvgFouls'),
                                       ('_TotalGoals', 'Ref_AvgGoals')]:
                    if stat in out_sorted.columns:
                        ref_means = (
                            out_sorted.groupby('referee')[stat]
                            .transform(lambda x: x.shift(1).expanding().mean())
                        )
                        out.loc[out_sorted.index, out_col] = ref_means.values

                # Count how many matches this referee has done (experience proxy)
                ref_counts = (
                    out_sorted.groupby('referee').cumcount()
                )
                out.loc[out_sorted.index, 'Ref_MatchCount'] = ref_counts.values

                # Clean up temp columns
                for c in ['_TotalCards', '_TotalFouls', '_TotalGoals']:
                    if c in out.columns:
                        out = out.drop(columns=[c])

                ref_feature_count = sum(1 for c in ['Ref_AvgCards', 'Ref_AvgFouls', 'Ref_AvgGoals', 'Ref_MatchCount']
                                       if c in out.columns)
                print(f"   Added {ref_feature_count} referee tendency features")
        except Exception as e:
            print(f"   [INFO] Referee features skipped: {e}")

    # Injury count features (if available from API)
    try:
        from api_football_adapter import get_injury_counts_from_db
        injuries = get_injury_counts_from_db()
        if not injuries.empty:
            # Merge home team injuries
            home_inj = injuries.rename(columns={'InjuryCount': 'Home_InjuryCount', 'Team': '_Team'})
            out = out.merge(
                home_inj[['League', 'Date', '_Team', 'Home_InjuryCount']],
                left_on=['League', 'Date', 'HomeTeam'],
                right_on=['League', 'Date', '_Team'],
                how='left'
            )
            if '_Team' in out.columns:
                out = out.drop(columns=['_Team'])

            # Merge away team injuries
            away_inj = injuries.rename(columns={'InjuryCount': 'Away_InjuryCount', 'Team': '_Team'})
            out = out.merge(
                away_inj[['League', 'Date', '_Team', 'Away_InjuryCount']],
                left_on=['League', 'Date', 'AwayTeam'],
                right_on=['League', 'Date', '_Team'],
                how='left'
            )
            if '_Team' in out.columns:
                out = out.drop(columns=['_Team'])

            out['Home_InjuryCount'] = out['Home_InjuryCount'].fillna(0)
            out['Away_InjuryCount'] = out['Away_InjuryCount'].fillna(0)
            out['InjuryDiff'] = out['Home_InjuryCount'] - out['Away_InjuryCount']

            inj_matches = (out['Home_InjuryCount'] > 0).sum() + (out['Away_InjuryCount'] > 0).sum()
            print(f"   Added injury features ({inj_matches} team-match injury records)")
    except (ImportError, Exception) as e:
        print(f"   [INFO] Injury features not available: {e}")

    # League quality tier features (allows model to learn league-specific patterns)
    LEAGUE_TIERS = {
        # Elite (top 5 European leagues)
        'E0': 1, 'SP1': 1, 'I1': 1, 'D1': 1, 'F1': 1,
        # High quality
        'E1': 2, 'SP2': 2, 'I2': 2, 'D2': 2, 'F2': 2, 'N1': 2,
        'P1': 2, 'B1': 2, 'SC0': 2, 'T1': 2,
        # Medium quality
        'E2': 3, 'E3': 3, 'SC1': 3, 'A1': 3, 'G1': 3, 'SWZ': 3,
        'POL': 3, 'RUS': 3, 'EC': 3,
    }
    LEAGUE_AVG_GOALS = {
        'E0': 2.72, 'E1': 2.65, 'E2': 2.58, 'E3': 2.61, 'EC': 2.65,
        'SP1': 2.48, 'SP2': 2.35, 'I1': 2.68, 'I2': 2.45,
        'D1': 3.05, 'D2': 2.85, 'F1': 2.55, 'F2': 2.42,
        'N1': 2.95, 'B1': 2.78, 'P1': 2.52, 'SC0': 2.65, 'SC1': 2.58,
        'T1': 3.10, 'G1': 2.35, 'A1': 2.92, 'SWZ': 2.75,
        'POL': 2.62, 'RUS': 2.45,
    }
    LEAGUE_HOME_ADV = {
        'E0': 0.12, 'E1': 0.10, 'E2': 0.11, 'E3': 0.13, 'EC': 0.12,
        'SP1': 0.15, 'SP2': 0.14, 'I1': 0.11, 'I2': 0.12,
        'D1': 0.09, 'D2': 0.10, 'F1': 0.13, 'F2': 0.12,
        'N1': 0.08, 'B1': 0.10, 'P1': 0.16, 'SC0': 0.11, 'SC1': 0.13,
        'T1': 0.14, 'G1': 0.18, 'A1': 0.10, 'SWZ': 0.09,
        'POL': 0.12, 'RUS': 0.14,
    }
    LEAGUE_STYLE = {
        # 1 = attacking, 0 = balanced, -1 = defensive
        'E0': 0, 'E1': 0, 'E2': 0, 'E3': 0, 'EC': 0,
        'SP1': -1, 'SP2': -1, 'I1': 0, 'I2': -1,
        'D1': 1, 'D2': 1, 'F1': 0, 'F2': -1,
        'N1': 1, 'B1': 0, 'P1': -1, 'SC0': 0, 'SC1': 0,
        'T1': 1, 'G1': -1, 'A1': 1, 'SWZ': 0,
        'POL': 0, 'RUS': -1,
    }

    if 'League' in out.columns:
        out['League_Tier'] = out['League'].map(LEAGUE_TIERS).fillna(4).astype(int)
        out['League_AvgGoals'] = out['League'].map(LEAGUE_AVG_GOALS).fillna(2.60)
        out['League_HomeAdv'] = out['League'].map(LEAGUE_HOME_ADV).fillna(0.12)
        out['League_Style'] = out['League'].map(LEAGUE_STYLE).fillna(0).astype(int)
        tier_count = out['League_Tier'].nunique()
        print(f"   Added league quality tier features ({tier_count} tiers across {out['League'].nunique()} leagues)")

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

    # Ensure numeric columns
    out["FTHG"] = pd.to_numeric(out["FTHG"], errors='coerce').fillna(0).astype(int)
    out["FTAG"] = pd.to_numeric(out["FTAG"], errors='coerce').fillna(0).astype(int)
    total = out["FTHG"] + out["FTAG"]
    goal_diff = out["FTHG"] - out["FTAG"]

    # ===========================================================================
    # CORE MARKETS
    # ===========================================================================

    # 1X2 Match Result
    out["y_1X2"] = out["FTR"].where(out["FTR"].notna()).astype(object)

    # BTTS (Both Teams To Score)
    btts = (out["FTHG"] > 0) & (out["FTAG"] > 0)
    out["y_BTTS"] = np.where(btts, "Y", "N")

    # Over/Under Total Goals
    for line in OU_LINES:
        line_str = str(line).replace('.', '_')
        out[f"y_OU_{line_str}"] = np.where(total > line, "O", "U")

    # Goal Range (0, 1, 2, 3, 4, 5+)
    bins = pd.cut(total, bins=[-1,0,1,2,3,4,100], labels=["0","1","2","3","4","5+"])
    out["y_GOAL_RANGE"] = bins.where(bins.notna()).astype(object)

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
        # Preserve NaN for rows with missing FTR/HT data
        ht_str = out["y_HT"].where(out["y_HT"].notna())
        ftr_str = out["FTR"].where(out["FTR"].notna())
        out["y_HTFT"] = (ht_str + "-" + ftr_str).where(ht_str.notna() & ftr_str.notna())

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

    return out

# -----------------------------
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

    # 2. Rolling form/stats
    print("2. Calculating rolling form...")
    if USE_ROLLING_FORM:
        # Store Elo columns before reshape (they get lost in pivot)
        elo_cols = [c for c in df.columns if c.startswith('Elo_')]
        if elo_cols:
            elo_data = df[['League', 'Date', 'HomeTeam', 'AwayTeam'] + elo_cols].copy()

        side_feats = _build_side_features(df)
        df = _pivot_back(df, side_feats)

        # Re-merge Elo columns if they existed
        if elo_cols:
            df = df.merge(elo_data, on=['League', 'Date', 'HomeTeam', 'AwayTeam'], how='left')

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

    # 5. Targets
    print("5. Creating target variables...")
    df = _add_all_targets(df)
    print(f"   Added targets")

    # 6. Handle NaN values
    print("6. Handling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if not col.startswith('y_'):
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
    """Return list of feature columns (not targets or metadata)"""
    exclude = ['Date', 'League', 'HomeTeam', 'AwayTeam', 'Referee',
               'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
               'fixture_id', 'Home_ID', 'Away_ID', 'League_ID', 'Season']
    
    df = pd.read_parquet(FEATURES_PARQUET)
    
    return [col for col in df.columns 
            if col not in exclude 
            and not col.startswith('y_')]


if __name__ == "__main__":
    build_features(force=True)
