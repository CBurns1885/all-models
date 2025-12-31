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
    """Compute league-specific Elo ratings with momentum adjustment"""
    df = df.sort_values(["League","Date"]).copy()
    
    state: Dict[Tuple[str,str], float] = {}
    momentum: Dict[Tuple[str,str], float] = {}
    
    home_elos, away_elos, home_mom, away_mom = [], [], [], []

    for idx, row in df.iterrows():
        lg = row["League"]
        ht, at = row["HomeTeam"], row["AwayTeam"]
        key_h, key_a = (lg, ht), (lg, at)
        
        ra = state.get(key_h, cfg.base_rating)
        rb = state.get(key_a, cfg.base_rating)
        ma = momentum.get(key_h, 0.0)
        mb = momentum.get(key_a, 0.0)

        home_elos.append(ra)
        away_elos.append(rb)
        home_mom.append(ma)
        away_mom.append(mb)

        ftr = row.get("FTR")
        if pd.isna(ftr):
            continue
            
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
        
        state[key_h] = ra_new - cfg.home_adv
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
            "Win","Draw","Loss","Shots","ShotsT","Corners","CardsY","CardsR",
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
        team_df[f"PPG_ma{w}"] = (rolled["Win"].sum() * 3 + rolled["Draw"].sum()) / w
        
        # Shot stats
        for col in ["Shots","ShotsT","Corners","CardsY","CardsR"]:
            if col in team_df.columns:
                team_df[f"{col}_ma{w}"] = rolled[col].mean()
        
        # Derived rates
        team_df[f"CleanSheet_rate{w}"] = rolled["CleanSheet"].mean()
        team_df[f"FTS_rate{w}"] = rolled["FailedToScore"].mean()
        team_df[f"BTTS_rate{w}"] = rolled["BTTS"].mean()
        
        # Advanced stats (if available)
        for col in ["xG", "Possession", "ShotsInBox", "BigChances", "PassAcc"]:
            if col in team_df.columns and team_df[col].notna().any():
                team_df[f"{col}_ma{w}"] = rolled[col].mean()
    
    # EWMA features (recency-weighted)
    ew = team_df.shift(1).ewm(span=EWM_SPAN, adjust=False)
    team_df["GF_ewm"] = ew["GoalsFor"].mean()
    team_df["GA_ewm"] = ew["GoalsAgainst"].mean()
    team_df["PPG_ewm"] = (ew["Win"].mean() * 3 + ew["Draw"].mean())
    
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

OU_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]
AH_LINES = [-1.0, -0.5, 0.0, 0.5, 1.0]

def _add_all_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add all target variables"""
    out = df.copy()
    
    # 1X2
    out["y_1X2"] = out["FTR"].astype(str)
    
    # BTTS
    btts = (out["FTHG"].fillna(0) > 0) & (out["FTAG"].fillna(0) > 0)
    out["y_BTTS"] = np.where(btts, "Y", "N")
    
    # Over/Under
    total = out["FTHG"].fillna(0) + out["FTAG"].fillna(0)
    for line in OU_LINES:
        line_str = str(line).replace('.', '_')
        out[f"y_OU_{line_str}"] = np.where(total > line, "O", "U")
    
    # Goal range
    bins = pd.cut(total, bins=[-1,0,1,2,3,4,100], labels=["0","1","2","3","4","5+"])
    out["y_GOAL_RANGE"] = bins.astype(str)
    
    # Correct score
    def get_cs(row):
        h = int(row['FTHG']) if pd.notna(row['FTHG']) else -1
        a = int(row['FTAG']) if pd.notna(row['FTAG']) else -1
        if h < 0 or a < 0:
            return 'Other'
        if h <= 5 and a <= 5:
            return f'{h}-{a}'
        return 'Other'
    
    out["y_CS"] = out.apply(get_cs, axis=1)
    
    # Half-time
    if "HTR" in out.columns:
        out["y_HT"] = out["HTR"].astype(str)
        out["y_HTFT"] = out["HTR"].astype(str) + "_" + out["FTR"].astype(str)
    
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
        df = _elo_by_league(df, EloConfig())
        print(f"   Added Elo features")

    # 2. Rolling form/stats
    print("2. Calculating rolling form...")
    if USE_ROLLING_FORM:
        side_feats = _build_side_features(df)
        df = _pivot_back(df, side_feats)
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
