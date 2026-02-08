"""
API-Football Database Adapter for all_models
Reads from shared football_api.db (SQLite) instead of downloading CSVs
Provides richer data: xG, injuries, detailed statistics
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from config import API_FOOTBALL_DB, PROCESSED_DIR, log_header

def get_fixtures_from_db(
    seasons: List[int] = None,
    leagues: List[str] = None,
    status: str = 'FT'
) -> pd.DataFrame:
    """
    Read fixtures from shared API-Football database, joining match_stats
    for detailed per-team statistics (shots, possession, corners, etc.)

    Args:
        seasons: List of seasons (e.g., [2024, 2025])
        leagues: List of league codes (e.g., ['E0', 'E1'])
        status: Match status ('FT' for finished, 'NS' for not started)

    Returns:
        DataFrame with columns matching old CSV format + xG + detailed stats
    """
    if not API_FOOTBALL_DB.exists():
        raise FileNotFoundError(
            f"API-Football database not found at {API_FOOTBALL_DB}. "
            f"Run dc_laptop/runner.py first to download data."
        )

    conn = sqlite3.connect(API_FOOTBALL_DB)

    # Build query - join match_stats for home and away team statistics
    query = """
        SELECT
            f.league_code as League,
            f.season as Season,
            f.date as Date,
            f.home_team as HomeTeam,
            f.away_team as AwayTeam,
            f.home_goals as FTHG,
            f.away_goals as FTAG,
            f.ht_home_goals as HTHG,
            f.ht_away_goals as HTAG,
            CASE
                WHEN f.home_goals > f.away_goals THEN 'H'
                WHEN f.home_goals < f.away_goals THEN 'A'
                ELSE 'D'
            END as FTR,
            COALESCE(f.home_xG, ms_h.expected_goals) as home_xG,
            COALESCE(f.away_xG, ms_a.expected_goals) as away_xG,
            f.league_type,
            f.referee,
            f.venue_name,
            -- Home team stats from match_stats
            ms_h.shots_on_goal as Home_ShotsOnGoal,
            ms_h.shots_off_goal as Home_ShotsOffGoal,
            ms_h.total_shots as HS,
            ms_h.blocked_shots as Home_BlockedShots,
            ms_h.shots_inside_box as Home_Shots_Inside_Box,
            ms_h.shots_outside_box as Home_ShotsOutsideBox,
            ms_h.fouls as Home_Fouls,
            ms_h.corners as HC,
            ms_h.offsides as Home_Offsides,
            ms_h.possession as Home_Possession,
            ms_h.yellow_cards as HY,
            ms_h.red_cards as HR,
            ms_h.goalkeeper_saves as Home_GKSaves,
            ms_h.total_passes as Home_TotalPasses,
            ms_h.pass_accuracy as Home_Pass_Accuracy,
            ms_h.expected_goals as Home_xG_Stats,  -- kept for diagnostics; home_xG uses COALESCE above
            -- Away team stats from match_stats
            ms_a.shots_on_goal as Away_ShotsOnGoal,
            ms_a.shots_off_goal as Away_ShotsOffGoal,
            ms_a.total_shots as AS_shots,
            ms_a.blocked_shots as Away_BlockedShots,
            ms_a.shots_inside_box as Away_Shots_Inside_Box,
            ms_a.shots_outside_box as Away_ShotsOutsideBox,
            ms_a.fouls as Away_Fouls,
            ms_a.corners as AC,
            ms_a.offsides as Away_Offsides,
            ms_a.possession as Away_Possession,
            ms_a.yellow_cards as AY,
            ms_a.red_cards as AR,
            ms_a.goalkeeper_saves as Away_GKSaves,
            ms_a.total_passes as Away_TotalPasses,
            ms_a.pass_accuracy as Away_Pass_Accuracy,
            ms_a.expected_goals as Away_xG_Stats  -- kept for diagnostics; away_xG uses COALESCE above
        FROM fixtures f
        LEFT JOIN match_stats ms_h ON f.fixture_id = ms_h.fixture_id
            AND f.home_team_id = ms_h.team_id
        LEFT JOIN match_stats ms_a ON f.fixture_id = ms_a.fixture_id
            AND f.away_team_id = ms_a.team_id
        WHERE f.status = ?
    """

    params = [status]

    if seasons:
        placeholders = ','.join(['?'] * len(seasons))
        query += f" AND f.season IN ({placeholders})"
        params.extend(seasons)

    if leagues:
        placeholders = ','.join(['?'] * len(leagues))
        query += f" AND f.league_code IN ({placeholders})"
        params.extend(leagues)

    query += " ORDER BY f.date, f.league_code"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Rename AS_shots to AS (avoiding SQL keyword conflict)
    if 'AS_shots' in df.columns:
        df = df.rename(columns={'AS_shots': 'AS'})

    # Map shots_on_goal to the standard HST/AST column names
    if 'Home_ShotsOnGoal' in df.columns:
        df['HST'] = df['Home_ShotsOnGoal']
    if 'Away_ShotsOnGoal' in df.columns:
        df['AST'] = df['Away_ShotsOnGoal']

    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Ensure numeric columns
    numeric_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'home_xG', 'away_xG', 'Season',
                    'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
                    'Home_Possession', 'Away_Possession', 'Home_Shots_Inside_Box',
                    'Away_Shots_Inside_Box', 'Home_Pass_Accuracy', 'Away_Pass_Accuracy',
                    'Home_GKSaves', 'Away_GKSaves', 'Home_TotalPasses', 'Away_TotalPasses',
                    'Home_Fouls', 'Away_Fouls', 'Home_Offsides', 'Away_Offsides',
                    'Home_BlockedShots', 'Away_BlockedShots',
                    'Home_ShotsOffGoal', 'Away_ShotsOffGoal',
                    'Home_ShotsOutsideBox', 'Away_ShotsOutsideBox']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Log stats availability
    stats_available = df['HS'].notna().sum()
    print(f"  Match stats available: {stats_available}/{len(df)} fixtures "
          f"({stats_available/max(len(df),1)*100:.1f}%)")

    return df


def get_standings_from_db(
    seasons: List[int] = None,
    leagues: List[str] = None
) -> pd.DataFrame:
    """
    Read league standings from shared API-Football database.
    Returns team position, points, goals diff, form string.

    Returns:
        DataFrame with standings data per team/league/season
    """
    if not API_FOOTBALL_DB.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(API_FOOTBALL_DB)

    # Check if standings table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='standings'")
    if not cursor.fetchone():
        conn.close()
        return pd.DataFrame()

    query = """
        SELECT
            league_code as League,
            season as Season,
            team_name as Team,
            rank as LeaguePosition,
            points as LeaguePoints,
            goals_diff as LeagueGD,
            played as LeaguePlayed,
            win as LeagueWins,
            draw as LeagueDraws,
            lose as LeagueLosses,
            goals_for as LeagueGF,
            goals_against as LeagueGA,
            form as LeagueForm
        FROM standings
        WHERE 1=1
    """
    params = []

    if seasons:
        placeholders = ','.join(['?'] * len(seasons))
        query += f" AND season IN ({placeholders})"
        params.extend(seasons)

    if leagues:
        placeholders = ','.join(['?'] * len(leagues))
        query += f" AND league_code IN ({placeholders})"
        params.extend(leagues)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        # Compute win rate and points-per-game
        df['LeaguePlayed'] = pd.to_numeric(df['LeaguePlayed'], errors='coerce')
        df['LeaguePoints'] = pd.to_numeric(df['LeaguePoints'], errors='coerce')
        df['LeaguePosition'] = pd.to_numeric(df['LeaguePosition'], errors='coerce')
        df['LeaguePPG'] = df['LeaguePoints'] / df['LeaguePlayed'].clip(lower=1)

        # Parse form string (e.g. "WWDLW") into numeric recent form
        def form_to_score(form_str):
            if not isinstance(form_str, str):
                return np.nan
            score = 0
            for ch in form_str[-5:]:  # Last 5 results
                if ch == 'W': score += 3
                elif ch == 'D': score += 1
            return score / max(len(form_str[-5:]), 1)
        df['LeagueFormScore'] = df['LeagueForm'].apply(form_to_score)

    return df


def get_injuries_from_db(
    seasons: List[int] = None,
    leagues: List[str] = None
) -> pd.DataFrame:
    """
    Read injury data from shared API-Football database

    Returns:
        DataFrame with injury information
    """
    if not API_FOOTBALL_DB.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(API_FOOTBALL_DB)

    query = """
        SELECT
            i.team_id,
            i.team_name,
            i.player_name,
            i.player_type,
            i.injury_reason,
            f.league_code as League,
            f.season as Season,
            f.date as Date
        FROM injuries i
        JOIN fixtures f ON i.fixture_id = f.fixture_id
        WHERE f.status = 'FT'
    """

    params = []

    if seasons:
        placeholders = ','.join(['?'] * len(seasons))
        query += f" AND f.season IN ({placeholders})"
        params.extend(seasons)

    if leagues:
        placeholders = ','.join(['?'] * len(leagues))
        query += f" AND f.league_code IN ({placeholders})"
        params.extend(leagues)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])

    return df


def get_injury_counts_from_db() -> pd.DataFrame:
    """
    Get aggregated injury counts per team per fixture date.
    Returns a DataFrame with columns: League, Date, Team, InjuryCount
    suitable for merging into features as Home_InjuryCount / Away_InjuryCount.
    """
    if not API_FOOTBALL_DB.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(API_FOOTBALL_DB)

    # Check injuries table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='injuries'")
    if not cursor.fetchone():
        conn.close()
        return pd.DataFrame()

    query = """
        SELECT
            f.league_code as League,
            f.date as Date,
            i.team_name as Team,
            COUNT(*) as InjuryCount
        FROM injuries i
        JOIN fixtures f ON i.fixture_id = f.fixture_id
        WHERE f.status = 'FT'
        GROUP BY f.league_code, f.date, i.team_name
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df['InjuryCount'] = pd.to_numeric(df['InjuryCount'], errors='coerce').fillna(0).astype(int)

    return df


def build_historical_from_api(
    seasons: List[int] = None,
    leagues: List[str] = None,
    force: bool = False
) -> Path:
    """
    Build historical_matches.parquet from API-Football database
    This replaces the old CSV download method

    Args:
        seasons: List of seasons to include
        leagues: List of league codes to include
        force: Force rebuild even if file exists

    Returns:
        Path to historical_matches.parquet
    """
    out_path = PROCESSED_DIR / "historical_matches.parquet"

    if out_path.exists() and not force:
        log_header(f"Historical parquet already exists at {out_path}. Skipping.")
        return out_path

    log_header("Building historical_matches.parquet from API-Football database...")
    print(f"  Database: {API_FOOTBALL_DB}")
    print(f"  Seasons: {seasons}")
    print(f"  Leagues: {leagues}")

    # Get fixtures from database
    df = get_fixtures_from_db(seasons=seasons, leagues=leagues, status='FT')

    if df.empty:
        raise ValueError(
            "No fixtures found in database. "
            "Run dc_laptop/runner.py first to download data."
        )

    print(f"  Loaded {len(df)} fixtures from database")

    # Add standard columns that might be missing
    # (all_models expects these from old CSV format)
    standard_cols = {
        'B365H': np.nan,  # Bet365 home odds (not in API-Football)
        'B365D': np.nan,  # Bet365 draw odds
        'B365A': np.nan,  # Bet365 away odds
        'PSCH': np.nan,   # Pinnacle home odds
        'PSCD': np.nan,   # Pinnacle draw odds
        'PSCA': np.nan,   # Pinnacle away odds
    }

    for col, default_val in standard_cols.items():
        if col not in df.columns:
            df[col] = default_val

    # Ensure expected column order (for compatibility)
    expected_cols = [
        'League', 'Date', 'Season', 'HomeTeam', 'AwayTeam',
        'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG',
        'B365H', 'B365D', 'B365A', 'PSCH', 'PSCD', 'PSCA',
        'home_xG', 'away_xG', 'league_type', 'referee', 'venue_name',
        # Match stats columns (from match_stats table join)
        'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
        'Home_Possession', 'Away_Possession',
        'Home_Shots_Inside_Box', 'Away_Shots_Inside_Box',
        'Home_Pass_Accuracy', 'Away_Pass_Accuracy',
        'Home_GKSaves', 'Away_GKSaves',
        'Home_TotalPasses', 'Away_TotalPasses',
        'Home_Fouls', 'Away_Fouls',
        'Home_Offsides', 'Away_Offsides',
        'Home_BlockedShots', 'Away_BlockedShots',
        'Home_ShotsOffGoal', 'Away_ShotsOffGoal',
        'Home_ShotsOutsideBox', 'Away_ShotsOutsideBox',
    ]

    available_cols = [c for c in expected_cols if c in df.columns]
    df = df[available_cols]

    # Save to parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"  [OK] Saved {len(df)} fixtures to {out_path}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Leagues: {df['League'].nunique()} ({', '.join(sorted(df['League'].unique()))})")
    print(f"  Seasons: {sorted(df['Season'].unique())}")

    # Show xG availability
    xg_available = df['home_xG'].notna().sum()
    print(f"  xG available: {xg_available}/{len(df)} fixtures ({xg_available/max(len(df),1)*100:.1f}%)")

    return out_path


def check_api_football_db() -> bool:
    """
    Check if API-Football database exists and has data

    Returns:
        True if database exists and has fixtures
    """
    if not API_FOOTBALL_DB.exists():
        return False

    try:
        conn = sqlite3.connect(API_FOOTBALL_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM fixtures")
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


if __name__ == "__main__":
    # Test the adapter
    print("Testing API-Football Database Adapter...")
    print("=" * 60)

    if not check_api_football_db():
        print("[ERROR] API-Football database not found or empty")
        print(f"   Expected location: {API_FOOTBALL_DB}")
        print("\n   Run dc_laptop/runner.py first to download data.")
    else:
        print("[OK] API-Football database found")

        # Test reading fixtures
        df = get_fixtures_from_db(seasons=[2024, 2025])
        print(f"\n[OK] Loaded {len(df)} fixtures")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  Leagues: {df['League'].nunique()}")
        print(f"  Seasons: {sorted(df['Season'].unique())}")

        # Show sample
        print("\nSample data:")
        print(df[['Date', 'League', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'home_xG', 'away_xG']].head(3))

        # Test building historical
        print("\n" + "=" * 60)
        print("Building historical_matches.parquet...")
        path = build_historical_from_api(
            seasons=[2024, 2025],
            leagues=['E0', 'E1', 'D1', 'SP1', 'I1', 'F1'],
            force=True
        )
        print(f"\n[OK] Successfully created {path}")
