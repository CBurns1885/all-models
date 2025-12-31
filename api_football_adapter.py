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
    Read fixtures from shared API-Football database

    Args:
        seasons: List of seasons (e.g., [2024, 2025])
        leagues: List of league codes (e.g., ['E0', 'E1'])
        status: Match status ('FT' for finished, 'NS' for not started)

    Returns:
        DataFrame with columns matching old CSV format + new xG columns
    """
    if not API_FOOTBALL_DB.exists():
        raise FileNotFoundError(
            f"API-Football database not found at {API_FOOTBALL_DB}. "
            f"Run dc_laptop/runner.py first to download data."
        )

    conn = sqlite3.connect(API_FOOTBALL_DB)

    # Build query
    query = """
        SELECT
            league_code as League,
            season as Season,
            date as Date,
            home_team as HomeTeam,
            away_team as AwayTeam,
            home_goals as FTHG,
            away_goals as FTAG,
            ht_home_goals as HTHG,
            ht_away_goals as HTAG,
            CASE
                WHEN home_goals > away_goals THEN 'H'
                WHEN home_goals < away_goals THEN 'A'
                ELSE 'D'
            END as FTR,
            home_xG,
            away_xG,
            league_type,
            referee,
            venue_name
        FROM fixtures
        WHERE status = ?
    """

    params = [status]

    if seasons:
        placeholders = ','.join(['?'] * len(seasons))
        query += f" AND season IN ({placeholders})"
        params.extend(seasons)

    if leagues:
        placeholders = ','.join(['?'] * len(leagues))
        query += f" AND league_code IN ({placeholders})"
        params.extend(leagues)

    query += " ORDER BY date, league_code"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Ensure numeric columns
    numeric_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'home_xG', 'away_xG', 'Season']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

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
        'home_xG', 'away_xG', 'league_type', 'referee', 'venue_name'
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
    print(f"  xG available: {xg_available}/{len(df)} fixtures ({xg_available/len(df)*100:.1f}%)")

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
