#!/usr/bin/env python3
"""
API-Football Client for fetching fixtures and historical data
Integrates with https://v3.football.api-sports.io/
Creates and populates shared SQLite database at ../data/football_api.db
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from config import (
    API_FOOTBALL_KEY, API_FOOTBALL_BASE, API_FOOTBALL_DB,
    API_LEAGUE_MAP, DATA_DIR, log_header
)

# Rate limiting
RATE_LIMIT_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3

# API Headers
def _get_headers():
    return {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key": API_FOOTBALL_KEY
    }


def test_api_connection() -> bool:
    """
    Test if API-Football connection is working

    Returns:
        True if connection successful
    """
    try:
        url = f"{API_FOOTBALL_BASE}/status"
        response = requests.get(url, headers=_get_headers(), timeout=10)

        if response.status_code == 200:
            data = response.json()
            if 'errors' in data and data['errors']:
                print(f"[WARN] API errors: {data['errors']}")
                return False

            # Check remaining requests
            if 'response' in data:
                account = data['response'].get('account', {})
                requests_info = data['response'].get('requests', {})

                current = requests_info.get('current', 0)
                limit_day = requests_info.get('limit_day', 7500)

                print(f"[OK] API Status: {account.get('firstname', 'User')}")
                print(f"   Requests today: {current}/{limit_day}")

                if current >= limit_day - 100:
                    print("[WARN] Approaching daily limit!")

            return True
        else:
            print(f"[ERROR] API returned status {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Connection failed: {e}")
        return False


def _make_request(endpoint: str, params: Dict = None) -> Optional[Dict]:
    """
    Make a request to API-Football with retry logic

    Args:
        endpoint: API endpoint (e.g., 'fixtures')
        params: Query parameters

    Returns:
        Response JSON or None on failure
    """
    url = f"{API_FOOTBALL_BASE}/{endpoint}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url,
                headers=_get_headers(),
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                # Check for API errors
                if 'errors' in data and data['errors']:
                    if isinstance(data['errors'], dict):
                        errors = list(data['errors'].values())
                    else:
                        errors = data['errors']
                    print(f"[WARN] API errors: {errors}")

                time.sleep(RATE_LIMIT_DELAY)
                return data

            elif response.status_code == 429:
                # Rate limited
                wait_time = 2 ** (attempt + 1)
                print(f"[WARN] Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)

            else:
                print(f"[ERROR] Request failed: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            wait_time = 2 ** attempt
            print(f"[WARN] Request error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    return None


def _init_database():
    """
    Initialize the SQLite database with required tables
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(API_FOOTBALL_DB)
    cursor = conn.cursor()

    # Fixtures table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fixtures (
            fixture_id INTEGER PRIMARY KEY,
            league_code TEXT,
            league_id INTEGER,
            season INTEGER,
            date TEXT,
            home_team TEXT,
            away_team TEXT,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_goals INTEGER,
            away_goals INTEGER,
            ht_home_goals INTEGER,
            ht_away_goals INTEGER,
            home_xG REAL,
            away_xG REAL,
            status TEXT,
            referee TEXT,
            venue_name TEXT,
            league_type TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Injuries table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS injuries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER,
            team_id INTEGER,
            team_name TEXT,
            player_name TEXT,
            player_type TEXT,
            injury_reason TEXT,
            date TEXT,
            FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
        )
    """)

    # Statistics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS match_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER,
            team_id INTEGER,
            team_name TEXT,
            shots_on_goal INTEGER,
            shots_off_goal INTEGER,
            total_shots INTEGER,
            blocked_shots INTEGER,
            shots_inside_box INTEGER,
            shots_outside_box INTEGER,
            fouls INTEGER,
            corners INTEGER,
            offsides INTEGER,
            possession REAL,
            yellow_cards INTEGER,
            red_cards INTEGER,
            goalkeeper_saves INTEGER,
            total_passes INTEGER,
            pass_accuracy REAL,
            expected_goals REAL,
            FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
        )
    """)

    # Standings table (league position, points, form)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS standings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            league_id INTEGER,
            league_code TEXT,
            season INTEGER,
            team_id INTEGER,
            team_name TEXT,
            rank INTEGER,
            points INTEGER,
            goals_diff INTEGER,
            played INTEGER,
            win INTEGER,
            draw INTEGER,
            lose INTEGER,
            goals_for INTEGER,
            goals_against INTEGER,
            form TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indices for faster queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_league ON fixtures(league_code)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_status ON fixtures(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_season ON fixtures(season)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_standings_team ON standings(team_name, league_code, season)")

    conn.commit()
    conn.close()

    print(f"[OK] Database initialized: {API_FOOTBALL_DB}")


def _get_league_type(league_id: int) -> str:
    """Determine league quality tier"""
    elite_leagues = [39, 140, 135, 78, 61, 203]  # PL, La Liga, Serie A, Bundesliga, Ligue 1, Süper Lig
    high_leagues = [40, 141, 136, 79, 62, 88, 144, 94, 179]  # Championship, La Liga 2, etc.

    if league_id in elite_leagues:
        return 'elite'
    elif league_id in high_leagues:
        return 'high'
    else:
        return 'medium'


def fetch_fixtures_for_league(league_code: str, season: int, status: str = 'FT') -> int:
    """
    Fetch fixtures for a specific league and season

    Args:
        league_code: football-data.co.uk league code (e.g., 'E0')
        season: Season year (e.g., 2024 for 2024-25 season)
        status: 'FT' for finished, 'NS' for not started, 'ALL' for all

    Returns:
        Number of fixtures fetched
    """
    if league_code not in API_LEAGUE_MAP:
        print(f"[WARN] Unknown league code: {league_code}")
        return 0

    league_id = API_LEAGUE_MAP[league_code]

    params = {
        'league': league_id,
        'season': season
    }

    if status != 'ALL':
        params['status'] = status

    data = _make_request('fixtures', params)

    if not data or 'response' not in data:
        return 0

    fixtures = data['response']

    if not fixtures:
        return 0

    # Store in database
    conn = sqlite3.connect(API_FOOTBALL_DB)
    cursor = conn.cursor()

    count = 0
    for fixture in fixtures:
        try:
            fixture_data = fixture.get('fixture', {})
            league_data = fixture.get('league', {})
            teams = fixture.get('teams', {})
            goals = fixture.get('goals', {})
            score = fixture.get('score', {})

            fixture_id = fixture_data.get('id')

            # Extract halftime score
            ht_score = score.get('halftime', {})
            ht_home = ht_score.get('home') if ht_score else None
            ht_away = ht_score.get('away') if ht_score else None

            cursor.execute("""
                INSERT OR REPLACE INTO fixtures
                (fixture_id, league_code, league_id, season, date,
                 home_team, away_team, home_team_id, away_team_id,
                 home_goals, away_goals, ht_home_goals, ht_away_goals,
                 status, referee, venue_name, league_type, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                fixture_id,
                league_code,
                league_id,
                season,
                fixture_data.get('date', '')[:10],  # Date only
                teams.get('home', {}).get('name'),
                teams.get('away', {}).get('name'),
                teams.get('home', {}).get('id'),
                teams.get('away', {}).get('id'),
                goals.get('home'),
                goals.get('away'),
                ht_home,
                ht_away,
                fixture_data.get('status', {}).get('short', 'NS'),
                fixture_data.get('referee'),
                fixture_data.get('venue', {}).get('name'),
                _get_league_type(league_id)
            ))

            count += 1

        except Exception as e:
            print(f"[WARN] Error processing fixture: {e}")
            continue

    conn.commit()
    conn.close()

    return count


def fetch_fixture_statistics(fixture_id: int) -> bool:
    """
    Fetch detailed statistics for a fixture

    Args:
        fixture_id: API-Football fixture ID

    Returns:
        True if successful
    """
    data = _make_request('fixtures/statistics', {'fixture': fixture_id})

    if not data or 'response' not in data:
        return False

    stats = data['response']

    conn = sqlite3.connect(API_FOOTBALL_DB)
    cursor = conn.cursor()

    for team_stats in stats:
        team = team_stats.get('team', {})
        statistics = team_stats.get('statistics', [])

        # Parse statistics into dict
        stat_dict = {}
        for stat in statistics:
            stat_type = stat.get('type', '').lower().replace(' ', '_')
            value = stat.get('value')
            stat_dict[stat_type] = value

        # Handle possession percentage
        possession = stat_dict.get('ball_possession', '0%')
        if possession and isinstance(possession, str):
            possession = float(possession.replace('%', '')) if '%' in possession else 0

        # Handle pass accuracy
        pass_acc = stat_dict.get('passes_%', '0%')
        if pass_acc and isinstance(pass_acc, str):
            pass_acc = float(pass_acc.replace('%', '')) if '%' in pass_acc else 0

        cursor.execute("""
            INSERT OR REPLACE INTO match_stats
            (fixture_id, team_id, team_name, shots_on_goal, shots_off_goal,
             total_shots, blocked_shots, shots_inside_box, shots_outside_box,
             fouls, corners, offsides, possession, yellow_cards, red_cards,
             goalkeeper_saves, total_passes, pass_accuracy, expected_goals)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fixture_id,
            team.get('id'),
            team.get('name'),
            stat_dict.get('shots_on_goal'),
            stat_dict.get('shots_off_goal'),
            stat_dict.get('total_shots'),
            stat_dict.get('blocked_shots'),
            stat_dict.get('shots_insidebox'),
            stat_dict.get('shots_outsidebox'),
            stat_dict.get('fouls'),
            stat_dict.get('corner_kicks'),
            stat_dict.get('offsides'),
            possession,
            stat_dict.get('yellow_cards'),
            stat_dict.get('red_cards'),
            stat_dict.get('goalkeeper_saves'),
            stat_dict.get('total_passes'),
            pass_acc,
            stat_dict.get('expected_goals')
        ))

    conn.commit()
    conn.close()

    return True


def download_upcoming_fixtures(leagues: List[str] = None, days_ahead: int = 7) -> pd.DataFrame:
    """
    Download upcoming fixtures for specified leagues

    Args:
        leagues: List of league codes (e.g., ['E0', 'D1'])
        days_ahead: Number of days ahead to fetch

    Returns:
        DataFrame with upcoming fixtures
    """
    if leagues is None:
        leagues = list(API_LEAGUE_MAP.keys())

    log_header("Fetching Upcoming Fixtures from API-Football")

    _init_database()

    today = datetime.now()
    end_date = today + timedelta(days=days_ahead)

    all_fixtures = []

    for league_code in leagues:
        if league_code not in API_LEAGUE_MAP:
            continue

        league_id = API_LEAGUE_MAP[league_code]

        # Determine current season (e.g., 2025 for 2025-2026 season)
        current_month = today.month
        current_season = today.year if current_month >= 7 else today.year - 1

        params = {
            'league': league_id,
            'season': current_season,
            'from': today.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        }

        data = _make_request('fixtures', params)

        if not data or 'response' not in data:
            continue

        for fixture in data['response']:
            try:
                fixture_data = fixture.get('fixture', {})
                teams = fixture.get('teams', {})

                all_fixtures.append({
                    'Date': fixture_data.get('date', '')[:10],
                    'League': league_code,
                    'HomeTeam': teams.get('home', {}).get('name'),
                    'AwayTeam': teams.get('away', {}).get('name'),
                    'fixture_id': fixture_data.get('id'),
                    'venue': fixture_data.get('venue', {}).get('name'),
                    'referee': fixture_data.get('referee')
                })
            except Exception as e:
                print(f"[WARN] Error parsing fixture: {e}")
                continue

        print(f"  {league_code}: {len([f for f in all_fixtures if f['League'] == league_code])} fixtures")

    if not all_fixtures:
        return pd.DataFrame()

    df = pd.DataFrame(all_fixtures)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Date', 'League']).reset_index(drop=True)

    print(f"\n[OK] Total: {len(df)} upcoming fixtures")

    return df


def populate_historical_data(leagues: List[str] = None, seasons: List[int] = None,
                            fetch_stats: bool = False) -> int:
    """
    Populate database with historical match data

    Args:
        leagues: List of league codes
        seasons: List of season start years (e.g., [2023, 2024])
        fetch_stats: Whether to also fetch detailed match statistics

    Returns:
        Total fixtures fetched
    """
    if leagues is None:
        leagues = list(API_LEAGUE_MAP.keys())

    if seasons is None:
        current_year = datetime.now().year
        current_month = datetime.now().month
        current_season = current_year if current_month >= 7 else current_year - 1
        seasons = [current_season - 1, current_season]

    log_header("Populating Historical Data from API-Football")
    print(f"Leagues: {leagues}")
    print(f"Seasons: {seasons}")

    _init_database()

    total = 0

    for season in seasons:
        print(f"\n--- Season {season}/{season+1} ---")

        for league_code in leagues:
            count = fetch_fixtures_for_league(league_code, season, status='FT')
            total += count

            if count > 0:
                print(f"  {league_code}: {count} matches")

            # Small delay to respect rate limits
            time.sleep(0.3)

    print(f"\n[OK] Total matches fetched: {total}")

    # Fetch statistics for recent matches if requested
    if fetch_stats and total > 0:
        print("\nFetching match statistics (this may take a while)...")

        conn = sqlite3.connect(API_FOOTBALL_DB)
        cursor = conn.cursor()

        # Get fixtures without statistics
        cursor.execute("""
            SELECT f.fixture_id FROM fixtures f
            LEFT JOIN match_stats ms ON f.fixture_id = ms.fixture_id
            WHERE f.status = 'FT' AND ms.fixture_id IS NULL
            ORDER BY f.date DESC
            LIMIT 500
        """)

        fixtures_to_update = [row[0] for row in cursor.fetchall()]
        conn.close()

        stats_count = 0
        for fixture_id in fixtures_to_update:
            if fetch_fixture_statistics(fixture_id):
                stats_count += 1

            if stats_count % 50 == 0:
                print(f"  Statistics fetched: {stats_count}/{len(fixtures_to_update)}")

        print(f"[OK] Statistics fetched for {stats_count} matches")

    return total


def fetch_standings(league_code: str, season: int) -> int:
    """
    Fetch league standings (table position, points, form) for a league/season.
    Stores in the standings table.

    Args:
        league_code: football-data.co.uk league code (e.g., 'E0')
        season: Season year (e.g., 2024 for 2024-25)

    Returns:
        Number of teams in standings
    """
    if league_code not in API_LEAGUE_MAP:
        return 0

    league_id = API_LEAGUE_MAP[league_code]
    data = _make_request('standings', {'league': league_id, 'season': season})

    if not data or 'response' not in data or not data['response']:
        return 0

    _init_database()
    conn = sqlite3.connect(API_FOOTBALL_DB)
    cursor = conn.cursor()

    count = 0
    for league_data in data['response']:
        standings_list = league_data.get('league', {}).get('standings', [])
        for group in standings_list:
            for team_entry in group:
                team = team_entry.get('team', {})
                stats_all = team_entry.get('all', {})

                cursor.execute("""
                    INSERT OR REPLACE INTO standings
                    (league_id, league_code, season, team_id, team_name,
                     rank, points, goals_diff, played, win, draw, lose,
                     goals_for, goals_against, form, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    league_id,
                    league_code,
                    season,
                    team.get('id'),
                    team.get('name'),
                    team_entry.get('rank'),
                    team_entry.get('points'),
                    team_entry.get('goalsDiff'),
                    stats_all.get('played'),
                    stats_all.get('win'),
                    stats_all.get('draw'),
                    stats_all.get('lose'),
                    stats_all.get('goals', {}).get('for'),
                    stats_all.get('goals', {}).get('against'),
                    team_entry.get('form'),
                ))
                count += 1

    conn.commit()
    conn.close()

    return count


def fetch_injuries_for_fixture(fixture_id: int) -> List[Dict]:
    """
    Fetch injury/lineup data for a specific fixture

    Args:
        fixture_id: API-Football fixture ID

    Returns:
        List of injury records
    """
    data = _make_request('injuries', {'fixture': fixture_id})

    if not data or 'response' not in data:
        return []

    injuries = []
    for injury in data['response']:
        player = injury.get('player', {})
        team = injury.get('team', {})

        injuries.append({
            'fixture_id': fixture_id,
            'team_id': team.get('id'),
            'team_name': team.get('name'),
            'player_id': player.get('id'),
            'player_name': player.get('name'),
            'player_type': player.get('type'),  # 'Missing Fixture' or type
            'injury_reason': player.get('reason')
        })

    return injuries


def fetch_lineups_for_fixture(fixture_id: int) -> Dict:
    """
    Fetch confirmed lineups for a specific fixture
    Usually available ~1 hour before kickoff

    Args:
        fixture_id: API-Football fixture ID

    Returns:
        Dictionary with home and away lineups
    """
    data = _make_request('fixtures/lineups', {'fixture': fixture_id})

    if not data or 'response' not in data or not data['response']:
        return {}

    result = {'home': None, 'away': None}

    for team_data in data['response']:
        team = team_data.get('team', {})
        team_id = team.get('id')
        formation = team_data.get('formation')

        starters = []
        for player in team_data.get('startXI', []):
            p = player.get('player', {})
            starters.append({
                'id': p.get('id'),
                'name': p.get('name'),
                'number': p.get('number'),
                'pos': p.get('pos'),
                'grid': p.get('grid')
            })

        subs = []
        for player in team_data.get('substitutes', []):
            p = player.get('player', {})
            subs.append({
                'id': p.get('id'),
                'name': p.get('name'),
                'number': p.get('number'),
                'pos': p.get('pos')
            })

        lineup_data = {
            'team_id': team_id,
            'team_name': team.get('name'),
            'formation': formation,
            'starters': starters,
            'substitutes': subs
        }

        # Determine if home or away (first is usually home)
        if result['home'] is None:
            result['home'] = lineup_data
        else:
            result['away'] = lineup_data

    return result


def fetch_live_data_for_upcoming(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch injuries and lineups for upcoming fixtures
    Adds injury count and lineup status columns

    Args:
        fixtures_df: DataFrame with upcoming fixtures (must have fixture_id column)

    Returns:
        DataFrame with added injury/lineup columns
    """
    if 'fixture_id' not in fixtures_df.columns:
        print("[WARN] No fixture_id column - cannot fetch live data")
        return fixtures_df

    log_header("Fetching Live Data (Injuries & Lineups)")

    df = fixtures_df.copy()
    df['home_injuries'] = 0
    df['away_injuries'] = 0
    df['home_injury_players'] = ''
    df['away_injury_players'] = ''
    df['home_formation'] = ''
    df['away_formation'] = ''
    df['lineups_confirmed'] = False

    for idx, row in df.iterrows():
        fixture_id = row['fixture_id']
        home_team = row.get('HomeTeam', '')
        away_team = row.get('AwayTeam', '')

        # Fetch injuries
        injuries = fetch_injuries_for_fixture(fixture_id)

        if injuries:
            home_inj = [i for i in injuries if i['team_name'] == home_team]
            away_inj = [i for i in injuries if i['team_name'] == away_team]

            df.at[idx, 'home_injuries'] = len(home_inj)
            df.at[idx, 'away_injuries'] = len(away_inj)
            df.at[idx, 'home_injury_players'] = ', '.join([i['player_name'] for i in home_inj[:5]])
            df.at[idx, 'away_injury_players'] = ', '.join([i['player_name'] for i in away_inj[:5]])

        # Fetch lineups (if available)
        lineups = fetch_lineups_for_fixture(fixture_id)

        if lineups and lineups.get('home') and lineups.get('away'):
            df.at[idx, 'home_formation'] = lineups['home'].get('formation', '')
            df.at[idx, 'away_formation'] = lineups['away'].get('formation', '')
            df.at[idx, 'lineups_confirmed'] = True

        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} fixtures...")

    # Summary
    total_home_inj = df['home_injuries'].sum()
    total_away_inj = df['away_injuries'].sum()
    lineups_available = df['lineups_confirmed'].sum()

    print(f"\n[OK] Live data fetched:")
    print(f"  Total injuries: {total_home_inj} (home) + {total_away_inj} (away)")
    print(f"  Lineups confirmed: {lineups_available}/{len(df)} fixtures")

    return df


def get_database_stats() -> Dict:
    """
    Get statistics about the current database

    Returns:
        Dictionary with database statistics
    """
    if not API_FOOTBALL_DB.exists():
        return {'exists': False}

    conn = sqlite3.connect(API_FOOTBALL_DB)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM fixtures")
    total_fixtures = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM fixtures WHERE status = 'FT'")
    completed_fixtures = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(date), MAX(date) FROM fixtures WHERE status = 'FT'")
    date_range = cursor.fetchone()

    cursor.execute("SELECT DISTINCT league_code FROM fixtures")
    leagues = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT DISTINCT season FROM fixtures ORDER BY season")
    seasons = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT COUNT(*) FROM match_stats")
    stats_count = cursor.fetchone()[0]

    conn.close()

    return {
        'exists': True,
        'total_fixtures': total_fixtures,
        'completed_fixtures': completed_fixtures,
        'date_range': date_range,
        'leagues': leagues,
        'seasons': seasons,
        'stats_available': stats_count
    }


def download_missing_leagues(seasons: List[int] = None) -> int:
    """
    Download data for leagues that are missing from the database.
    Useful for adding cups (UCL, UEL, FAC, etc.) and smaller leagues (CRO, SWZ, etc.)

    Args:
        seasons: List of season years to fetch (defaults to current + previous)

    Returns:
        Total fixtures downloaded
    """
    from config import LEAGUE_CODES, API_LEAGUE_MAP

    if seasons is None:
        current_year = datetime.now().year
        current_month = datetime.now().month
        current_season = current_year if current_month >= 7 else current_year - 1
        seasons = [current_season - 1, current_season]

    log_header("Downloading Missing Leagues")

    # Get leagues already in database
    stats = get_database_stats()
    existing_leagues = set(stats.get('leagues', []))

    # Find missing leagues that have API mappings
    missing = []
    for lg in LEAGUE_CODES:
        if lg not in existing_leagues and lg in API_LEAGUE_MAP:
            missing.append(lg)

    if not missing:
        print("[OK] All configured leagues already have data!")
        return 0

    print(f"Missing leagues to download: {missing}")
    print(f"Seasons: {seasons}")

    return populate_historical_data(leagues=missing, seasons=seasons, fetch_stats=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="API-Football Data Client")
    parser.add_argument('--download-missing', action='store_true',
                       help='Download data for missing leagues (cups, smaller leagues)')
    parser.add_argument('--download-all', action='store_true',
                       help='Download all configured leagues')
    parser.add_argument('--seasons', nargs='+', type=int, default=None,
                       help='Seasons to download (e.g., 2023 2024 2025)')
    args = parser.parse_args()

    print("="*60)
    print("API-FOOTBALL CLIENT")
    print("="*60)

    # Test connection
    print("\n1. Testing API connection...")
    if test_api_connection():
        print("\n2. Checking database status...")
        stats = get_database_stats()

        if stats['exists']:
            print(f"   Total fixtures: {stats['total_fixtures']}")
            print(f"   Completed matches: {stats['completed_fixtures']}")
            print(f"   Date range: {stats['date_range']}")
            print(f"   Leagues: {len(stats['leagues'])} ({', '.join(sorted(stats['leagues']))})")
            print(f"   Seasons: {stats['seasons']}")
        else:
            print("   Database not found. Initializing...")
            _init_database()

        # Handle command line options
        if args.download_missing:
            download_missing_leagues(seasons=args.seasons)
        elif args.download_all:
            from config import LEAGUE_CODES
            seasons = args.seasons or [2023, 2024, 2025]
            populate_historical_data(leagues=LEAGUE_CODES, seasons=seasons)
        else:
            # Interactive mode
            print("\n3. Options:")
            print("   [1] Download missing leagues (cups + smaller leagues)")
            print("   [2] Download ALL configured leagues")
            print("   [3] Exit")
            choice = input("   Enter choice (1/2/3): ").strip()

            if choice == '1':
                download_missing_leagues()
            elif choice == '2':
                from config import LEAGUE_CODES
                populate_historical_data(
                    leagues=LEAGUE_CODES,
                    seasons=[2023, 2024, 2025]
                )
    else:
        print("\n[ERROR] API connection failed. Check your API key.")
