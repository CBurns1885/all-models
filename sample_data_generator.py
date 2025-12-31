#!/usr/bin/env python3
"""
Sample Data Generator for Football Prediction System
Creates realistic historical match data for testing when API is unavailable
"""

import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from config import API_FOOTBALL_DB, DATA_DIR, PROCESSED_DIR

# Team names by league
LEAGUE_TEAMS = {
    'E0': [  # Premier League
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
        'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich',
        'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
        'Nott\'m Forest', 'Southampton', 'Tottenham', 'West Ham', 'Wolves'
    ],
    'E1': [  # Championship
        'Blackburn', 'Bristol City', 'Burnley', 'Cardiff', 'Coventry',
        'Derby', 'Hull', 'Leeds', 'Luton', 'Middlesbrough',
        'Millwall', 'Norwich', 'Oxford', 'Plymouth', 'Portsmouth',
        'Preston', 'QPR', 'Sheffield Utd', 'Sheffield Wed', 'Stoke',
        'Sunderland', 'Swansea', 'Watford', 'West Brom'
    ],
    'D1': [  # Bundesliga
        'Bayern Munich', 'Dortmund', 'RB Leipzig', 'Leverkusen', 'Frankfurt',
        'Wolfsburg', 'Freiburg', 'Hoffenheim', 'Mainz', 'Union Berlin',
        'Werder Bremen', 'Bochum', 'Augsburg', 'Stuttgart', 'Monchengladbach',
        'Koln', 'Heidenheim', 'Darmstadt'
    ],
    'SP1': [  # La Liga
        'Real Madrid', 'Barcelona', 'Ath Madrid', 'Sevilla', 'Valencia',
        'Betis', 'Real Sociedad', 'Villarreal', 'Ath Bilbao', 'Celta',
        'Osasuna', 'Mallorca', 'Getafe', 'Rayo Vallecano', 'Girona',
        'Alaves', 'Las Palmas', 'Cadiz', 'Granada', 'Almeria'
    ],
    'I1': [  # Serie A
        'Inter', 'Juventus', 'AC Milan', 'Napoli', 'Roma', 'Lazio',
        'Atalanta', 'Fiorentina', 'Bologna', 'Torino', 'Udinese',
        'Sassuolo', 'Monza', 'Lecce', 'Genoa', 'Cagliari',
        'Empoli', 'Verona', 'Salernitana', 'Frosinone'
    ],
    'F1': [  # Ligue 1
        'Paris SG', 'Monaco', 'Lyon', 'Marseille', 'Lille', 'Nice',
        'Lens', 'Rennes', 'Montpellier', 'Toulouse', 'Nantes', 'Strasbourg',
        'Lorient', 'Brest', 'Reims', 'Le Havre', 'Metz', 'Clermont'
    ],
    'N1': [  # Eredivisie
        'Ajax', 'PSV', 'Feyenoord', 'AZ Alkmaar', 'Twente', 'Utrecht',
        'Vitesse', 'Heerenveen', 'Groningen', 'Sparta Rotterdam',
        'NEC Nijmegen', 'RKC Waalwijk', 'Fortuna Sittard', 'Go Ahead Eagles',
        'Excelsior', 'Volendam', 'Heracles', 'Almere City'
    ],
    'T1': [  # Super Lig
        'Galatasaray', 'Fenerbahce', 'Besiktas', 'Trabzonspor', 'Basaksehir',
        'Antalyaspor', 'Alanyaspor', 'Sivasspor', 'Kasimpasa', 'Konyaspor',
        'Adana Demirspor', 'Hatayspor', 'Kayserispor', 'Pendikspor',
        'Samsunspor', 'Rizespor', 'Fatih Karagumruk', 'Ankaragucu'
    ],
    'SC0': [  # Scottish Premiership
        'Celtic', 'Rangers', 'Hearts', 'Aberdeen', 'Hibernian',
        'Dundee Utd', 'Motherwell', 'St Mirren', 'Ross County',
        'Kilmarnock', 'Livingston', 'St Johnstone'
    ],
    'B1': [  # Belgian Pro League
        'Club Brugge', 'Anderlecht', 'Antwerp', 'Genk', 'Gent',
        'Standard Liege', 'Union SG', 'Charleroi', 'Mechelen',
        'Westerlo', 'Cercle Brugge', 'Kortrijk', 'Leuven', 'Sint-Truiden',
        'Eupen', 'Seraing'
    ],
    'P1': [  # Portuguese Primeira Liga
        'Benfica', 'Porto', 'Sporting CP', 'Braga', 'Vitoria SC',
        'Boavista', 'Santa Clara', 'Gil Vicente', 'Famalicao', 'Estoril',
        'Rio Ave', 'Arouca', 'Vizela', 'Maritimo', 'Chaves', 'Casa Pia',
        'Portimonense', 'Estrela'
    ]
}

# League scoring profiles (avg goals per game, home win rate, draw rate)
LEAGUE_PROFILES = {
    'E0': (2.7, 0.44, 0.25),   # Premier League
    'E1': (2.6, 0.42, 0.26),   # Championship
    'D1': (3.0, 0.43, 0.23),   # Bundesliga
    'SP1': (2.5, 0.46, 0.26),  # La Liga
    'I1': (2.7, 0.45, 0.24),   # Serie A
    'F1': (2.6, 0.45, 0.25),   # Ligue 1
    'N1': (2.9, 0.42, 0.24),   # Eredivisie
    'T1': (3.1, 0.48, 0.22),   # Super Lig
    'SC0': (2.6, 0.44, 0.25),  # Scottish Premiership
    'B1': (2.8, 0.43, 0.24),   # Belgian Pro League
    'P1': (2.5, 0.47, 0.25),   # Portuguese Liga
}


def generate_score(avg_goals: float, home_advantage: float = 0.3) -> tuple:
    """Generate realistic match score using Poisson distribution"""
    # Split average goals between home and away with home advantage
    home_lambda = (avg_goals / 2) * (1 + home_advantage)
    away_lambda = (avg_goals / 2) * (1 - home_advantage * 0.5)

    home_goals = np.random.poisson(home_lambda)
    away_goals = np.random.poisson(away_lambda)

    # Cap at reasonable max
    return min(home_goals, 8), min(away_goals, 8)


def generate_halftime_score(home_goals: int, away_goals: int) -> tuple:
    """Generate realistic halftime score based on full time"""
    # Typically 40-60% of goals scored by halftime
    ht_home = random.randint(0, home_goals)
    ht_away = random.randint(0, away_goals)
    return ht_home, ht_away


def generate_season_fixtures(league_code: str, season: int, teams: list) -> list:
    """Generate a full season of fixtures (double round-robin)"""
    fixtures = []
    profile = LEAGUE_PROFILES.get(league_code, (2.7, 0.44, 0.25))
    avg_goals, home_rate, draw_rate = profile

    # Generate fixture dates throughout the season
    season_start = datetime(season, 8, 1)  # Season starts August
    season_end = datetime(season + 1, 5, 31)  # Season ends May

    # Create round-robin schedule
    n_teams = len(teams)
    matchdays = []

    # Generate all possible matchups (home and away)
    for home_team in teams:
        for away_team in teams:
            if home_team != away_team:
                matchdays.append((home_team, away_team))

    # Shuffle and distribute across season
    random.shuffle(matchdays)

    days_in_season = (season_end - season_start).days

    for i, (home, away) in enumerate(matchdays):
        # Distribute matches across season
        days_offset = int((i / len(matchdays)) * days_in_season)
        match_date = season_start + timedelta(days=days_offset)

        # Skip to weekend if weekday
        if match_date.weekday() < 5:  # Mon-Fri
            match_date += timedelta(days=(5 - match_date.weekday()))

        # Generate score
        home_goals, away_goals = generate_score(avg_goals)
        ht_home, ht_away = generate_halftime_score(home_goals, away_goals)

        fixtures.append({
            'league_code': league_code,
            'season': season,
            'date': match_date.strftime('%Y-%m-%d'),
            'home_team': home,
            'away_team': away,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'ht_home_goals': ht_home,
            'ht_away_goals': ht_away,
            'status': 'FT',
            'league_type': 'elite' if league_code in ['E0', 'D1', 'SP1', 'I1', 'F1'] else 'high'
        })

    return fixtures


def generate_upcoming_fixtures(league_code: str, teams: list, days_ahead: int = 7) -> list:
    """Generate upcoming fixtures for predictions"""
    fixtures = []
    today = datetime.now()

    # Generate some matches for each day
    for day_offset in range(1, days_ahead + 1):
        match_date = today + timedelta(days=day_offset)

        # 2-4 matches per day per league
        n_matches = random.randint(2, 4)
        available_teams = teams.copy()
        random.shuffle(available_teams)

        for i in range(0, min(n_matches * 2, len(available_teams)), 2):
            if i + 1 < len(available_teams):
                fixtures.append({
                    'Date': match_date.strftime('%Y-%m-%d'),
                    'League': league_code,
                    'HomeTeam': available_teams[i],
                    'AwayTeam': available_teams[i + 1]
                })

    return fixtures


def initialize_database_with_sample_data(leagues: list = None, seasons: list = None):
    """
    Create and populate database with sample historical data

    Args:
        leagues: List of league codes to generate
        seasons: List of seasons (start years)
    """
    if leagues is None:
        leagues = ['E0', 'E1', 'D1', 'SP1', 'I1', 'F1']

    if seasons is None:
        current_year = datetime.now().year
        current_month = datetime.now().month
        current_season = current_year if current_month >= 7 else current_year - 1
        seasons = [current_season - 1, current_season]

    print("="*60)
    print("GENERATING SAMPLE DATA")
    print("="*60)
    print(f"Leagues: {leagues}")
    print(f"Seasons: {seasons}")

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Create database
    conn = sqlite3.connect(API_FOOTBALL_DB)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fixtures (
            fixture_id INTEGER PRIMARY KEY AUTOINCREMENT,
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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS injuries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER,
            team_id INTEGER,
            team_name TEXT,
            player_name TEXT,
            player_type TEXT,
            injury_reason TEXT,
            date TEXT
        )
    """)

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
            expected_goals REAL
        )
    """)

    # Create indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_league ON fixtures(league_code)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_status ON fixtures(status)")

    conn.commit()

    # Clear existing data
    cursor.execute("DELETE FROM fixtures")
    conn.commit()

    # Generate fixtures for each league and season
    total_fixtures = 0

    for league in leagues:
        teams = LEAGUE_TEAMS.get(league, [])
        if not teams:
            print(f"[WARN] No teams defined for {league}")
            continue

        for season in seasons:
            fixtures = generate_season_fixtures(league, season, teams)

            for f in fixtures:
                cursor.execute("""
                    INSERT INTO fixtures
                    (league_code, season, date, home_team, away_team,
                     home_goals, away_goals, ht_home_goals, ht_away_goals,
                     status, league_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f['league_code'], f['season'], f['date'],
                    f['home_team'], f['away_team'],
                    f['home_goals'], f['away_goals'],
                    f['ht_home_goals'], f['ht_away_goals'],
                    f['status'], f['league_type']
                ))

            total_fixtures += len(fixtures)
            print(f"  {league} {season}/{season+1}: {len(fixtures)} matches")

    conn.commit()
    conn.close()

    print(f"\n[OK] Total fixtures generated: {total_fixtures}")
    print(f"[OK] Database saved: {API_FOOTBALL_DB}")

    return total_fixtures


def generate_upcoming_fixtures_file(leagues: list = None, output_path: Path = None) -> Path:
    """
    Generate upcoming fixtures CSV file for predictions

    Args:
        leagues: List of league codes
        output_path: Where to save the CSV

    Returns:
        Path to generated CSV
    """
    if leagues is None:
        leagues = ['E0', 'E1', 'D1', 'SP1', 'I1', 'F1']

    if output_path is None:
        from config import OUTPUT_DIR
        output_path = OUTPUT_DIR / "upcoming_fixtures.csv"

    print("="*60)
    print("GENERATING UPCOMING FIXTURES")
    print("="*60)

    all_fixtures = []

    for league in leagues:
        teams = LEAGUE_TEAMS.get(league, [])
        if teams:
            fixtures = generate_upcoming_fixtures(league, teams)
            all_fixtures.extend(fixtures)
            print(f"  {league}: {len(fixtures)} upcoming matches")

    if not all_fixtures:
        print("[WARN] No fixtures generated")
        return None

    df = pd.DataFrame(all_fixtures)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Date', 'League']).reset_index(drop=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n[OK] Generated {len(df)} fixtures")
    print(f"[OK] Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    print("SAMPLE DATA GENERATOR")
    print("="*60)

    # Generate historical data
    initialize_database_with_sample_data(
        leagues=['E0', 'E1', 'D1', 'SP1', 'I1', 'F1', 'N1', 'T1'],
        seasons=[2023, 2024]
    )

    print()

    # Generate upcoming fixtures
    generate_upcoming_fixtures_file(
        leagues=['E0', 'E1', 'D1', 'SP1', 'I1', 'F1']
    )
