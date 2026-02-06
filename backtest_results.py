# backtest_results.py
"""
Add Win/Lose results to high confidence bets by matching to historical results.
Combines predictions from dated output folders and adds Result column (Win/Lose).
Uses API-Football database for up-to-date results.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths - uses same structure as config.py
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"  # Shared data folder
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUT_CSV = OUTPUTS_DIR / "high_confidence_with_results.csv"


def normalize_team_name(name: str) -> str:
    """Normalize team name for fuzzy matching"""
    if pd.isna(name):
        return ""

    name = str(name).strip().lower()

    # Remove common prefixes/suffixes
    prefixes = ['fc ', 'cf ', 'sc ', 'ac ', 'as ', 'sl ', 'ss ', 'us ', 'afc ', 'rcd ', 'cd ', 'ud ', 'sd ', 'rc ', 'real ', 'sporting ']
    suffixes = [' fc', ' cf', ' sc', ' ac', ' afc', ' united', ' city', ' town', ' athletic', ' wanderers', ' rovers', ' hotspur']

    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]

    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    # Known team name mappings (prediction name -> API name variations)
    mappings = {
        'man utd': 'manchester united',
        'man united': 'manchester united',
        'man city': 'manchester city',
        'spurs': 'tottenham',
        'wolves': 'wolverhampton',
        'nott\'m forest': 'nottingham forest',
        'nottm forest': 'nottingham forest',
        'west ham': 'west ham',
        'newcastle': 'newcastle',
        'brighton': 'brighton',
        'bournemouth': 'bournemouth',
        'crystal palace': 'crystal palace',
        'hearts': 'heart of midlothian',
        'heart of midlothian': 'hearts',
        'hibs': 'hibernian',
        'benfica': 'benfica',
        'porto': 'porto',
        'sporting': 'sporting',
        'braga': 'braga',
        'psg': 'paris saint germain',
        'paris sg': 'paris saint germain',
        'marseille': 'marseille',
        'lyon': 'lyon',
        'bayern': 'bayern munich',
        'dortmund': 'borussia dortmund',
        'leverkusen': 'bayer leverkusen',
        'atletico': 'atletico madrid',
        'atletico madrid': 'atletico',
        'real madrid': 'real madrid',
        'barcelona': 'barcelona',
        'inter': 'inter milan',
        'inter milan': 'inter',
        'ac milan': 'milan',
        'milan': 'ac milan',
        'juventus': 'juventus',
        'napoli': 'napoli',
        'roma': 'roma',
        'lazio': 'lazio',
    }

    # Check if name matches any mapping
    if name in mappings:
        name = mappings[name]

    # Remove extra whitespace
    name = ' '.join(name.split())

    return name


def load_results_from_api_db() -> pd.DataFrame:
    """Load finished match results from API-Football database"""
    try:
        from api_football_adapter import get_fixtures_from_db
        df = get_fixtures_from_db(status='FT')
        return df
    except ImportError:
        print("  Warning: api_football_adapter not available")
        return pd.DataFrame()
    except Exception as e:
        print(f"  Warning: Could not load from API DB: {e}")
        return pd.DataFrame()


def combine_dated_predictions(min_confidence: float = 90.0) -> pd.DataFrame:
    """
    Combine prediction files from all dated output folders.
    Filters for 90%+ confidence and returns master DataFrame.
    """
    all_predictions = []

    # Find all dated output folders (YYYY-MM-DD format)
    output_folders = sorted(OUTPUTS_DIR.glob("20*-*-*"))
    print(f"  Found {len(output_folders)} dated output folders")

    for folder in output_folders:
        if not folder.is_dir():
            continue

        # Load each prediction type from this date
        prediction_files = [
            ('predictions_1x2.csv', '1X2'),
            ('predictions_btts.csv', 'BTTS'),
            ('predictions_ou_0_5.csv', 'OU_0_5'),
            ('predictions_ou_1_5.csv', 'OU_1_5'),
            ('predictions_ou_2_5.csv', 'OU_2_5'),
            ('predictions_ou_3_5.csv', 'OU_3_5'),
            ('predictions_ou_4_5.csv', 'OU_4_5'),
            ('predictions_ou_5_5.csv', 'OU_5_5'),
        ]

        for filename, market in prediction_files:
            filepath = folder / filename
            if not filepath.exists():
                continue

            try:
                df = pd.read_csv(filepath)
                if df.empty:
                    continue

                # Get confidence column
                conf_col = None
                for col in ['Confidence_%', 'Confidence', 'Confidence_Pct']:
                    if col in df.columns:
                        conf_col = col
                        break

                if conf_col is None:
                    continue

                # Filter for high confidence
                high_conf = df[df[conf_col] >= min_confidence].copy()

                if high_conf.empty:
                    continue

                # Standardize columns
                high_conf['Market'] = market
                high_conf['Confidence'] = high_conf[conf_col]

                # Normalize prediction values
                if 'Prediction' in high_conf.columns:
                    # Map short codes to readable format
                    pred_map = {
                        'H': 'H', 'D': 'D', 'A': 'A',
                        'O': 'Over', 'U': 'Under',
                        'Y': 'Yes', 'N': 'No'
                    }
                    high_conf['Prediction'] = high_conf['Prediction'].map(lambda x: pred_map.get(x, x))

                all_predictions.append(high_conf[['Date', 'League', 'HomeTeam', 'AwayTeam', 'Market', 'Prediction', 'Confidence']])

            except Exception as e:
                print(f"  Warning: Could not read {filepath}: {e}")
                continue

    if not all_predictions:
        return pd.DataFrame()

    combined = pd.concat(all_predictions, ignore_index=True)
    combined = combined.drop_duplicates()
    return combined


def get_actual_result(row, hist_df, debug=False):
    """
    Match a prediction to historical result and determine Win/Lose.

    Returns: 'Win', 'Lose', or 'Pending' (no result found)
    """
    # Find matching historical match - exact match first
    match = hist_df[
        (hist_df['Date'] == row['Date']) &
        (hist_df['League'] == row['League']) &
        (hist_df['HomeTeam'] == row['HomeTeam']) &
        (hist_df['AwayTeam'] == row['AwayTeam'])
    ]

    # If no exact match, try case-insensitive team name match
    if len(match) == 0:
        match = hist_df[
            (hist_df['Date'] == row['Date']) &
            (hist_df['League'] == row['League']) &
            (hist_df['HomeTeam'].str.lower() == str(row['HomeTeam']).lower()) &
            (hist_df['AwayTeam'].str.lower() == str(row['AwayTeam']).lower())
        ]

    # Try normalized team name matching
    if len(match) == 0:
        pred_home_norm = normalize_team_name(row['HomeTeam'])
        pred_away_norm = normalize_team_name(row['AwayTeam'])

        match = hist_df[
            (hist_df['Date'] == row['Date']) &
            (hist_df['League'] == row['League']) &
            (hist_df['HomeTeam_norm'] == pred_home_norm) &
            (hist_df['AwayTeam_norm'] == pred_away_norm)
        ]

    # Try partial/contains matching as last resort
    if len(match) == 0:
        same_date_league = hist_df[
            (hist_df['Date'] == row['Date']) &
            (hist_df['League'] == row['League'])
        ]
        if len(same_date_league) > 0:
            # Try to find a match where normalized names contain each other
            pred_home_norm = normalize_team_name(row['HomeTeam'])
            pred_away_norm = normalize_team_name(row['AwayTeam'])

            for idx, m in same_date_league.iterrows():
                hist_home_norm = m['HomeTeam_norm']
                hist_away_norm = m['AwayTeam_norm']

                # Check if names match (either direction contains)
                home_match = (pred_home_norm in hist_home_norm) or (hist_home_norm in pred_home_norm) or (pred_home_norm == hist_home_norm)
                away_match = (pred_away_norm in hist_away_norm) or (hist_away_norm in pred_away_norm) or (pred_away_norm == hist_away_norm)

                if home_match and away_match:
                    match = same_date_league.loc[[idx]]
                    break

    # Debug output
    if len(match) == 0 and debug:
        same_date_league = hist_df[
            (hist_df['Date'] == row['Date']) &
            (hist_df['League'] == row['League'])
        ]
        if len(same_date_league) > 0:
            print(f"  DEBUG: {row['Date']} {row['League']} - Looking for '{row['HomeTeam']}' v '{row['AwayTeam']}'")
            print(f"         Normalized: '{normalize_team_name(row['HomeTeam'])}' v '{normalize_team_name(row['AwayTeam'])}'")
            print(f"         Available: {same_date_league[['HomeTeam', 'AwayTeam']].values.tolist()[:3]}")

    if len(match) == 0:
        return 'Pending'

    match = match.iloc[0]
    home_goals = match['FTHG']
    away_goals = match['FTAG']
    total_goals = home_goals + away_goals

    market = row['Market']
    prediction = str(row['Prediction']).strip()

    # 1X2 Market
    if market == '1X2':
        actual = match['FTR']  # H, D, or A
        pred_norm = prediction.upper()
        if pred_norm in ['HOME', 'HOME WIN']:
            pred_norm = 'H'
        elif pred_norm in ['AWAY', 'AWAY WIN']:
            pred_norm = 'A'
        elif pred_norm == 'DRAW':
            pred_norm = 'D'
        return 'Win' if pred_norm == actual else 'Lose'

    # BTTS Market
    if market == 'BTTS':
        btts_actual = (home_goals > 0 and away_goals > 0)
        pred_norm = prediction.upper()
        pred_yes = pred_norm in ['Y', 'YES', 'YES (BOTH SCORE)']
        return 'Win' if pred_yes == btts_actual else 'Lose'

    # Over/Under Markets (OU_0_5, OU_1_5, OU_2_5, OU_3_5, OU_4_5, OU_5_5)
    if market.startswith('OU_'):
        parts = market.split('_')
        line = float(f"{parts[1]}.{parts[2]}")

        actual_over = total_goals > line
        pred_norm = prediction.upper()
        pred_over = pred_norm in ['O', 'OVER', f'OVER {line}']

        return 'Win' if pred_over == actual_over else 'Lose'

    return 'Unknown'


def main():
    """Add Win/Lose results to high confidence bets"""
    print("=" * 60)
    print("BACKTEST RESULTS - High Confidence Bets (90%+)")
    print("=" * 60)

    # Load results from API-Football database
    print(f"\nLoading results from API-Football database...")
    hist_df = load_results_from_api_db()

    if hist_df.empty:
        print("Error: No results loaded from API-Football database")
        return

    hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.date
    print(f"  Loaded {len(hist_df)} finished matches")
    print(f"  Date range: {hist_df['Date'].min()} to {hist_df['Date'].max()}")

    # Add normalized team names for fuzzy matching
    hist_df['HomeTeam_norm'] = hist_df['HomeTeam'].apply(normalize_team_name)
    hist_df['AwayTeam_norm'] = hist_df['AwayTeam'].apply(normalize_team_name)

    # Combine predictions from dated folders
    print(f"\nCombining predictions from dated output folders...")
    hc_df = combine_dated_predictions(min_confidence=90.0)

    if hc_df.empty:
        print("Error: No high confidence predictions found")
        return

    print(f"  Combined {len(hc_df)} high confidence predictions")

    hc_df['Date'] = pd.to_datetime(hc_df['Date']).dt.date
    print(f"  Prediction dates: {hc_df['Date'].min()} to {hc_df['Date'].max()}")

    # Check overlap
    hist_max = hist_df['Date'].max()
    hist_min = hist_df['Date'].min()
    pred_min = hc_df['Date'].min()
    pred_max = hc_df['Date'].max()

    # Filter predictions to only those within historical data range
    in_range = hc_df[(hc_df['Date'] >= hist_min) & (hc_df['Date'] <= hist_max)]
    print(f"\n  Predictions within database range: {len(in_range)} of {len(hc_df)}")

    if pred_min > hist_max:
        print(f"\n  WARNING: Predictions start ({pred_min}) after results end ({hist_max})")
        print(f"  Run update_results.py to fetch latest match results")

    # Add results
    print("\nMatching predictions to historical results...")

    # Debug: show sample of what we're trying to match
    print("\n  Sample prediction:")
    sample = hc_df.iloc[0]
    print(f"    Date: {sample['Date']} | League: {sample['League']}")
    print(f"    Home: {sample['HomeTeam']} | Away: {sample['AwayTeam']}")

    # Check if this exact match exists in hist_df
    sample_match = hist_df[
        (hist_df['Date'] == sample['Date']) &
        (hist_df['League'] == sample['League'])
    ]
    if len(sample_match) > 0:
        print(f"  Found {len(sample_match)} matches on same date/league:")
        for _, m in sample_match.head(3).iterrows():
            print(f"    {m['HomeTeam']} v {m['AwayTeam']}")
    else:
        print(f"  No matches found for {sample['Date']} in {sample['League']}")
        # Check what dates exist for this league
        league_dates = hist_df[hist_df['League'] == sample['League']]['Date'].unique()
        if len(league_dates) > 0:
            print(f"  Available dates for {sample['League']}: {sorted(league_dates)[-5:]}")

    # Run matching with debug for first 5
    results = []
    for i, (idx, row) in enumerate(hc_df.iterrows()):
        result = get_actual_result(row, hist_df, debug=(i < 5))
        results.append(result)
    hc_df['Result'] = results

    # Summary stats
    total = len(hc_df)
    wins = (hc_df['Result'] == 'Win').sum()
    losses = (hc_df['Result'] == 'Lose').sum()
    pending = (hc_df['Result'] == 'Pending').sum()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Bets:     {total}")
    print(f"Wins:           {wins}")
    print(f"Losses:         {losses}")
    print(f"Pending:        {pending}")

    if wins + losses > 0:
        win_rate = wins / (wins + losses) * 100
        print(f"\nWin Rate:       {win_rate:.1f}%")

    # Breakdown by market
    print("\n" + "-" * 60)
    print("BREAKDOWN BY MARKET")
    print("-" * 60)

    for market in sorted(hc_df['Market'].unique()):
        market_df = hc_df[hc_df['Market'] == market]
        m_wins = (market_df['Result'] == 'Win').sum()
        m_losses = (market_df['Result'] == 'Lose').sum()
        m_total = m_wins + m_losses

        if m_total > 0:
            m_rate = m_wins / m_total * 100
            print(f"{market:12} - Win Rate: {m_rate:5.1f}% ({m_wins}/{m_total})")

    # Breakdown by confidence tier
    print("\n" + "-" * 60)
    print("BREAKDOWN BY CONFIDENCE")
    print("-" * 60)

    tiers = [
        (90, 92, '90-92%'),
        (92, 95, '92-95%'),
        (95, 98, '95-98%'),
        (98, 100.1, '98%+'),
    ]

    for min_conf, max_conf, label in tiers:
        tier_df = hc_df[(hc_df['Confidence'] >= min_conf) & (hc_df['Confidence'] < max_conf)]
        t_wins = (tier_df['Result'] == 'Win').sum()
        t_losses = (tier_df['Result'] == 'Lose').sum()
        t_total = t_wins + t_losses

        if t_total > 0:
            t_rate = t_wins / t_total * 100
            print(f"{label:12} - Win Rate: {t_rate:5.1f}% ({t_wins}/{t_total})")

    # Save results
    hc_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved results to {OUTPUT_CSV}")

    # Show recent results
    print("\n" + "-" * 60)
    print("RECENT BETS (Last 20)")
    print("-" * 60)

    recent = hc_df.sort_values('Date', ascending=False).head(20)
    for _, row in recent.iterrows():
        result_marker = '[W]' if row['Result'] == 'Win' else '[L]' if row['Result'] == 'Lose' else '[?]'
        print(f"{result_marker} {row['Date']} {row['League']:4} {str(row['HomeTeam'])[:15]:15} v {str(row['AwayTeam'])[:15]:15} | {row['Market']:8} {str(row['Prediction']):6} ({row['Confidence']:.1f}%)")


def compare_team_names():
    """Compare team names between predictions and API database to identify mismatches"""
    print("=" * 60)
    print("TEAM NAME COMPARISON - Predictions vs API Database")
    print("=" * 60)

    # Load results from API-Football database
    print(f"\nLoading API-Football database...")
    hist_df = load_results_from_api_db()

    if hist_df.empty:
        print("Error: No results loaded from API-Football database")
        return

    hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.date
    print(f"  Loaded {len(hist_df)} finished matches")
    print(f"  Date range: {hist_df['Date'].min()} to {hist_df['Date'].max()}")

    # Load predictions
    print(f"\nLoading predictions...")
    pred_df = combine_dated_predictions(min_confidence=90.0)
    if pred_df.empty:
        print("Error: No predictions found")
        return

    pred_df['Date'] = pd.to_datetime(pred_df['Date']).dt.date
    print(f"  Loaded {len(pred_df)} high confidence predictions")

    # Get unique leagues from both
    pred_leagues = set(pred_df['League'].unique())
    hist_leagues = set(hist_df['League'].unique())

    print(f"\n" + "-" * 60)
    print("LEAGUE COVERAGE")
    print("-" * 60)
    print(f"Prediction leagues: {sorted(pred_leagues)}")
    print(f"API DB leagues: {sorted(hist_leagues)}")
    print(f"Matching leagues: {sorted(pred_leagues & hist_leagues)}")
    print(f"Missing from API: {sorted(pred_leagues - hist_leagues)}")

    # For each league, show date coverage
    print(f"\n" + "-" * 60)
    print("DATE COVERAGE BY LEAGUE")
    print("-" * 60)

    for league in sorted(pred_leagues & hist_leagues):
        pred_dates = pred_df[pred_df['League'] == league]['Date']
        hist_dates = hist_df[hist_df['League'] == league]['Date']

        pred_min, pred_max = pred_dates.min(), pred_dates.max()
        hist_min, hist_max = hist_dates.min(), hist_dates.max()

        overlap = pred_min <= hist_max
        status = "OK" if overlap else "NO OVERLAP"

        print(f"{league:6} | Predictions: {pred_min} to {pred_max}")
        print(f"       | API DB:      {hist_min} to {hist_max} [{status}]")

    # Now compare team names for overlapping dates
    print(f"\n" + "-" * 60)
    print("TEAM NAME COMPARISON (for overlapping date/league)")
    print("-" * 60)

    comparison_data = []

    for league in sorted(pred_leagues & hist_leagues):
        # Get date overlap
        pred_subset = pred_df[pred_df['League'] == league]
        hist_subset = hist_df[hist_df['League'] == league]

        hist_max = hist_subset['Date'].max()

        # Get predictions within API date range
        overlap_preds = pred_subset[pred_subset['Date'] <= hist_max]

        if len(overlap_preds) == 0:
            print(f"\n{league}: No overlapping dates")
            continue

        # Get unique teams from each
        pred_home = set(overlap_preds['HomeTeam'].unique())
        pred_away = set(overlap_preds['AwayTeam'].unique())
        pred_teams = pred_home | pred_away

        hist_home = set(hist_subset['HomeTeam'].unique())
        hist_away = set(hist_subset['AwayTeam'].unique())
        hist_teams = hist_home | hist_away

        print(f"\n{league}:")
        print(f"  Predictions have {len(pred_teams)} teams, API has {len(hist_teams)} teams")

        # Find teams in predictions not in API (potential mismatches)
        missing = pred_teams - hist_teams
        if missing:
            print(f"  Teams in predictions NOT in API DB:")
            for team in sorted(missing):
                # Try to find closest match in API
                team_norm = normalize_team_name(team)
                possible_matches = [t for t in hist_teams if normalize_team_name(t) == team_norm]
                if possible_matches:
                    print(f"    '{team}' -> possible match: {possible_matches}")
                else:
                    # Try partial match
                    partial = [t for t in hist_teams if team_norm in normalize_team_name(t) or normalize_team_name(t) in team_norm]
                    if partial:
                        print(f"    '{team}' -> partial matches: {partial[:3]}")
                    else:
                        print(f"    '{team}' -> NO MATCH FOUND")

                comparison_data.append({
                    'League': league,
                    'Prediction_Team': team,
                    'Normalized': team_norm,
                    'API_Match': possible_matches[0] if possible_matches else '',
                    'Partial_Match': partial[0] if partial and not possible_matches else ''
                })

    # Save comparison to CSV
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_csv = OUTPUTS_DIR / "team_name_comparison.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"\nSaved team name comparison to {comparison_csv}")

    # Also check specific predictions that should match but don't
    print(f"\n" + "-" * 60)
    print("SAMPLE UNMATCHED PREDICTIONS")
    print("-" * 60)

    hist_max = hist_df['Date'].max()
    in_range_preds = pred_df[pred_df['Date'] <= hist_max]

    unmatched = []
    for idx, row in in_range_preds.head(100).iterrows():
        # Try to find match
        match = hist_df[
            (hist_df['Date'] == row['Date']) &
            (hist_df['League'] == row['League']) &
            (hist_df['HomeTeam'] == row['HomeTeam']) &
            (hist_df['AwayTeam'] == row['AwayTeam'])
        ]

        if len(match) == 0:
            # What matches DO exist for this date/league?
            same_day = hist_df[(hist_df['Date'] == row['Date']) & (hist_df['League'] == row['League'])]
            unmatched.append({
                'date': row['Date'],
                'league': row['League'],
                'pred_home': row['HomeTeam'],
                'pred_away': row['AwayTeam'],
                'api_matches': same_day[['HomeTeam', 'AwayTeam']].values.tolist() if len(same_day) > 0 else []
            })

    print(f"Found {len(unmatched)} unmatched predictions out of first 100 in range")
    for um in unmatched[:10]:
        print(f"\n  {um['date']} {um['league']}: {um['pred_home']} v {um['pred_away']}")
        if um['api_matches']:
            print(f"    API has: {um['api_matches'][:3]}")
        else:
            print(f"    API has NO matches for this date/league")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        compare_team_names()
    else:
        main()
