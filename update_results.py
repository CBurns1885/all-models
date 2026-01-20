# update_results.py
"""
Update Accuracy Database with Actual Results
Run this weekly AFTER matches complete to track accuracy
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from accuracy_tracker import AccuracyTracker

def fetch_latest_results(data_dir: Path = None, days_back: int = 60) -> pd.DataFrame:
    """
    Fetch latest results from API-Football database

    Args:
        data_dir: Path to data directory (for CSV fallback)
        days_back: Number of days to look back for results (default: 60)
    """
    print("[FETCH] Fetching latest results from API-Football database...")

    try:
        # Try API-Football database first
        from api_football_adapter import get_fixtures_from_db, check_api_football_db

        if not check_api_football_db():
            print("   [WARN] API-Football database not found, trying CSV fallback...")
            return fetch_latest_results_from_csv(data_dir)

        # Get recent completed matches (configurable lookback)
        cutoff_date = datetime.now() - timedelta(days=days_back)

        # Get all finished matches
        df = get_fixtures_from_db(status='FT')

        if df.empty:
            raise ValueError("No finished matches found in database")

        # Filter to recent matches
        df['Date'] = pd.to_datetime(df['Date'])
        recent = df[df['Date'] >= cutoff_date].copy()

        print(f"   [OK] Found {len(recent)} finished matches (last {days_back} days)")
        print(f"   Date range: {recent['Date'].min()} to {recent['Date'].max()}")

        return recent

    except ImportError:
        print("   [WARN] API-Football adapter not available, trying CSV fallback...")
        return fetch_latest_results_from_csv(data_dir)
    except Exception as e:
        print(f"   [WARN] Database read failed ({e}), trying CSV fallback...")
        return fetch_latest_results_from_csv(data_dir)


def fetch_latest_results_from_csv(data_dir: Path = None) -> pd.DataFrame:
    """
    Fallback: Fetch latest results from downloaded CSVs
    """
    print("[FETCH] Fetching latest results from CSV files...")

    # Auto-detect data location
    if data_dir is None:
        possible_dirs = [
            Path("data/raw"),
            Path("downloaded_data"),
            Path("data"),
        ]
        for d in possible_dirs:
            if d.exists() and list(d.glob("*.csv")):
                data_dir = d
                print(f"   [OK] Found data in: {data_dir}")
                break

        if data_dir is None:
            raise FileNotFoundError("No CSV files found in data/raw, downloaded_data, or data folders")

    # Find most recent CSV files
    csv_files = sorted(data_dir.glob("*.csv"),
                      key=lambda x: x.stat().st_mtime,
                      reverse=True)

    # Load recent data (last 30 days)
    cutoff_date = datetime.now() - timedelta(days=30)
    all_data = []

    for csv_file in csv_files[:10]:  # Check last 10 files
        try:
            df = pd.read_csv(csv_file)

            # Ensure date column
            if 'Date' not in df.columns:
                continue

            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

            # Filter to recent matches
            recent = df[df['Date'] >= cutoff_date]

            if len(recent) > 0:
                all_data.append(recent)
                print(f"   * {csv_file.name}: {len(recent)} recent matches")

        except Exception as e:
            print(f"   [WARN] Skipping {csv_file.name}: {e}")

    if not all_data:
        raise ValueError("No recent match data found")

    combined = pd.concat(all_data, ignore_index=True)

    # Standardize columns
    if 'Div' in combined.columns and 'League' not in combined.columns:
        combined['League'] = combined['Div']

    print(f"[OK] Found {len(combined)} recent matches from CSV")
    return combined


def prepare_results_for_update(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw results to format needed for accuracy tracker

    Creates target columns: y_1X2, y_BTTS, y_OU_*, y_AH_*, etc.
    """
    df = results_df.copy()

    import numpy as np

    # Ensure we have goals data
    if 'FTHG' not in df.columns or 'FTAG' not in df.columns:
        print("[WARN] Missing goals data (FTHG/FTAG), limited market calculation")
        return df

    total_goals = df['FTHG'] + df['FTAG']
    goal_diff = df['FTHG'] - df['FTAG']
    btts = (df['FTHG'] > 0) & (df['FTAG'] > 0)

    # 1X2 Market
    if 'FTR' in df.columns:
        df['y_1X2'] = df['FTR']
    else:
        df['y_1X2'] = np.where(goal_diff > 0, 'H', np.where(goal_diff < 0, 'A', 'D'))

    # BTTS Market
    df['y_BTTS'] = btts.map({True: 'Y', False: 'N'})

    # Over/Under Markets
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        line_str = str(line).replace('.', '_')
        df[f'y_OU_{line_str}'] = (total_goals > line).map({True: 'O', False: 'U'})

    # Double Chance
    df['y_DC_1X'] = df['y_1X2'].isin(['H', 'D']).map({True: 'Y', False: 'N'})
    df['y_DC_12'] = df['y_1X2'].isin(['H', 'A']).map({True: 'Y', False: 'N'})
    df['y_DC_X2'] = df['y_1X2'].isin(['D', 'A']).map({True: 'Y', False: 'N'})

    # Draw No Bet
    df['y_DNB_H'] = np.where(df['y_1X2'] == 'H', 'Y', np.where(df['y_1X2'] == 'D', 'P', 'N'))
    df['y_DNB_A'] = np.where(df['y_1X2'] == 'A', 'Y', np.where(df['y_1X2'] == 'D', 'P', 'N'))

    # Team to Score
    df['y_HomeToScore'] = (df['FTHG'] > 0).map({True: 'Y', False: 'N'})
    df['y_AwayToScore'] = (df['FTAG'] > 0).map({True: 'Y', False: 'N'})

    # Team Goals O/U
    df['y_HomeTG_0_5'] = (df['FTHG'] > 0.5).map({True: 'O', False: 'U'})
    df['y_HomeTG_1_5'] = (df['FTHG'] > 1.5).map({True: 'O', False: 'U'})
    df['y_AwayTG_0_5'] = (df['FTAG'] > 0.5).map({True: 'O', False: 'U'})
    df['y_AwayTG_1_5'] = (df['FTAG'] > 1.5).map({True: 'O', False: 'U'})

    # Asian Handicap
    df['y_AH_0_0'] = np.where(goal_diff > 0, 'H', np.where(goal_diff < 0, 'A', 'P'))
    df['y_AH_-0_5'] = np.where(goal_diff > 0.5, 'H', np.where(goal_diff < -0.5, 'A', 'P'))
    df['y_AH_+0_5'] = np.where(goal_diff > -0.5, 'H', np.where(goal_diff < 0.5, 'A', 'P'))
    df['y_AH_-1_0'] = np.where(goal_diff > 1, 'H', np.where(goal_diff < -1, 'A', 'P'))
    df['y_AH_+1_0'] = np.where(goal_diff > -1, 'H', np.where(goal_diff < 1, 'A', 'P'))

    # Result + BTTS Combos
    df['y_HomeWin_BTTS_Y'] = ((df['y_1X2'] == 'H') & btts).map({True: 'Y', False: 'N'})
    df['y_HomeWin_BTTS_N'] = ((df['y_1X2'] == 'H') & ~btts).map({True: 'Y', False: 'N'})
    df['y_AwayWin_BTTS_Y'] = ((df['y_1X2'] == 'A') & btts).map({True: 'Y', False: 'N'})
    df['y_AwayWin_BTTS_N'] = ((df['y_1X2'] == 'A') & ~btts).map({True: 'Y', False: 'N'})

    # DC + O/U Combos
    df['y_DC1X_O25'] = ((df['y_1X2'].isin(['H', 'D'])) & (total_goals > 2.5)).map({True: 'Y', False: 'N'})
    df['y_DCX2_O25'] = ((df['y_1X2'].isin(['D', 'A'])) & (total_goals > 2.5)).map({True: 'Y', False: 'N'})

    # Keep only necessary columns (base + all y_* columns)
    base_cols = ['Date', 'League', 'HomeTeam', 'AwayTeam']
    y_cols = [c for c in df.columns if c.startswith('y_')]

    return df[base_cols + y_cols].dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])


def update_accuracy_database():
    """Main function to update accuracy database with results"""
    print("\n[UPDATE] ACCURACY DATABASE UPDATE")
    print("="*60)
    
    try:
        # Fetch latest results
        results_df = fetch_latest_results()
        
        # Prepare for update
        prepared_df = prepare_results_for_update(results_df)
        print(f"[DATA] Prepared {len(prepared_df)} matches for update")
        
        # Update database
        tracker = AccuracyTracker()
        tracker.update_results(prepared_df)
        
        # Calculate accuracy for affected weeks
        print("\n[STATS] Calculating weekly accuracy...")
        unique_weeks = prepared_df['Date'].apply(lambda x: x.strftime('%Y-W%W')).unique()
        
        for week in unique_weeks:
            tracker.calculate_weekly_accuracy(week)
            print(f"   [OK] Week {week} updated")
        
        # Update market weights
        print("\n[WEIGHTS] Updating market weights...")
        weights_df = tracker.get_market_weights(lookback_weeks=12)
        
        if not weights_df.empty:
            print("\nCurrent Market Performance:")
            print(weights_df[['market', 'rolling_accuracy', 'rolling_roi', 'weight']].to_string(index=False))
        
        # Export report
        print("\n[DATA] Exporting accuracy report...")
        tracker.export_accuracy_report()
        
        print("\n" + "="*60)
        print("[OK] ACCURACY DATABASE UPDATED SUCCESSFULLY")
        print("="*60)
        print("[OUTPUT] Check outputs/accuracy_report.csv for details")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_recent_performance(weeks: int = 4):
    """Show recent performance summary"""
    print(f"\n[DATA] LAST {weeks} WEEKS PERFORMANCE")
    print("="*60)
    
    tracker = AccuracyTracker()
    
    import sqlite3
    conn = sqlite3.connect(tracker.db_path)
    
    # Get recent weeks
    query = """
        SELECT 
            week_id,
            market,
            SUM(total_predictions) as total,
            SUM(correct_predictions) as correct,
            AVG(accuracy) as avg_accuracy,
            SUM(profit_loss) as profit
        FROM weekly_accuracy
        WHERE week_id >= date('now', '-' || ? || ' days')
        GROUP BY week_id, market
        ORDER BY week_id DESC, avg_accuracy DESC
    """
    
    df = pd.read_sql_query(query, conn, params=(weeks * 7,))
    conn.close()
    
    if df.empty:
        print("No data for this period")
        return
    
    # Group by week
    for week, week_data in df.groupby('week_id'):
        print(f"\n[DATE] Week {week}:")
        for _, row in week_data.iterrows():
            print(f"   * {row['market']}: {row['avg_accuracy']:.1%} accuracy " +
                  f"({int(row['correct'])}/{int(row['total'])}) " +
                  f"[{row['profit']:+.1f} units]")


def check_pending_predictions():
    """Check how many predictions are waiting for results"""
    print("\n[PENDING] PENDING PREDICTIONS")
    print("="*60)
    
    tracker = AccuracyTracker()
    
    import sqlite3
    conn = sqlite3.connect(tracker.db_path)
    
    cursor = conn.cursor()
    pending = cursor.execute("""
        SELECT COUNT(*) FROM predictions WHERE actual_outcome IS NULL
    """).fetchone()[0]
    
    print(f"[DATA] {pending} predictions awaiting results")
    
    if pending > 0:
        # Show breakdown by week
        breakdown = cursor.execute("""
            SELECT week_id, COUNT(*) as count
            FROM predictions 
            WHERE actual_outcome IS NULL
            GROUP BY week_id
            ORDER BY week_id DESC
        """).fetchall()
        
        print("\nBreakdown by week:")
        for week, count in breakdown:
            print(f"   * {week}: {count} predictions")
    
    conn.close()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update accuracy database with results")
    parser.add_argument('--auto', '-y', action='store_true', help='Run without confirmation prompt')
    args = parser.parse_args()

    print("[TRACKER] ACCURACY TRACKING - RESULTS UPDATER")
    print("="*60)

    # Check pending predictions
    check_pending_predictions()

    # Ask user to proceed (unless --auto flag)
    proceed = True
    if not args.auto:
        print("\n" + "="*60)
        try:
            user_input = input("Update database with latest results? (y/n): ").lower().strip()
            proceed = (user_input == 'y')
        except EOFError:
            # Non-interactive mode - proceed automatically
            proceed = True

    if not proceed:
        print("[INFO] Update cancelled")
    else:
        # Update database
        success = update_accuracy_database()

        if success:
            # Show recent performance
            show_recent_performance(weeks=4)

            print("\n[TIP] Run this script weekly after matches complete")
            print("   It will keep your accuracy database up-to-date")
