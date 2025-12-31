# update_results.py
"""
Update Accuracy Database with Actual Results
Run this weekly AFTER matches complete to track accuracy
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from accuracy_tracker import AccuracyTracker

def fetch_latest_results(data_dir: Path = None) -> pd.DataFrame:
    """
    Fetch latest results from downloaded CSVs
    """
    print("📥 Fetching latest results from downloaded data...")
    
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
                print(f"   ✅ Found data in: {data_dir}")
                break
        
        if data_dir is None:
            raise FileNotFoundError("No CSV files found in data/raw, downloaded_data, or data folders")
    
    # Find most recent CSV files
    csv_files = sorted(data_dir.glob("*.csv"), 
                      key=lambda x: x.stat().st_mtime, 
                      reverse=True)
    
    # Rest of function stays the same...
    
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
                print(f"   • {csv_file.name}: {len(recent)} recent matches")
                
        except Exception as e:
            print(f"   ⚠️ Skipping {csv_file.name}: {e}")
    
    if not all_data:
        raise ValueError("No recent match data found")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Standardize columns
    if 'Div' in combined.columns and 'League' not in combined.columns:
        combined['League'] = combined['Div']
    
    print(f"✅ Found {len(combined)} recent matches")
    return combined


def prepare_results_for_update(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw results to format needed for accuracy tracker
    
    Creates target columns: y_1X2, y_BTTS, y_OU_2_5, etc.
    """
    df = results_df.copy()
    
    # 1X2 Market
    if 'FTR' in df.columns:
        df['y_1X2'] = df['FTR']
    
    # BTTS Market
    if 'FTHG' in df.columns and 'FTAG' in df.columns:
        df['y_BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).map({True: 'Y', False: 'N'})
    
    # Over/Under 2.5
    if 'FTHG' in df.columns and 'FTAG' in df.columns:
        total_goals = df['FTHG'] + df['FTAG']
        df['y_OU_2_5'] = (total_goals > 2.5).map({True: 'O', False: 'U'})
    
    # Asian Handicap 0.0 (essentially 1X2 but different format)
    if 'FTR' in df.columns:
        df['y_AH_0_0'] = df['FTR'].map({'H': 'H', 'D': 'D', 'A': 'A'})
    
    # Keep only necessary columns
    keep_cols = ['Date', 'League', 'HomeTeam', 'AwayTeam', 
                 'y_1X2', 'y_BTTS', 'y_OU_2_5', 'y_AH_0_0']
    
    available_cols = [c for c in keep_cols if c in df.columns]
    
    return df[available_cols].dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])


def update_accuracy_database():
    """Main function to update accuracy database with results"""
    print("\n🔄 ACCURACY DATABASE UPDATE")
    print("="*60)
    
    try:
        # Fetch latest results
        results_df = fetch_latest_results()
        
        # Prepare for update
        prepared_df = prepare_results_for_update(results_df)
        print(f"📊 Prepared {len(prepared_df)} matches for update")
        
        # Update database
        tracker = AccuracyTracker()
        tracker.update_results(prepared_df)
        
        # Calculate accuracy for affected weeks
        print("\n📈 Calculating weekly accuracy...")
        unique_weeks = prepared_df['Date'].apply(lambda x: x.strftime('%Y-W%W')).unique()
        
        for week in unique_weeks:
            tracker.calculate_weekly_accuracy(week)
            print(f"   ✅ Week {week} updated")
        
        # Update market weights
        print("\n⚖️ Updating market weights...")
        weights_df = tracker.get_market_weights(lookback_weeks=12)
        
        if not weights_df.empty:
            print("\nCurrent Market Performance:")
            print(weights_df[['market', 'rolling_accuracy', 'rolling_roi', 'weight']].to_string(index=False))
        
        # Export report
        print("\n📊 Exporting accuracy report...")
        tracker.export_accuracy_report()
        
        print("\n" + "="*60)
        print("✅ ACCURACY DATABASE UPDATED SUCCESSFULLY")
        print("="*60)
        print("📂 Check outputs/accuracy_report.csv for details")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_recent_performance(weeks: int = 4):
    """Show recent performance summary"""
    print(f"\n📊 LAST {weeks} WEEKS PERFORMANCE")
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
        print(f"\n📅 Week {week}:")
        for _, row in week_data.iterrows():
            print(f"   • {row['market']}: {row['avg_accuracy']:.1%} accuracy " +
                  f"({int(row['correct'])}/{int(row['total'])}) " +
                  f"[{row['profit']:+.1f} units]")


def check_pending_predictions():
    """Check how many predictions are waiting for results"""
    print("\n⏳ PENDING PREDICTIONS")
    print("="*60)
    
    tracker = AccuracyTracker()
    
    import sqlite3
    conn = sqlite3.connect(tracker.db_path)
    
    cursor = conn.cursor()
    pending = cursor.execute("""
        SELECT COUNT(*) FROM predictions WHERE actual_outcome IS NULL
    """).fetchone()[0]
    
    print(f"📊 {pending} predictions awaiting results")
    
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
            print(f"   • {week}: {count} predictions")
    
    conn.close()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    print("🎯 ACCURACY TRACKING - RESULTS UPDATER")
    print("="*60)
    
    # Check pending predictions
    check_pending_predictions()
    
    # Ask user to proceed
    print("\n" + "="*60)
    proceed = input("Update database with latest results? (y/n): ").lower().strip()
    
    if proceed != 'y':
        print("❌ Update cancelled")
    else:
        # Update database
        success = update_accuracy_database()
        
        if success:
            # Show recent performance
            show_recent_performance(weeks=4)
            
            print("\n💡 TIP: Run this script weekly after matches complete")
            print("   It will keep your accuracy database up-to-date")
