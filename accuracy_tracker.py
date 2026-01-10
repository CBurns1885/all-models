# accuracy_tracker.py
"""
Live Accuracy Tracking System
1. Log predictions when made
2. Update with actual results after matches
3. Calculate rolling accuracy by market/league
4. Weight future predictions by historical accuracy
"""

import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

class AccuracyTracker:
    """Track and analyze prediction accuracy over time"""
    
    def __init__(self, db_path: Path = Path("outputs/accuracy_database.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                week_id TEXT NOT NULL,
                prediction_date DATE NOT NULL,
                match_date DATE NOT NULL,
                league TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                market TEXT NOT NULL,
                predicted_outcome TEXT NOT NULL,
                predicted_probability REAL NOT NULL,
                actual_outcome TEXT,
                correct INTEGER,
                logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Weekly accuracy summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weekly_accuracy (
                week_id TEXT NOT NULL,
                league TEXT NOT NULL,
                market TEXT NOT NULL,
                total_predictions INTEGER NOT NULL,
                correct_predictions INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                avg_probability REAL NOT NULL,
                brier_score REAL,
                profit_loss REAL,
                PRIMARY KEY (week_id, league, market)
            )
        """)
        
        # Market weights table (for weighted output)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_weights (
                market TEXT PRIMARY KEY,
                league TEXT,
                rolling_accuracy REAL NOT NULL,
                rolling_roi REAL NOT NULL,
                total_predictions INTEGER NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                weight REAL NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
        
        print(f"[OK] Database initialized: {self.db_path}")
    
    def log_predictions(self, predictions_df: pd.DataFrame, week_id: str):
        """
        Log predictions for a week
        
        Args:
            predictions_df: DataFrame with columns: Date, League, HomeTeam, AwayTeam,
                           P_y_1X2_H, P_y_1X2_D, P_y_1X2_A, etc.
            week_id: Unique identifier for this week (e.g., '2025-W40')
        """
        conn = sqlite3.connect(self.db_path)
        
        prediction_date = datetime.now().date()
        records = []
        
        # Extract predictions for each market
        for idx, row in predictions_df.iterrows():
            match_date = pd.to_datetime(row['Date']).date()
            league = row['League']
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Find all probability columns
            prob_cols = [c for c in predictions_df.columns if c.startswith('P_')]
            
            for prob_col in prob_cols:
                # Parse market and outcome from column name
                # e.g., P_y_1X2_H -> market=y_1X2, outcome=H
                parts = prob_col.replace('P_', '').split('_')
                if len(parts) < 2:
                    continue
                
                # Reconstruct market name
                market = '_'.join(parts[:-1])
                outcome = parts[-1]
                probability = row[prob_col]
                
                if pd.notna(probability) and probability > 0:
                    records.append({
                        'week_id': week_id,
                        'prediction_date': prediction_date,
                        'match_date': match_date,
                        'league': league,
                        'home_team': home_team,
                        'away_team': away_team,
                        'market': market,
                        'predicted_outcome': outcome,
                        'predicted_probability': float(probability),
                        'actual_outcome': None,
                        'correct': None
                    })
        
        # Insert into database
        if records:
            pd.DataFrame(records).to_sql('predictions', conn, if_exists='append', index=False)
            print(f"[OK] Logged {len(records)} predictions for week {week_id}")
        
        conn.close()
    
    def update_results(self, results_df: pd.DataFrame):
        """
        Update predictions with actual results
        
        Args:
            results_df: DataFrame with columns: Date, League, HomeTeam, AwayTeam,
                       y_1X2, y_BTTS, y_OU_2_5, etc. (actual outcomes)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updated_count = 0
        
        for idx, row in results_df.iterrows():
            match_date = pd.to_datetime(row['Date']).date()
            league = row['League']
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Find actual outcomes for each market
            outcome_cols = [c for c in results_df.columns if c.startswith('y_')]
            
            for outcome_col in outcome_cols:
                market = outcome_col  # e.g., y_1X2
                actual_outcome = row[outcome_col]
                
                if pd.notna(actual_outcome):
                    # Update matching predictions
                    cursor.execute("""
                        UPDATE predictions
                        SET actual_outcome = ?,
                            correct = CASE 
                                WHEN predicted_outcome = ? THEN 1 
                                ELSE 0 
                            END
                        WHERE match_date = ?
                        AND league = ?
                        AND home_team = ?
                        AND away_team = ?
                        AND market = ?
                        AND actual_outcome IS NULL
                    """, (actual_outcome, actual_outcome, match_date, league, 
                         home_team, away_team, market))
                    
                    updated_count += cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"[OK] Updated {updated_count} predictions with actual results")
    
    def calculate_weekly_accuracy(self, week_id: str):
        """Calculate accuracy metrics for a specific week"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                league,
                market,
                COUNT(*) as total,
                SUM(correct) as correct,
                AVG(predicted_probability) as avg_prob,
                AVG(CASE 
                    WHEN correct = 1 THEN POWER(predicted_probability - 1, 2)
                    ELSE POWER(predicted_probability, 2)
                END) as brier_score
            FROM predictions
            WHERE week_id = ? AND correct IS NOT NULL
            GROUP BY league, market
        """
        
        df = pd.read_sql_query(query, conn, params=(week_id,))
        
        if df.empty:
            conn.close()
            return
        
        # Calculate metrics
        df['accuracy'] = df['correct'] / df['total']
        df['profit_loss'] = df['correct'] - df['total']  # Simple profit calculation
        
        # Insert into weekly_accuracy table
        for _, row in df.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO weekly_accuracy 
                (week_id, league, market, total_predictions, correct_predictions,
                 accuracy, avg_probability, brier_score, profit_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (week_id, row['league'], row['market'], int(row['total']),
                 int(row['correct']), float(row['accuracy']), float(row['avg_prob']),
                 float(row['brier_score']), float(row['profit_loss'])))
        
        conn.commit()
        conn.close()
        
        print(f"[OK] Calculated accuracy metrics for week {week_id}")
    
    def get_market_weights(self, lookback_weeks: int = 12) -> pd.DataFrame:
        """
        Calculate current market weights based on rolling accuracy
        
        Args:
            lookback_weeks: Number of recent weeks to consider
        
        Returns:
            DataFrame with market weights
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get recent performance by market
        query = """
            SELECT 
                market,
                league,
                SUM(total_predictions) as total_preds,
                SUM(correct_predictions) as correct_preds,
                AVG(accuracy) as avg_accuracy,
                SUM(profit_loss) as total_profit
            FROM weekly_accuracy
            WHERE week_id >= date('now', '-' || ? || ' days')
            GROUP BY market, league
            HAVING total_preds >= 20
        """
        
        df = pd.read_sql_query(query, conn, params=(lookback_weeks * 7,))
        
        if df.empty:
            conn.close()
            return pd.DataFrame()
        
        # Calculate weights
        df['rolling_accuracy'] = df['correct_preds'] / df['total_preds']
        df['rolling_roi'] = (df['total_profit'] / df['total_preds']) * 100
        
        # Weight formula: accuracy above 50% baseline, boosted by ROI
        df['weight'] = ((df['rolling_accuracy'] - 0.50) * 2) + (df['rolling_roi'] / 100)
        df['weight'] = df['weight'].clip(lower=0.5, upper=2.0)  # Clamp between 0.5x and 2.0x
        
        # Update weights table
        for _, row in df.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO market_weights
                (market, league, rolling_accuracy, rolling_roi, total_predictions, weight)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (row['market'], row['league'], float(row['rolling_accuracy']),
                 float(row['rolling_roi']), int(row['total_preds']), float(row['weight'])))
        
        conn.commit()
        conn.close()
        
        print(f"[OK] Updated weights for {len(df)} market/league combinations")
        return df
    
    def get_market_rankings(self) -> pd.DataFrame:
        """Get current rankings of markets by performance"""
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query("""
            SELECT 
                market,
                rolling_accuracy,
                rolling_roi,
                total_predictions,
                weight,
                last_updated
            FROM market_weights
            ORDER BY weight DESC
        """, conn)
        
        conn.close()
        return df
    
    def export_accuracy_report(self, output_path: Path = Path("outputs/accuracy_report.csv")):
        """Export detailed accuracy report with confidence buckets"""
        conn = sqlite3.connect(self.db_path)

        # Get all predictions with results
        df = pd.read_sql_query("""
            SELECT
                league,
                market,
                predicted_probability,
                correct
            FROM predictions
            WHERE correct IS NOT NULL
        """, conn)

        conn.close()

        if df.empty:
            print("[WARN] No completed predictions to report")
            return pd.DataFrame()

        # Create confidence buckets (60%, 65%, 70%, ..., 100%)
        buckets = list(range(60, 105, 5))  # [60, 65, 70, 75, 80, 85, 90, 95, 100]
        bucket_labels = [f"{b}%-{b+4}%" for b in buckets[:-1]] + ["100%"]

        # Assign each prediction to a bucket
        df['confidence_bucket'] = pd.cut(
            df['predicted_probability'] * 100,
            bins=[0] + buckets,
            labels=['<60%'] + bucket_labels,
            include_lowest=True
        )

        # Group by league, market, and confidence bucket
        report = df.groupby(['league', 'market', 'confidence_bucket']).agg({
            'correct': ['sum', 'count']
        }).reset_index()

        report.columns = ['League', 'Market', 'Confidence', 'Correct', 'Total']
        report['Accuracy'] = (report['Correct'] / report['Total'] * 100).round(1)

        # Filter to only show buckets >= 60%
        report = report[report['Confidence'] != '<60%'].copy()

        # Sort by league, market, and confidence
        report = report.sort_values(['League', 'Market', 'Confidence'])

        # Save to CSV
        report.to_csv(output_path, index=False)
        print(f"[OK] Exported detailed accuracy report: {output_path}")
        print(f"    {len(report)} league/market/confidence combinations")

        return report


# ============================================================================
# INTEGRATION FUNCTIONS
# ============================================================================

def log_weekly_predictions(predictions_csv: Path, week_id: Optional[str] = None):
    """
    Log predictions from weekly_bets.csv
    Call this after predict_week() in RUN_WEEKLY.py
    """
    if week_id is None:
        week_id = datetime.now().strftime('%Y-W%W')
    
    tracker = AccuracyTracker()
    
    try:
        df = pd.read_csv(predictions_csv)
        tracker.log_predictions(df, week_id)
        return True
    except Exception as e:
        print(f"[WARN] Failed to log predictions: {e}")
        return False


def update_with_results(results_csv: Path):
    """
    Update predictions with actual results
    Call this weekly after matches complete
    """
    tracker = AccuracyTracker()
    
    try:
        df = pd.read_csv(results_csv)
        tracker.update_results(df)
        
        # Calculate accuracy for affected weeks
        weeks = df['Date'].apply(lambda x: pd.to_datetime(x).strftime('%Y-W%W')).unique()
        for week in weeks:
            tracker.calculate_weekly_accuracy(week)
        
        # Update market weights
        tracker.get_market_weights()
        
        return True
    except Exception as e:
        print(f"[WARN] Failed to update results: {e}")
        return False


def get_current_weights() -> Dict[str, float]:
    """Get current market weights for prediction weighting"""
    tracker = AccuracyTracker()
    weights_df = tracker.get_market_rankings()

    if weights_df.empty:
        return {}

    # Return as dictionary: market -> weight
    return dict(zip(weights_df['market'], weights_df['weight']))


def generate_weekly_accuracy_report(week_id: Optional[str] = None, output_dir: Path = None):
    """
    Generate a detailed weekly accuracy report showing each prediction's result.

    Args:
        week_id: Week to report on (default: last week)
        output_dir: Output directory for HTML report

    Outputs:
        - accuracy_report_<week>.html - Full HTML report
        - accuracy_report_<week>.csv - CSV data
    """
    if output_dir is None:
        output_dir = Path("outputs") / datetime.now().strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = AccuracyTracker()
    conn = sqlite3.connect(tracker.db_path)

    # Default to last week
    if week_id is None:
        last_week = datetime.now() - timedelta(days=7)
        week_id = last_week.strftime('%Y-W%W')

    print(f"\n📊 WEEKLY ACCURACY REPORT - {week_id}")
    print("=" * 60)

    # Get all predictions for the week with results
    query = """
        SELECT
            match_date,
            league,
            home_team,
            away_team,
            market,
            predicted_outcome,
            predicted_probability,
            actual_outcome,
            correct
        FROM predictions
        WHERE week_id = ?
        ORDER BY match_date, league, home_team
    """

    df = pd.read_sql_query(query, conn, params=(week_id,))

    if df.empty:
        print(f"No predictions found for week {week_id}")
        conn.close()
        return None

    # Calculate summary stats
    total = len(df)
    with_results = df[df['actual_outcome'].notna()]
    pending = len(df) - len(with_results)
    correct = with_results['correct'].sum() if len(with_results) > 0 else 0
    accuracy = (correct / len(with_results) * 100) if len(with_results) > 0 else 0

    print(f"Total predictions: {total}")
    print(f"With results: {len(with_results)}")
    print(f"Pending: {pending}")
    print(f"Correct: {int(correct)}")
    print(f"Accuracy: {accuracy:.1f}%")

    # Accuracy by market type
    if len(with_results) > 0:
        print("\n📈 ACCURACY BY MARKET:")
        market_stats = with_results.groupby('market').agg({
            'correct': ['sum', 'count'],
            'predicted_probability': 'mean'
        }).round(3)
        market_stats.columns = ['Correct', 'Total', 'Avg Prob']
        market_stats['Accuracy'] = (market_stats['Correct'] / market_stats['Total'] * 100).round(1)
        market_stats = market_stats.sort_values('Accuracy', ascending=False)
        print(market_stats.to_string())

        print("\n🏆 ACCURACY BY LEAGUE:")
        league_stats = with_results.groupby('league').agg({
            'correct': ['sum', 'count']
        }).round(3)
        league_stats.columns = ['Correct', 'Total']
        league_stats['Accuracy'] = (league_stats['Correct'] / league_stats['Total'] * 100).round(1)
        league_stats = league_stats.sort_values('Accuracy', ascending=False)
        print(league_stats.to_string())
    else:
        market_stats = pd.DataFrame()
        league_stats = pd.DataFrame()

    # Generate HTML report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Weekly Accuracy Report - {week_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #667eea; margin-bottom: 5px; }}
        .subtitle {{ color: #666; margin-bottom: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .stat-box {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-box .number {{ font-size: 2em; font-weight: bold; }}
        .stat-box .label {{ font-size: 0.9em; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; color: #667eea; font-weight: 600; position: sticky; top: 0; }}
        .correct {{ background: #d4edda; }}
        .incorrect {{ background: #f8d7da; }}
        .pending {{ background: #fff3cd; }}
        .prob {{ font-weight: bold; }}
        .market-summary {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .market-summary h3 {{ margin-top: 0; color: #667eea; }}
    </style>
</head>
<body>
    <div class='container'>
        <h1>📊 Weekly Accuracy Report</h1>
        <p class='subtitle'>Week: {week_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

        <div class='stats'>
            <div class='stat-box'>
                <div class='number'>{total}</div>
                <div class='label'>Total Predictions</div>
            </div>
            <div class='stat-box'>
                <div class='number'>{int(correct)}</div>
                <div class='label'>Correct</div>
            </div>
            <div class='stat-box'>
                <div class='number'>{accuracy:.1f}%</div>
                <div class='label'>Accuracy</div>
            </div>
            <div class='stat-box'>
                <div class='number'>{pending}</div>
                <div class='label'>Pending</div>
            </div>
        </div>
"""

    if not market_stats.empty:
        html += """
        <div class='market-summary'>
            <h3>📈 Accuracy by Market</h3>
            <table>
                <tr><th>Market</th><th>Correct</th><th>Total</th><th>Accuracy</th><th>Avg Prob</th></tr>
"""
        for market, row in market_stats.iterrows():
            acc_class = 'correct' if row['Accuracy'] >= 60 else ('incorrect' if row['Accuracy'] < 50 else '')
            html += f"<tr class='{acc_class}'><td><strong>{market}</strong></td><td>{int(row['Correct'])}</td><td>{int(row['Total'])}</td><td>{row['Accuracy']:.1f}%</td><td>{row['Avg Prob']:.1%}</td></tr>"
        html += "</table></div>"

    if not league_stats.empty:
        html += """
        <div class='market-summary'>
            <h3>🏆 Accuracy by League</h3>
            <table>
                <tr><th>League</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>
"""
        for league, row in league_stats.iterrows():
            acc_class = 'correct' if row['Accuracy'] >= 60 else ('incorrect' if row['Accuracy'] < 50 else '')
            html += f"<tr class='{acc_class}'><td><strong>{league}</strong></td><td>{int(row['Correct'])}</td><td>{int(row['Total'])}</td><td>{row['Accuracy']:.1f}%</td></tr>"
        html += "</table></div>"

    html += """
        <h3>📋 All Predictions (Line by Line)</h3>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>League</th>
                    <th>Match</th>
                    <th>Market</th>
                    <th>Prediction</th>
                    <th>Probability</th>
                    <th>Actual</th>
                    <th>Result</th>
                </tr>
            </thead>
            <tbody>
"""

    for _, row in df.iterrows():
        if pd.isna(row['actual_outcome']):
            result_class = 'pending'
            result_text = '⏳ Pending'
        elif row['correct'] == 1:
            result_class = 'correct'
            result_text = '✅ Correct'
        else:
            result_class = 'incorrect'
            result_text = '❌ Wrong'

        html += f"""
                <tr class='{result_class}'>
                    <td>{row['match_date']}</td>
                    <td><strong>{row['league']}</strong></td>
                    <td>{row['home_team']} vs {row['away_team']}</td>
                    <td>{row['market']}</td>
                    <td>{row['predicted_outcome']}</td>
                    <td class='prob'>{row['predicted_probability']:.1%}</td>
                    <td>{row['actual_outcome'] if pd.notna(row['actual_outcome']) else '-'}</td>
                    <td>{result_text}</td>
                </tr>"""

    html += """
            </tbody>
        </table>
    </div>
</body>
</html>"""

    # Save HTML
    html_path = output_dir / f"accuracy_report_{week_id}.html"
    html_path.write_text(html, encoding='utf-8')
    print(f"\n✅ HTML report saved: {html_path}")

    # Also save CSV
    csv_path = output_dir / f"accuracy_report_{week_id}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV report saved: {csv_path}")

    conn.close()
    return df


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    print("🎯 Accuracy Tracking System")
    print("=" * 50)

    tracker = AccuracyTracker()

    # Check for command line arguments
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()

        if cmd == 'report':
            # Generate weekly report
            week_id = sys.argv[2] if len(sys.argv) > 2 else None
            generate_weekly_accuracy_report(week_id)

        elif cmd == 'rankings':
            rankings = tracker.get_market_rankings()
            if not rankings.empty:
                print(rankings.to_string(index=False))
            else:
                print("No data yet")

        else:
            print(f"Unknown command: {cmd}")
    else:
        print("\nUsage:")
        print("  python accuracy_tracker.py report [week_id]  - Generate weekly accuracy report")
        print("  python accuracy_tracker.py rankings          - Show market rankings")
        print("\nCurrent Market Rankings:")
        rankings = tracker.get_market_rankings()
        if not rankings.empty:
            print(rankings.to_string(index=False))
        else:
            print("No data yet - start logging predictions!")

        print("\n📊 Export report:")
        tracker.export_accuracy_report()
