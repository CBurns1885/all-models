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
        """Export detailed accuracy report"""
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query("""
            SELECT 
                week_id,
                league,
                market,
                total_predictions,
                correct_predictions,
                accuracy,
                brier_score,
                profit_loss
            FROM weekly_accuracy
            ORDER BY week_id DESC, accuracy DESC
        """, conn)
        
        conn.close()
        
        df.to_csv(output_path, index=False)
        print(f"[OK] Exported accuracy report: {output_path}")
        return df


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


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    print("🎯 Accuracy Tracking System")
    print("="*50)
    
    tracker = AccuracyTracker()
    
    print("\nCurrent Market Rankings:")
    rankings = tracker.get_market_rankings()
    
    if not rankings.empty:
        print(rankings.to_string(index=False))
    else:
        print("No data yet - start logging predictions!")
    
    print("\n📊 Export report:")
    tracker.export_accuracy_report()
