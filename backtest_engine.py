# backtest.py
"""
Backtesting Engine - Test prediction system on historical data
Walk-forward validation: Only use data BEFORE each week to predict that week
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
from collections import defaultdict

from config import DATA_DIR, OUTPUT_DIR
from features import build_features
from models import train_models

class BacktestEngine:
    """Walk-forward backtesting for prediction models"""
    
    def __init__(self, 
                 start_date: str,
                 end_date: str,
                 test_window_days: int = 7,
                 min_training_days: int = 365):
        """
        Args:
            start_date: Start of backtest period (YYYY-MM-DD)
            end_date: End of backtest period (YYYY-MM-DD)
            test_window_days: Days per test window (default 7 = weekly)
            min_training_days: Minimum days of training data required
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.test_window_days = test_window_days
        self.min_training_days = min_training_days
        
        self.results = []
        self.summary_stats = {}
        
    def load_features_data(self) -> pd.DataFrame:
        """Load full features dataset"""
        features_path = DATA_DIR / "features.pkl"
        
        if not features_path.exists():
            raise FileNotFoundError(
                "Features not found. Run build_features(force=True) first"
            )
        
        df = pd.read_pickle(features_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} matches from {df['Date'].min()} to {df['Date'].max()}")
        return df
    
    def get_test_weeks(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Generate list of (train_end, test_end) date pairs"""
        weeks = []
        current = self.start_date
        
        while current <= self.end_date:
            week_end = current + timedelta(days=self.test_window_days)
            weeks.append((current, week_end))
            current = week_end
        
        return weeks
    
    def split_train_test(self, 
                         df: pd.DataFrame, 
                         test_start: pd.Timestamp,
                         test_end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/test ensuring no data leakage
        
        Returns:
            train_df, test_df
        """
        # Training: all data BEFORE test period
        train_cutoff = test_start - timedelta(days=1)
        train_df = df[df['Date'] <= train_cutoff].copy()
        
        # Test: only matches in the test window
        test_df = df[(df['Date'] >= test_start) & (df['Date'] < test_end)].copy()
        
        # Ensure minimum training data
        if len(train_df) < 100:  # Arbitrary minimum
            return None, None
        
        return train_df, test_df
    
    def train_on_period(self, train_df: pd.DataFrame) -> Dict:
        """Train models on training period"""
        # Get feature columns (your existing logic)
        from features import _feature_columns
        feature_cols = _feature_columns()
        
        # Prepare training data
        X_train = train_df[feature_cols].fillna(0)
        
        # Train for each market
        models = {}
        markets = ['y_1X2', 'y_BTTS', 'y_OU_2_5', 'y_AH_0_0']
        
        for market in markets:
            if market not in train_df.columns:
                continue
            
            y_train = train_df[market].dropna()
            X_train_market = X_train.loc[y_train.index]
            
            # Simple model training (you'll use your actual train_models logic)
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            
            try:
                model.fit(X_train_market, y_train)
                models[market] = model
            except Exception as e:
                print(f"   ⚠️ Failed to train {market}: {e}")
        
        return models
    
    def predict_period(self, 
                      models: Dict, 
                      test_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for test period"""
        from features import _feature_columns
        feature_cols = _feature_columns()
        
        X_test = test_df[feature_cols].fillna(0)
        predictions = test_df.copy()
        
        for market, model in models.items():
            try:
                pred_proba = model.predict_proba(X_test)
                
                # Store probabilities for each class
                for idx, class_label in enumerate(model.classes_):
                    col_name = f"P_{market}_{class_label}"
                    predictions[col_name] = pred_proba[:, idx]
                
            except Exception as e:
                print(f"   ⚠️ Prediction failed for {market}: {e}")
        
        return predictions
    
    def evaluate_predictions(self, predictions: pd.DataFrame) -> Dict:
        """Evaluate prediction accuracy for a test period"""
        results = {
            'total_matches': len(predictions),
            'markets': {}
        }
        
        # Evaluate each market
        markets = {
            'y_1X2': ['H', 'D', 'A'],
            'y_BTTS': ['Y', 'N'],
            'y_OU_2_5': ['O', 'U'],
        }
        
        for market, classes in markets.items():
            if market not in predictions.columns:
                continue
            
            market_results = {
                'total': 0,
                'correct': 0,
                'accuracy': 0.0,
                'brier_score': 0.0,
                'profit': 0.0
            }
            
            # Get actual outcomes
            actual = predictions[market].dropna()
            if len(actual) == 0:
                continue
            
            market_results['total'] = len(actual)
            
            # Find predicted class (highest probability)
            prob_cols = [f"P_{market}_{c}" for c in classes]
            available_cols = [c for c in prob_cols if c in predictions.columns]
            
            if not available_cols:
                continue
            
            pred_class = predictions.loc[actual.index, available_cols].idxmax(axis=1)
            pred_class = pred_class.str.replace(f'P_{market}_', '')
            
            # Calculate accuracy
            correct = (pred_class == actual).sum()
            market_results['correct'] = int(correct)
            market_results['accuracy'] = correct / len(actual)
            
            # Calculate Brier score (calibration)
            brier_scores = []
            for idx in actual.index:
                true_class = actual[idx]
                for class_label in classes:
                    prob_col = f"P_{market}_{class_label}"
                    if prob_col in predictions.columns:
                        pred_prob = predictions.loc[idx, prob_col]
                        true_prob = 1.0 if class_label == true_class else 0.0
                        brier_scores.append((pred_prob - true_prob) ** 2)
            
            if brier_scores:
                market_results['brier_score'] = np.mean(brier_scores)
            
            # Simple profit calculation (assuming odds of 2.0 for correct predictions)
            # In real backtest, you'd use actual odds
            market_results['profit'] = correct - len(actual)  # Units won/lost
            
            results['markets'][market] = market_results
        
        return results
    
    def run_backtest(self) -> pd.DataFrame:
        """Run full walk-forward backtest"""
        print("\n🔬 BACKTESTING ENGINE")
        print("="*60)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Test window: {self.test_window_days} days")
        
        # Load data
        full_df = self.load_features_data()
        
        # Filter to backtest period
        full_df = full_df[
            (full_df['Date'] >= self.start_date) & 
            (full_df['Date'] <= self.end_date)
        ]
        
        # Generate test periods
        test_weeks = self.get_test_weeks()
        print(f"📅 Testing {len(test_weeks)} periods\n")
        
        all_results = []
        
        for i, (test_start, test_end) in enumerate(test_weeks, 1):
            print(f"Period {i}/{len(test_weeks)}: {test_start.date()} to {test_end.date()}")
            
            # Split data
            train_df, test_df = self.split_train_test(full_df, test_start, test_end)
            
            if train_df is None or len(test_df) == 0:
                print(f"   ⚠️ Skipping: insufficient data")
                continue
            
            print(f"   📊 Training: {len(train_df)} matches | Testing: {len(test_df)} matches")
            
            # Train models
            models = self.train_on_period(train_df)
            if not models:
                print(f"   ⚠️ No models trained")
                continue
            
            # Generate predictions
            predictions = self.predict_period(models, test_df)
            
            # Evaluate
            period_results = self.evaluate_predictions(predictions)
            period_results['period_start'] = test_start
            period_results['period_end'] = test_end
            
            all_results.append(period_results)
            
            # Print quick summary
            for market, stats in period_results.get('markets', {}).items():
                acc = stats.get('accuracy', 0)
                print(f"   • {market}: {acc:.1%} accuracy ({stats['correct']}/{stats['total']})")
        
        # Convert to DataFrame
        self.results = all_results
        return self.generate_summary()
    
    def generate_summary(self) -> pd.DataFrame:
        """Generate summary statistics across all periods"""
        if not self.results:
            return pd.DataFrame()
        
        print("\n" + "="*60)
        print("📊 BACKTEST SUMMARY")
        print("="*60)
        
        # Aggregate by market
        market_summary = {}
        
        all_markets = set()
        for result in self.results:
            all_markets.update(result.get('markets', {}).keys())
        
        for market in all_markets:
            total_matches = 0
            total_correct = 0
            total_profit = 0
            brier_scores = []
            
            for result in self.results:
                if market in result.get('markets', {}):
                    stats = result['markets'][market]
                    total_matches += stats['total']
                    total_correct += stats['correct']
                    total_profit += stats.get('profit', 0)
                    if stats.get('brier_score'):
                        brier_scores.append(stats['brier_score'])
            
            market_summary[market] = {
                'Total_Matches': total_matches,
                'Correct_Predictions': total_correct,
                'Accuracy': total_correct / total_matches if total_matches > 0 else 0,
                'Avg_Brier_Score': np.mean(brier_scores) if brier_scores else 0,
                'Total_Profit_Units': total_profit,
                'ROI': (total_profit / total_matches * 100) if total_matches > 0 else 0
            }
        
        summary_df = pd.DataFrame.from_dict(market_summary, orient='index')
        
        print("\n" + summary_df.to_string())
        print("\n" + "="*60)
        
        # Save to file
        output_path = OUTPUT_DIR / "backtest_summary.csv"
        summary_df.to_csv(output_path)
        print(f"✅ Saved summary: {output_path}")
        
        return summary_df
    
    def export_detailed_results(self) -> Path:
        """Export detailed period-by-period results"""
        detailed = []
        
        for result in self.results:
            for market, stats in result.get('markets', {}).items():
                detailed.append({
                    'period_start': result['period_start'],
                    'period_end': result['period_end'],
                    'market': market,
                    **stats
                })
        
        detailed_df = pd.DataFrame(detailed)
        output_path = OUTPUT_DIR / "backtest_detailed.csv"
        detailed_df.to_csv(output_path, index=False)
        
        print(f"✅ Saved detailed results: {output_path}")
        return output_path


# ============================================================================
# CLI INTERFACE
# ============================================================================

def run_backtest_cli():
    """Command-line interface for backtesting"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest prediction system")
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--window', type=int, default=7, help='Test window in days (default: 7)')
    
    args = parser.parse_args()
    
    engine = BacktestEngine(
        start_date=args.start,
        end_date=args.end,
        test_window_days=args.window
    )
    
    engine.run_backtest()
    engine.export_detailed_results()


if __name__ == "__main__":
    # Example: Backtest last 6 months
    from datetime import datetime
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    engine = BacktestEngine(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        test_window_days=7
    )
    
    engine.run_backtest()
    engine.export_detailed_results()
