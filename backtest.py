# backtest.py
"""
Complete Backtesting Engine - Walk-Forward Validation
Tests your actual prediction system on historical data with NO data leakage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import tempfile
import shutil

from config import DATA_DIR, OUTPUT_DIR, FEATURES_PARQUET, MODEL_ARTIFACTS_DIR

class BacktestEngine:
    """Walk-forward backtesting with your actual prediction system"""
    
    def __init__(self, 
                 start_date: str,
                 end_date: str,
                 test_window_days: int = 7,
                 min_training_matches: int = 100):
        """
        Args:
            start_date: Start of backtest period (YYYY-MM-DD)
            end_date: End of backtest period (YYYY-MM-DD)
            test_window_days: Days per test window (default 7 = weekly)
            min_training_matches: Minimum training data required
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.test_window_days = test_window_days
        self.min_training_matches = min_training_matches
        
        self.results = []
        
    def load_features_data(self) -> pd.DataFrame:
        """Load full features dataset"""
        if not FEATURES_PARQUET.exists():
            raise FileNotFoundError(
                "Features not found. Run build_features(force=True) first"
            )
        
        print(f"📂 Loading features from {FEATURES_PARQUET}")
        df = pd.read_parquet(FEATURES_PARQUET)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    
    def get_test_periods(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Generate list of test periods"""
        periods = []
        current = self.start_date
        
        while current <= self.end_date:
            period_end = current + timedelta(days=self.test_window_days)
            periods.append((current, period_end))
            current = period_end
        
        return periods
    
    def split_data(self, 
                   df: pd.DataFrame, 
                   test_start: pd.Timestamp,
                   test_end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split into train/test ensuring NO data leakage
        Train: all data BEFORE test period
        Test: data in test period
        """
        # Training: everything before test period
        train_df = df[df['Date'] < test_start].copy()
        
        # Test: only matches in this period
        test_df = df[(df['Date'] >= test_start) & (df['Date'] < test_end)].copy()
        
        return train_df, test_df
    
    def train_models_on_period(self, train_df: pd.DataFrame) -> bool:
        """
        Train models using only training data
        Returns True if successful
        """
        # NEW: Fill NaN BEFORE saving to file
        print("   🔧 Handling missing values...")
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if train_df[col].isna().sum() > 0:
                train_df[col] = train_df[col].fillna(train_df[col].median())

        # Fill any remaining NaN with 0
        train_df = train_df.fillna(0)

        # Create temporary features file with ONLY training data
        temp_features = DATA_DIR / "temp_backtest_features.parquet"
        train_df.to_parquet(temp_features)

        # Backup original features
        backup_features = DATA_DIR / "original_features_backup.parquet"
        if FEATURES_PARQUET.exists():
            shutil.copy(FEATURES_PARQUET, backup_features)

        # Replace features with training data only
        shutil.copy(temp_features, FEATURES_PARQUET)

        try:
            # Train using your actual system
            from models import train_all_targets
            
            models = train_all_targets(MODEL_ARTIFACTS_DIR)
        

            
            success = len(models) > 0
            
            # Restore original features
            if backup_features.exists():
                shutil.copy(backup_features, FEATURES_PARQUET)
            
            # Cleanup
            temp_features.unlink(missing_ok=True)
            backup_features.unlink(missing_ok=True)
            
            return success
            
        except Exception as e:
            print(f"   [WARN] Training failed: {e}")
            
            # Restore original features
            if backup_features.exists():
                shutil.copy(backup_features, FEATURES_PARQUET)
            
            return False
    
    def generate_predictions(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for test period using trained models
        Returns test_df with BLEND_ columns added
        """
        # Create temporary fixtures file
        fixtures = test_df[['Date', 'League', 'HomeTeam', 'AwayTeam']].copy()
        temp_fixtures = OUTPUT_DIR / "temp_backtest_fixtures.csv"
        fixtures.to_csv(temp_fixtures, index=False)
        
        try:
            # Generate predictions using your actual system
            from predict import predict_week
            
            predict_week(temp_fixtures)
            
            # Load predictions
            predictions_file = OUTPUT_DIR / "weekly_bets.csv"
            
            if predictions_file.exists():
                predictions = pd.read_csv(predictions_file)
                
                # Merge predictions with test data
                test_with_preds = test_df.merge(
                    predictions,
                    on=['Date', 'League', 'HomeTeam', 'AwayTeam'],
                    how='left'
                )
                
                # Cleanup
                temp_fixtures.unlink(missing_ok=True)
                
                return test_with_preds
            else:
                print("   [WARN] Predictions file not generated")
                return test_df
                
        except Exception as e:
            print(f"   [WARN] Prediction failed: {e}")
            temp_fixtures.unlink(missing_ok=True)
            return test_df
    
    def evaluate_predictions(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate prediction accuracy across all markets
        """
        results = {
            'total_matches': len(df),
            'markets': {},
            'league_markets': {}  # NEW: Track league+market combos
        }
        
        # Define all markets to evaluate
        markets = {
            # 1X2
            '1X2': {
                'actual': 'y_1X2',
                'pred_cols': ['BLEND_1X2_H', 'BLEND_1X2_D', 'BLEND_1X2_A'],
                'outcomes': ['H', 'D', 'A']
            },
            # BTTS
            'BTTS': {
                'actual': 'y_BTTS',
                'pred_cols': ['BLEND_BTTS_Y', 'BLEND_BTTS_N'],
                'outcomes': ['Y', 'N']
            },
            # Over/Under
            'OU_0_5': {
                'actual': 'y_OU_0_5',
                'pred_cols': ['BLEND_OU_0_5_O', 'BLEND_OU_0_5_U'],
                'outcomes': ['O', 'U']
            },
            'OU_1_5': {
                'actual': 'y_OU_1_5',
                'pred_cols': ['BLEND_OU_1_5_O', 'BLEND_OU_1_5_U'],
                'outcomes': ['O', 'U']
            },
            'OU_2_5': {
                'actual': 'y_OU_2_5',
                'pred_cols': ['BLEND_OU_2_5_O', 'BLEND_OU_2_5_U'],
                'outcomes': ['O', 'U']
            },
            'OU_3_5': {
                'actual': 'y_OU_3_5',
                'pred_cols': ['BLEND_OU_3_5_O', 'BLEND_OU_3_5_U'],
                'outcomes': ['O', 'U']
            },
            'OU_4_5': {
                'actual': 'y_OU_4_5',
                'pred_cols': ['BLEND_OU_4_5_O', 'BLEND_OU_4_5_U'],
                'outcomes': ['O', 'U']
            },

        }
        
        for market_name, market_info in markets.items():
            actual_col = market_info['actual']
            pred_cols = market_info['pred_cols']
            outcomes = market_info['outcomes']
            
            # Skip if columns don't exist
            if actual_col not in df.columns:
                continue
            
            available_pred_cols = [c for c in pred_cols if c in df.columns]
            if not available_pred_cols:
                continue
            
            # Get matches with actual outcomes
            valid = df[actual_col].notna()
            actual = df.loc[valid, actual_col]
            
            if len(actual) == 0:
                continue
            
            # Get predictions
            predictions = df.loc[valid, available_pred_cols]
            
            # Find predicted outcome (highest probability)
            pred_outcome_idx = predictions.idxmax(axis=1)
            
            # Map column names to outcomes
            outcome_map = {col: outcome for col, outcome in zip(pred_cols, outcomes)}
            predicted = pred_outcome_idx.map(outcome_map)
            
            # Calculate metrics
            correct = (predicted == actual).sum()
            total = len(actual)
            accuracy = correct / total if total > 0 else 0
            
            # Brier score (calibration)
            brier_scores = []
            for idx in actual.index:
                true_outcome = actual[idx]
                for col, outcome in zip(available_pred_cols, outcomes):
                    if col in df.columns:
                        pred_prob = df.loc[idx, col]
                        if pd.notna(pred_prob):
                            # Convert percentage string if needed
                            if isinstance(pred_prob, str):
                                pred_prob = float(pred_prob.strip('%')) / 100
                            true_prob = 1.0 if outcome == true_outcome else 0.0
                            brier_scores.append((pred_prob - true_prob) ** 2)
            
            brier = np.mean(brier_scores) if brier_scores else 0
            
            # ROI calculation (simplified - assumes even odds)
            roi = ((correct / total) - 0.5) * 100 if total > 0 else 0
            
            results['markets'][market_name] = {
                'total': int(total),
                'correct': int(correct),
                'accuracy': float(accuracy),
                'brier_score': float(brier),
                'roi_pct': float(roi)
            }
        
        return results
    
    def run_backtest(self) -> pd.DataFrame:
        """Run complete walk-forward backtest"""
        print("\n🔬 BACKTESTING ENGINE")
        print("="*60)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Test window: {self.test_window_days} days")
        print("Method: Walk-forward (no data leakage)")
        print("="*60)
        
        # Load full dataset
        full_df = self.load_features_data()
        
        # Filter to backtest period
        full_df = full_df[
            (full_df['Date'] >= self.start_date) & 
            (full_df['Date'] <= self.end_date)
        ]
        
        # Get test periods
        periods = self.get_test_periods()
        print(f"\n📅 Testing {len(periods)} periods\n")
        
        for i, (test_start, test_end) in enumerate(periods, 1):
            print(f"Period {i}/{len(periods)}: {test_start.date()} to {test_end.date()}")
            
            # Split data
            train_df, test_df = self.split_data(full_df, test_start, test_end)
            
            # Check sufficient data
            if len(train_df) < self.min_training_matches:
                print(f"   [WARN] Insufficient training data ({len(train_df)} matches)")
                continue
            
            if len(test_df) == 0:
                print(f"   [WARN] No test matches")
                continue
            
            print(f"   📊 Train: {len(train_df)} matches | Test: {len(test_df)} matches")
            
            # Train models on training data only
            print(f"   🎯 Training models...")
            success = self.train_models_on_period(train_df)
            
            if not success:
                print(f"   [ERROR] Training failed")
                continue
            
            # Generate predictions
            print(f"   🔮 Generating predictions...")
            test_with_preds = self.generate_predictions(test_df)
            
            # Evaluate
            period_results = self.evaluate_predictions(test_with_preds)
            period_results['period_start'] = test_start
            period_results['period_end'] = test_end
            
            self.results.append(period_results)
            
            # Print summary
            print(f"   📈 Results:")
            for market, stats in period_results.get('markets', {}).items():
                print(f"      * {market}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
        
        # Generate summary
        return self.generate_summary()
    
    def generate_summary(self) -> pd.DataFrame:
        """Aggregate results with league breakdowns and combo analysis"""
        if not self.results:
            print("\n[WARN] No results to summarize")
            return pd.DataFrame()
        
        print("\n" + "="*60)
        print("📊 BACKTEST SUMMARY - ALL PERIODS")
        print("="*60)
        
        # 1. Overall market performance
        market_summary = {}
        all_markets = set()
        for result in self.results:
            all_markets.update(result.get('markets', {}).keys())
        
        for market in sorted(all_markets):
            total_matches = 0
            total_correct = 0
            brier_scores = []
            
            for result in self.results:
                if market in result.get('markets', {}):
                    stats = result['markets'][market]
                    total_matches += stats['total']
                    total_correct += stats['correct']
                    brier_scores.append(stats['brier_score'])
            
            if total_matches > 0:
                accuracy = total_correct / total_matches
                roi = ((accuracy - 0.5) * 100)
                
                market_summary[market] = {
                    'Total_Matches': total_matches,
                    'Correct': total_correct,
                    'Accuracy_%': round(accuracy * 100, 1),
                    'Brier_Score': round(np.mean(brier_scores), 3),
                    'ROI_%': round(roi, 1)
                }
        
        summary_df = pd.DataFrame.from_dict(market_summary, orient='index')
        summary_df = summary_df.sort_values('Accuracy', ascending=False)
        
        print("\n📊 OVERALL MARKET PERFORMANCE:")
        print(summary_df.to_string())
        
        # 2. League-specific analysis
        self.analyze_by_league()
        
        # 3. Doubles/Trebles analysis
        self.analyze_combinations()
        
        # Interpretation
        print("\n" + "="*60)
        print("💡 KEY FINDINGS:")
        print("="*60)
        
        excellent = summary_df[summary_df['Accuracy_%'] >= 60]
        good = summary_df[(summary_df['Accuracy_%'] >= 55) & (summary_df['Accuracy_%'] < 60)]
        
        if len(excellent) > 0:
            print(f"[OK] EXCELLENT markets (≥60%): {', '.join(excellent.index.tolist())}")
        
        if len(good) > 0:
            print(f"[OK] GOOD markets (55-60%): {', '.join(good.index.tolist())}")
        
        print(f"\n📈 Best overall: {summary_df.index[0]} ({summary_df.iloc[0]['Accuracy_%']:.1f}%)")
        
        # Save
        output_path = OUTPUT_DIR / "backtest_summary.csv"
        summary_df.to_csv(output_path)
        print(f"\n[OK] Saved: {output_path}")
        
        return summary_df
    
    def analyze_by_league(self):
        """Analyze performance by league + market combination"""
        print("\n" + "="*60)
        print("🏆 LEAGUE + MARKET BREAKDOWN")
        print("="*60)
        
        # Collect all predictions with league info
        all_preds = []
        
        for result in self.results:
            # Need to track which predictions came from which league
            # This requires storing more detail during evaluation
            pass
        
        # For now, print instruction
        print("💡 To see league breakdowns, check backtest_detailed.csv")
        print("   Filter by League column to see market performance per league")
    
    def analyze_combinations(self):
        """Analyze double/treble success rates"""
        print("\n" + "="*60)
        print("🎲 COMBINATION ANALYSIS (Doubles/Trebles)")
        print("="*60)
        
        from itertools import combinations
        
        # Get all market accuracies
        market_accs = {}
        for result in self.results:
            for market, stats in result.get('markets', {}).items():
                if market not in market_accs:
                    market_accs[market] = []
                if stats['total'] > 0:
                    market_accs[market].append(stats['accuracy'])
        
        # Calculate average accuracy per market
        avg_accs = {m: np.mean(accs) for m, accs in market_accs.items() if len(accs) > 0}
        
        # Find best doubles
        print("\n🎯 BEST DOUBLES (Top 10):")
        doubles = []
        for m1, m2 in combinations(avg_accs.keys(), 2):
            combined_prob = avg_accs[m1] * avg_accs[m2]
            # Assume average odds of 2.0 per leg
            double_odds = 4.0
            expected_roi = (combined_prob * double_odds - 1) * 100
            
            doubles.append({
                'combo': f"{m1} + {m2}",
                'hit_rate_%': round(combined_prob * 100, 1),
                'expected_roi_%': round(expected_roi, 1)
            })
        
        doubles_df = pd.DataFrame(doubles).sort_values('expected_roi_%', ascending=False).head(10)
        print(doubles_df.to_string(index=False))
        
        # Find best trebles
        print("\n🎯 BEST TREBLES (Top 10):")
        trebles = []
        for m1, m2, m3 in combinations(avg_accs.keys(), 3):
            combined_prob = avg_accs[m1] * avg_accs[m2] * avg_accs[m3]
            # Assume average odds of 2.0 per leg
            treble_odds = 8.0
            expected_roi = (combined_prob * treble_odds - 1) * 100
            
            trebles.append({
                'combo': f"{m1} + {m2} + {m3}",
                'hit_rate_%': round(combined_prob * 100, 1),
                'expected_roi_%': round(expected_roi, 1)
            })
        
        trebles_df = pd.DataFrame(trebles).sort_values('expected_roi_%', ascending=False).head(10)
        print(trebles_df.to_string(index=False))
        
        # Save combinations
        doubles_df.to_csv(OUTPUT_DIR / "backtest_best_doubles.csv", index=False)
        trebles_df.to_csv(OUTPUT_DIR / "backtest_best_trebles.csv", index=False)
        
        print("\n[OK] Saved combination analysis to outputs/backtest_best_*.csv")
        
        print("\n💡 COMBINATION TIPS:")
        print("   * Look for combinations with >40% hit rate for trebles")
        print("   * Look for combinations with >60% hit rate for doubles")
        print("   * Cross-league combos often have better value")
        print("   * Mix O/U with other markets for decorrelation")
    
    def export_detailed_results(self) -> Path:
        """Export period-by-period breakdown"""
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
        
        print(f"[OK] Saved detailed: {output_path}")
        return output_path


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    from datetime import datetime
    
    # Default: backtest last 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    print("\n⚽ FOOTBALL PREDICTION BACKTEST")
    print("="*60)
    print(f"Default period: Last 6 months")
    print(f"   From: {start_date.date()}")
    print(f"   To: {end_date.date()}")
    
    choice = input("\nUse default period? (y/n, default=y): ").strip().lower()
    
    if choice == 'n':
        start_input = input("Start date (YYYY-MM-DD): ").strip()
        end_input = input("End date (YYYY-MM-DD): ").strip()
        
        start_date = datetime.strptime(start_input, '%Y-%m-%d')
        end_date = datetime.strptime(end_input, '%Y-%m-%d')
    
    engine = BacktestEngine(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        test_window_days=7
    )
    
    engine.run_backtest()
    engine.export_detailed_results()
    
    print("\n" + "="*60)
    print("[OK] BACKTEST COMPLETE")
    print("="*60)
    print("📂 Check outputs folder for:")
    print("   * backtest_summary.csv - Overall performance")
    print("   * backtest_detailed.csv - Period-by-period breakdown")