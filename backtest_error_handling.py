# backtest_detailed.py
"""
Detailed Backtest Analysis
Breaks down accuracy by:
- League
- Confidence buckets (e.g., 85%+, 80-85%, etc.)
- League + Confidence combinations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import shutil

from config import DATA_DIR, OUTPUT_DIR, FEATURES_PARQUET, MODEL_ARTIFACTS_DIR

class DetailedBacktest:
    """Backtest with granular league and confidence analysis"""
    
    # Markets to analyze
    MARKETS = {
        # Overs
        'OU_0_5_O': {'actual': 'y_OU_0_5', 'winning': 'O', 'name': 'Over 0.5'},
        'OU_1_5_O': {'actual': 'y_OU_1_5', 'winning': 'O', 'name': 'Over 1.5'},
        'OU_2_5_O': {'actual': 'y_OU_2_5', 'winning': 'O', 'name': 'Over 2.5'},
        'OU_3_5_O': {'actual': 'y_OU_3_5', 'winning': 'O', 'name': 'Over 3.5'},
        'OU_4_5_O': {'actual': 'y_OU_4_5', 'winning': 'O', 'name': 'Over 4.5'},
        # Unders
        'OU_0_5_U': {'actual': 'y_OU_0_5', 'winning': 'U', 'name': 'Under 0.5'},
        'OU_1_5_U': {'actual': 'y_OU_1_5', 'winning': 'U', 'name': 'Under 1.5'},
        'OU_2_5_U': {'actual': 'y_OU_2_5', 'winning': 'U', 'name': 'Under 2.5'},
        'OU_3_5_U': {'actual': 'y_OU_3_5', 'winning': 'U', 'name': 'Under 3.5'},
        'OU_4_5_U': {'actual': 'y_OU_4_5', 'winning': 'U', 'name': 'Under 4.5'},
        # BTTS
        'BTTS_Y': {'actual': 'y_BTTS', 'winning': 'Y', 'name': 'BTTS Yes'},
        'BTTS_N': {'actual': 'y_BTTS', 'winning': 'N', 'name': 'BTTS No'},
        # 1X2
        '1X2_H': {'actual': 'y_1X2', 'winning': 'H', 'name': '1X2 Home'},
        '1X2_D': {'actual': 'y_1X2', 'winning': 'D', 'name': '1X2 Draw'},
        '1X2_A': {'actual': 'y_1X2', 'winning': 'A', 'name': '1X2 Away'},
    }
    
    # Confidence buckets
    CONFIDENCE_BUCKETS = [
        (0.90, 1.00, '90-100%'),
        (0.85, 0.90, '85-90%'),
        (0.80, 0.85, '80-85%'),
        (0.75, 0.80, '75-80%'),
        (0.70, 0.75, '70-75%'),
        (0.65, 0.70, '65-70%'),
        (0.60, 0.65, '60-65%'),
    ]
    
    def __init__(self, 
                 start_date: str,
                 end_date: str,
                 test_window_days: int = 7):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.test_window_days = test_window_days
        
        # Storage for detailed results
        self.all_predictions = []
        
        # Create output directory
        self.backtest_dir = OUTPUT_DIR / "backtest_outputs"
        self.backtest_dir.mkdir(exist_ok=True)
        
        self.backup_existing_outputs()
    
    def backup_existing_outputs(self):
        """Backup existing files"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            weekly_bets = OUTPUT_DIR / "weekly_bets.csv"
            if weekly_bets.exists():
                backup_path = OUTPUT_DIR / f"weekly_bets_backup_{timestamp}.csv"
                shutil.copy(weekly_bets, backup_path)
                print(f"✅ Backed up weekly_bets.csv")
        except Exception as e:
            print(f"⚠️ Backup error: {e}")
    
    def load_features_data(self) -> pd.DataFrame:
        """Load features"""
        try:
            if not FEATURES_PARQUET.exists():
                raise FileNotFoundError("Run build_features(force=True) first")
            
            print(f"📂 Loading features...")
            df = pd.read_parquet(FEATURES_PARQUET)
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"✅ Loaded {len(df)} matches")
            return df
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def get_weekly_periods(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Generate weekly test periods"""
        periods = []
        current = self.start_date
        
        while current <= self.end_date:
            week_end = current + timedelta(days=self.test_window_days)
            periods.append((current, week_end))
            current = week_end
        
        return periods
    
    def train_models(self, train_df: pd.DataFrame) -> bool:
        """Train models"""
        try:
            print("   🔧 Training...")
            
            # Fill NaN
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if train_df[col].isna().sum() > 0:
                    train_df[col] = train_df[col].fillna(train_df[col].median())
            train_df = train_df.fillna(0)
            
            # Backup and replace
            temp_features = DATA_DIR / "temp_backtest_features.parquet"
            backup_features = DATA_DIR / "original_features_backup.parquet"
            
            train_df.to_parquet(temp_features)
            
            if FEATURES_PARQUET.exists():
                shutil.copy(FEATURES_PARQUET, backup_features)
            
            shutil.copy(temp_features, FEATURES_PARQUET)
            
            try:
                from models import train_all_targets
                models = train_all_targets(MODEL_ARTIFACTS_DIR)
                success = len(models) > 0
                
                if backup_features.exists():
                    shutil.copy(backup_features, FEATURES_PARQUET)
                
                temp_features.unlink(missing_ok=True)
                backup_features.unlink(missing_ok=True)
                
                return success
            except Exception as e:
                print(f"   ⚠️ Training failed: {e}")
                if backup_features.exists():
                    shutil.copy(backup_features, FEATURES_PARQUET)
                return False
                
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            return False
    
    def generate_predictions(self, test_df: pd.DataFrame, week_num: int) -> pd.DataFrame:
        """Generate predictions"""
        try:
            fixtures = test_df[['Date', 'League', 'HomeTeam', 'AwayTeam']].copy()
            temp_fixtures = self.backtest_dir / f"temp_fixtures_{week_num}.csv"
            fixtures.to_csv(temp_fixtures, index=False)
            
            try:
                from predict import predict_week
                predict_week(temp_fixtures)
                
                predictions_file = OUTPUT_DIR / "weekly_bets.csv"
                
                if predictions_file.exists():
                    predictions = pd.read_csv(predictions_file)
                    predictions['Date'] = pd.to_datetime(predictions['Date'])
                    
                    test_with_preds = test_df.merge(
                        predictions,
                        on=['Date', 'League', 'HomeTeam', 'AwayTeam'],
                        how='left'
                    )
                    
                    temp_fixtures.unlink(missing_ok=True)
                    return test_with_preds
                else:
                    print("   ⚠️ No predictions")
                    return test_df
                    
            except Exception as e:
                print(f"   ⚠️ Prediction failed: {e}")
                temp_fixtures.unlink(missing_ok=True)
                return test_df
                
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            return test_df
    
    def extract_prediction_details(self, df: pd.DataFrame):
        """Extract all predictions with details for analysis"""
        for idx, row in df.iterrows():
            league = row.get('League', 'Unknown')
            match = f"{row.get('HomeTeam', 'Home')} vs {row.get('AwayTeam', 'Away')}"
            
            for market_key, market_info in self.MARKETS.items():
                pred_col = f'BLEND_{market_key}'
                actual_col = market_info['actual']
                
                if pred_col not in df.columns or actual_col not in df.columns:
                    continue
                
                confidence = row.get(pred_col)
                actual_outcome = row.get(actual_col)
                
                if pd.isna(confidence) or pd.isna(actual_outcome):
                    continue
                
                # Convert confidence if string
                if isinstance(confidence, str):
                    confidence = float(confidence.strip('%')) / 100
                
                # Check if prediction correct
                won = (actual_outcome == market_info['winning'])
                
                # Find confidence bucket
                bucket = None
                for min_conf, max_conf, bucket_name in self.CONFIDENCE_BUCKETS:
                    if min_conf <= confidence < max_conf:
                        bucket = bucket_name
                        break
                
                if bucket is None:
                    continue
                
                self.all_predictions.append({
                    'Date': row['Date'],
                    'League': league,
                    'Match': match,
                    'Market': market_info['name'],
                    'Market_Key': market_key,
                    'Confidence': confidence,
                    'Confidence_Bucket': bucket,
                    'Predicted': market_info['winning'],
                    'Actual': actual_outcome,
                    'Correct': won
                })
    
    def run_backtest(self):
        """Run backtest and collect detailed data"""
        try:
            print("\n🔬 DETAILED BACKTEST ANALYSIS")
            print("="*60)
            print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
            print(f"Analyzing: League + Confidence breakdowns")
            print("="*60)
            
            full_df = self.load_features_data()
            full_df = full_df[
                (full_df['Date'] >= self.start_date) & 
                (full_df['Date'] <= self.end_date)
            ]
            
            periods = self.get_weekly_periods()
            print(f"\n📅 Testing {len(periods)} weeks\n")
            
            for i, (week_start, week_end) in enumerate(periods, 1):
                try:
                    print(f"Week {i}/{len(periods)}: {week_start.date()} to {week_end.date()}")
                    
                    train_df = full_df[full_df['Date'] < week_start].copy()
                    test_df = full_df[(full_df['Date'] >= week_start) & (full_df['Date'] < week_end)].copy()
                    
                    if len(train_df) < 100 or len(test_df) == 0:
                        print(f"   ⚠️ Insufficient data")
                        continue
                    
                    print(f"   📊 {len(test_df)} matches")
                    
                    # Train
                    if not self.train_models(train_df):
                        print(f"   ❌ Training failed")
                        continue
                    
                    # Predict
                    print(f"   🔮 Predicting...")
                    test_with_preds = self.generate_predictions(test_df, i)
                    
                    # Extract details
                    self.extract_prediction_details(test_with_preds)
                    print(f"   ✅ Collected {len(self.all_predictions)} total predictions")
                    
                except Exception as e:
                    print(f"   ⚠️ Error: {e}")
                    continue
            
            # Generate analysis
            self.generate_detailed_analysis()
            self.restore_outputs()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.restore_outputs()
    
    def generate_detailed_analysis(self):
        """Generate comprehensive analysis"""
        try:
            if not self.all_predictions:
                print("\n⚠️ No predictions to analyze")
                return
            
            df = pd.DataFrame(self.all_predictions)
            
            print("\n" + "="*60)
            print("📊 DETAILED ANALYSIS")
            print("="*60)
            
            # 1. Overall Market + Confidence
            print("\n🎯 ACCURACY BY MARKET + CONFIDENCE:")
            market_conf = df.groupby(['Market', 'Confidence_Bucket']).agg({
                'Correct': ['count', 'sum', 'mean']
            }).round(3)
            market_conf.columns = ['Total', 'Correct', 'Accuracy']
            market_conf = market_conf.sort_values(['Market', 'Accuracy'], ascending=[True, False])
            print(market_conf.to_string())
            
            # Save
            market_conf.to_csv(self.backtest_dir / "market_confidence_analysis.csv")
            
            # 2. League + Market breakdown
            print("\n🏆 ACCURACY BY LEAGUE + MARKET:")
            league_market = df.groupby(['League', 'Market']).agg({
                'Correct': ['count', 'sum', 'mean']
            }).round(3)
            league_market.columns = ['Total', 'Correct', 'Accuracy']
            league_market = league_market[league_market['Total'] >= 10]  # Min 10 samples
            league_market = league_market.sort_values('Accuracy', ascending=False)
            print(league_market.head(20).to_string())
            
            # Save
            league_market.to_csv(self.backtest_dir / "league_market_analysis.csv")
            
            # 3. League + Market + Confidence (THE HOLY GRAIL)
            print("\n💎 BEST COMBINATIONS (League + Market + Confidence):")
            triple = df.groupby(['League', 'Market', 'Confidence_Bucket']).agg({
                'Correct': ['count', 'sum', 'mean']
            }).round(3)
            triple.columns = ['Total', 'Correct', 'Accuracy']
            triple = triple[triple['Total'] >= 5]  # Min 5 samples
            triple = triple.sort_values('Accuracy', ascending=False)
            
            # Show top 30
            print(triple.head(30).to_string())
            
            # Save full
            triple.to_csv(self.backtest_dir / "league_market_confidence_analysis.csv")
            
            # 4. High confidence gold mines (85%+)
            print("\n🥇 HIGH CONFIDENCE BETS (85%+ prediction confidence):")
            high_conf = df[df['Confidence'] >= 0.85]
            if len(high_conf) > 0:
                high_conf_stats = high_conf.groupby(['League', 'Market']).agg({
                    'Correct': ['count', 'sum', 'mean']
                }).round(3)
                high_conf_stats.columns = ['Total', 'Correct', 'Accuracy']
                high_conf_stats = high_conf_stats[high_conf_stats['Total'] >= 3]
                high_conf_stats = high_conf_stats.sort_values('Accuracy', ascending=False)
                print(high_conf_stats.head(20).to_string())
                
                high_conf_stats.to_csv(self.backtest_dir / "high_confidence_gold.csv")
            else:
                print("   No predictions with 85%+ confidence")
            
            # 5. Save raw predictions
            df.to_csv(self.backtest_dir / "all_predictions_detailed.csv", index=False)
            
            print("\n" + "="*60)
            print("💡 KEY INSIGHTS:")
            print("="*60)
            
            # Find best league + market combos
            best_combos = league_market[league_market['Accuracy'] >= 0.70].head(10)
            if len(best_combos) > 0:
                print("\n✅ BEST LEAGUE + MARKET (70%+ accuracy):")
                for idx, row in best_combos.iterrows():
                    print(f"   • {idx[0]} - {idx[1]}: {row['Accuracy']*100:.1f}% ({int(row['Correct'])}/{int(row['Total'])})")
            
            # Find best high confidence bets
            if len(high_conf) > 0:
                best_high = high_conf_stats[high_conf_stats['Accuracy'] >= 0.80].head(5)
                if len(best_high) > 0:
                    print("\n🎯 GOLD STANDARD (85%+ confidence, 80%+ accuracy):")
                    for idx, row in best_high.iterrows():
                        print(f"   • {idx[0]} - {idx[1]}: {row['Accuracy']*100:.1f}% ({int(row['Correct'])}/{int(row['Total'])})")
            
            print("\n✅ Analysis complete! Check backtest_outputs/ for detailed CSVs")
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    def restore_outputs(self):
        """Restore original files"""
        try:
            print(f"\n🧹 Cleaning up...")
            
            weekly_bets = OUTPUT_DIR / "weekly_bets.csv"
            if weekly_bets.exists():
                latest_backup = None
                for backup in OUTPUT_DIR.glob("weekly_bets_backup_*.csv"):
                    if latest_backup is None or backup.stat().st_mtime > latest_backup.stat().st_mtime:
                        latest_backup = backup
                
                if latest_backup:
                    shutil.copy(latest_backup, weekly_bets)
                    print(f"✅ Restored original weekly_bets.csv")
                else:
                    weekly_bets.unlink()
            
            print(f"✅ All files saved to: {self.backtest_dir}")
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    try:
        print("\n⚽ DETAILED BACKTEST ANALYSIS")
        print("="*60)
        print("This will analyze:")
        print("  • Accuracy by confidence buckets (60-65%, 65-70%, etc.)")
        print("  • Accuracy by league")
        print("  • League + Market combinations")
        print("  • League + Market + Confidence (find gold mines!)")
        print("="*60)
        
        # Get period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        print(f"\nDefault: Last 6 months ({start_date.date()} to {end_date.date()})")
        choice = input("Use default? (y/n, default=y): ").strip().lower()
        
        if choice == 'n':
            start_input = input("Start date (YYYY-MM-DD): ").strip()
            end_input = input("End date (YYYY-MM-DD): ").strip()
            start_date = datetime.strptime(start_input, '%Y-%m-%d')
            end_date = datetime.strptime(end_input, '%Y-%m-%d')
        
        # Run
        backtest = DetailedBacktest(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            test_window_days=7
        )
        
        backtest.run_backtest()
        
        print("\n✅ COMPLETE!")
        print("📂 Check outputs/backtest_outputs/ for:")
        print("   • market_confidence_analysis.csv")
        print("   • league_market_analysis.csv")
        print("   • league_market_confidence_analysis.csv")
        print("   • high_confidence_gold.csv (85%+ bets)")
        print("   • all_predictions_detailed.csv (raw data)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        input("\nPress Enter to exit...")