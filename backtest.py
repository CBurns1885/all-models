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
    
    def train_models_on_period(self, train_df: pd.DataFrame, test_start: pd.Timestamp = None) -> bool:
        """
        Train models using only training data, then re-learn blend weights
        on a temporal validation slice to avoid leakage.
        Returns True if successful.
        """
        print("   Handling missing values...")
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if train_df[col].isna().sum() > 0:
                train_df[col] = train_df[col].fillna(train_df[col].median())

        train_df = train_df.fillna(0)

        temp_features = DATA_DIR / "temp_backtest_features.parquet"
        train_df.to_parquet(temp_features)

        backup_features = DATA_DIR / "original_features_backup.parquet"
        if FEATURES_PARQUET.exists():
            shutil.copy(FEATURES_PARQUET, backup_features)

        shutil.copy(temp_features, FEATURES_PARQUET)

        try:
            from models import train_all_targets
            models = train_all_targets(MODEL_ARTIFACTS_DIR)
            success = len(models) > 0

            # Re-learn blend weights on temporal validation slice
            if success and test_start is not None:
                try:
                    from blending import learn_blend_weights_temporal
                    val_end = test_start.strftime('%Y-%m-%d')
                    print(f"   Re-learning blend weights (val_end={val_end})...")
                    learn_blend_weights_temporal(val_end)
                except Exception as e:
                    print(f"   [WARN] Blend weight re-learning failed: {e}")

            if backup_features.exists():
                shutil.copy(backup_features, FEATURES_PARQUET)

            temp_features.unlink(missing_ok=True)
            backup_features.unlink(missing_ok=True)

            return success

        except Exception as e:
            print(f"   [WARN] Training failed: {e}")

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
                predictions['Date'] = pd.to_datetime(predictions['Date'])
                test_df_merge = test_df.copy()
                test_df_merge['Date'] = pd.to_datetime(test_df_merge['Date'])
                
                # Merge predictions with test data
                test_with_preds = test_df_merge.merge(
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
    
    def _discover_markets(self, df: pd.DataFrame) -> Dict:
        """
        Auto-discover all evaluable markets by scanning for y_* actual columns
        and matching BLEND_* or P_* prediction columns.
        """
        markets = {}
        PREFIX_MAP = {'GOAL_RANGE': 'GR'}

        y_cols = sorted(c for c in df.columns if c.startswith('y_'))

        for y_col in y_cols:
            market = y_col[2:]
            pred_prefix = PREFIX_MAP.get(market, market)
            actual_values = set(str(v) for v in df[y_col].dropna().unique())

            if not actual_values:
                continue

            for src in ('BLEND', 'P'):
                search = f'{src}_{pred_prefix}_'
                candidates = [c for c in df.columns if c.startswith(search)]
                if not candidates:
                    continue

                outcome_map = {}
                for col in candidates:
                    suffix = col[len(search):]
                    if suffix in actual_values:
                        outcome_map[col] = suffix
                    elif suffix.replace('_', '-') in actual_values:
                        outcome_map[col] = suffix.replace('_', '-')

                if outcome_map:
                    markets[market] = {
                        'actual': y_col,
                        'pred_cols': list(outcome_map.keys()),
                        'outcomes': list(outcome_map.values()),
                        'source': src,
                    }
                    break

        return markets

    def evaluate_predictions(self, df: pd.DataFrame) -> Dict:
        """
        Auto-discover and evaluate ALL markets where both actual (y_*)
        and prediction (BLEND_* or P_*) columns exist.
        Uses real bookmaker odds for ROI where available.
        """
        results = {
            'total_matches': len(df),
            'markets': {},
        }

        ODDS_MAP = {
            '1X2': {'H': 'B365H', 'D': 'B365D', 'A': 'B365A'},
            'BTTS': {'Y': 'Odds_BTTS_Y', 'N': 'Odds_BTTS_N'},
            'OU_2_5': {'O': 'Odds_O25', 'U': 'Odds_U25'},
        }

        MARKET_WEIGHTS = {
            '1X2': 3.0, 'BTTS': 2.0, 'OU_2_5': 3.0,
            'OU_1_5': 1.5, 'OU_3_5': 1.5, 'OU_4_5': 1.0,
        }

        discovered = self._discover_markets(df)

        for market, info in discovered.items():
            actual_col = info['actual']
            pred_cols = info['pred_cols']
            outcomes = info['outcomes']

            valid = df[actual_col].notna()
            actual = df.loc[valid, actual_col].astype(str)

            if len(actual) == 0:
                continue

            predictions = df.loc[valid, pred_cols]
            has_preds = predictions.notna().any(axis=1) & (predictions != 0).any(axis=1)
            valid_idx = has_preds[has_preds].index

            if len(valid_idx) < 5:
                continue

            actual = actual.loc[valid_idx]
            predictions = predictions.loc[valid_idx]

            outcome_map = dict(zip(pred_cols, outcomes))
            pred_col_idx = predictions.idxmax(axis=1)
            predicted = pred_col_idx.map(outcome_map)

            correct = (predicted == actual).sum()
            total = len(actual)
            accuracy = correct / total if total > 0 else 0

            brier_sum = 0.0
            brier_count = 0
            for idx in actual.index:
                true_out = actual[idx]
                for col, outcome in zip(pred_cols, outcomes):
                    prob = predictions.at[idx, col]
                    if pd.notna(prob):
                        if isinstance(prob, str):
                            prob = float(str(prob).strip('%')) / 100
                        target = 1.0 if outcome == true_out else 0.0
                        brier_sum += (float(prob) - target) ** 2
                        brier_count += 1

            brier = brier_sum / brier_count if brier_count > 0 else 0

            odds_cols = ODDS_MAP.get(market, {})
            has_real_odds = odds_cols and all(
                c in df.columns for c in odds_cols.values()
            )

            if has_real_odds:
                total_staked = 0
                total_return = 0.0
                for idx in actual.index:
                    pred_out = predicted.loc[idx]
                    odds_col = odds_cols.get(pred_out)
                    if odds_col and odds_col in df.columns:
                        match_odds = df.loc[idx, odds_col]
                        if pd.notna(match_odds) and match_odds > 1.0:
                            total_staked += 1
                            if pred_out == actual.loc[idx]:
                                total_return += float(match_odds)
                roi = ((total_return - total_staked) / total_staked * 100) if total_staked > 0 else 0.0
                roi_type = 'real'
            else:
                n_outcomes = len(outcomes)
                baseline = 1.0 / n_outcomes if n_outcomes > 0 else 0.5
                roi = (accuracy - baseline) / baseline * 100 if baseline > 0 else 0.0
                roi_type = 'approx'

            weight = MARKET_WEIGHTS.get(market, 1.0)

            results['markets'][market] = {
                'total': int(total),
                'correct': int(correct),
                'accuracy': float(accuracy),
                'brier_score': float(brier),
                'roi_pct': float(roi),
                'roi_type': roi_type,
                'weight': weight,
                'n_outcomes': len(outcomes),
                'source': info['source'],
            }

        return results
    
    def run(self) -> pd.DataFrame:
        """Alias for run_backtest (used by auto_tune.py)."""
        return self.run_backtest()

    def run_backtest(self) -> pd.DataFrame:
        """Run complete walk-forward backtest"""
        print("\n🔬 BACKTESTING ENGINE")
        print("="*60)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Test window: {self.test_window_days} days")
        print("Method: Walk-forward (no data leakage)")
        print("="*60)
        
        # Load full dataset (ALL historical data — needed for training)
        full_df = self.load_features_data()

        # Keep unfiltered copy for training (walk-forward needs all history)
        all_history_df = full_df

        # Test periods are restricted to the backtest window
        periods = self.get_test_periods()
        print(f"\n📅 Testing {len(periods)} periods\n")
        
        for i, (test_start, test_end) in enumerate(periods, 1):
            print(f"Period {i}/{len(periods)}: {test_start.date()} to {test_end.date()}")
            
            # Split data — train on ALL history before test_start
            train_df, test_df = self.split_data(all_history_df, test_start, test_end)
            
            # Check sufficient data
            if len(train_df) < self.min_training_matches:
                print(f"   [WARN] Insufficient training data ({len(train_df)} matches)")
                continue
            
            if len(test_df) == 0:
                print(f"   [WARN] No test matches")
                continue
            
            print(f"   📊 Train: {len(train_df)} matches | Test: {len(test_df)} matches")
            
            # Train models on training data only
            print(f"   Training models...")
            success = self.train_models_on_period(train_df, test_start=test_start)
            
            if not success:
                print(f"   [ERROR] Training failed")
                continue
            
            # Generate predictions
            print(f"   Generating predictions...")
            test_with_preds = self.generate_predictions(test_df)

            # Merge bookmaker odds for realistic ROI
            try:
                from odds_utils import merge_odds_into_df
                test_with_preds = merge_odds_into_df(test_with_preds)
            except Exception:
                pass

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
            roi_values = []
            weight = 1.0
            roi_type = 'approx'
            n_outcomes = 2
            source = 'P'

            for result in self.results:
                if market in result.get('markets', {}):
                    stats = result['markets'][market]
                    total_matches += stats['total']
                    total_correct += stats['correct']
                    brier_scores.append(stats['brier_score'])
                    roi_values.append(stats.get('roi_pct', 0.0))
                    weight = stats.get('weight', 1.0)
                    roi_type = stats.get('roi_type', 'approx')
                    n_outcomes = stats.get('n_outcomes', 2)
                    source = stats.get('source', 'P')

            if total_matches > 0:
                accuracy = total_correct / total_matches
                roi = np.mean(roi_values) if roi_values else 0.0
                baseline = 1.0 / n_outcomes if n_outcomes > 0 else 0.5
                edge = accuracy - baseline

                market_summary[market] = {
                    'Total_Matches': total_matches,
                    'Correct': total_correct,
                    'Accuracy_%': round(accuracy * 100, 1),
                    'Baseline_%': round(baseline * 100, 1),
                    'Edge_%': round(edge * 100, 1),
                    'Brier_Score': round(np.mean(brier_scores), 3),
                    'ROI_%': round(roi, 1),
                    'ROI_Type': roi_type,
                    'Weight': weight,
                    'Outcomes': n_outcomes,
                    'Source': source,
                }

        summary_df = pd.DataFrame.from_dict(market_summary, orient='index')
        if 'Accuracy_%' in summary_df.columns:
            summary_df = summary_df.sort_values('Accuracy_%', ascending=False)
        
        print("\n📊 OVERALL MARKET PERFORMANCE:")
        print(summary_df.to_string())

        # Weighted Brier (the primary tuning objective)
        if 'Brier_Score' in summary_df.columns and 'Weight' in summary_df.columns:
            weights = summary_df['Weight'].values
            briers = summary_df['Brier_Score'].values
            weighted_brier = np.average(briers, weights=weights)
            print(f"\n📈 WEIGHTED BRIER (primary metric): {weighted_brier:.4f}")
            print(f"   (lower = better calibrated; <0.20 is good, <0.15 is excellent)")

            # Check ROI types
            roi_types = summary_df['ROI_Type'].unique() if 'ROI_Type' in summary_df.columns else []
            if 'real' in roi_types:
                real_roi = summary_df[summary_df['ROI_Type'] == 'real']
                avg_real_roi = np.average(real_roi['ROI_%'], weights=real_roi['Weight'])
                print(f"   REAL ROI (odds-based): {avg_real_roi:+.1f}%")
            if 'approx' in roi_types:
                print(f"   (some markets use approximate odds — get more odds data for precision)")

        # 2. League-specific analysis
        self.analyze_by_league()
        
        # 3. Doubles/Trebles analysis
        self.analyze_combinations()
        
        # Interpretation: use Edge_% (accuracy above random baseline)
        print("\n" + "="*60)
        print("💡 KEY FINDINGS:")
        print("="*60)

        if 'Edge_%' in summary_df.columns:
            edge_sorted = summary_df.sort_values('Edge_%', ascending=False)
            strong_edge = edge_sorted[edge_sorted['Edge_%'] >= 10]
            good_edge = edge_sorted[(edge_sorted['Edge_%'] >= 5) & (edge_sorted['Edge_%'] < 10)]
            no_edge = edge_sorted[edge_sorted['Edge_%'] <= 0]

            if len(strong_edge) > 0:
                top_markets = ', '.join(f"{m} (+{edge_sorted.loc[m,'Edge_%']:.1f}%)" for m in strong_edge.index[:10])
                print(f"[OK] STRONG EDGE (≥10% above baseline): {top_markets}")

            if len(good_edge) > 0:
                mid_markets = ', '.join(f"{m} (+{edge_sorted.loc[m,'Edge_%']:.1f}%)" for m in good_edge.index[:10])
                print(f"[OK] MODERATE EDGE (5-10% above baseline): {mid_markets}")

            if len(no_edge) > 0:
                weak_markets = ', '.join(no_edge.index[:10].tolist())
                print(f"[  ] NO EDGE (at or below baseline): {weak_markets}")

            print(f"\n📈 Best edge: {edge_sorted.index[0]} (+{edge_sorted.iloc[0]['Edge_%']:.1f}% above {edge_sorted.iloc[0]['Baseline_%']:.0f}% baseline)")
        else:
            print(f"\n📈 Best accuracy: {summary_df.index[0]} ({summary_df.iloc[0]['Accuracy_%']:.1f}%)")

        n_markets = len(summary_df)
        n_blend = len(summary_df[summary_df['Source'] == 'BLEND']) if 'Source' in summary_df.columns else 0
        n_ml = n_markets - n_blend
        print(f"\n📊 Evaluated {n_markets} markets ({n_blend} blended, {n_ml} ML-only)")
        
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
        """Analyze double/treble success rates using top-edge markets"""
        print("\n" + "="*60)
        print("🎲 COMBINATION ANALYSIS (Doubles/Trebles)")
        print("="*60)

        from itertools import combinations

        market_data = {}
        for result in self.results:
            for market, stats in result.get('markets', {}).items():
                if market not in market_data:
                    market_data[market] = {'accs': [], 'n_outcomes': stats.get('n_outcomes', 2)}
                if stats['total'] > 0:
                    market_data[market]['accs'].append(stats['accuracy'])

        avg_accs = {}
        avg_odds = {}
        for m, data in market_data.items():
            if data['accs']:
                acc = np.mean(data['accs'])
                n = data['n_outcomes']
                baseline = 1.0 / n if n > 0 else 0.5
                if acc > baseline:
                    avg_accs[m] = acc
                    avg_odds[m] = 1.0 / acc if acc > 0 else 2.0

        if len(avg_accs) < 2:
            print("   Not enough markets with positive edge for combination analysis.")
            return

        top_markets = dict(sorted(avg_accs.items(), key=lambda x: x[1], reverse=True)[:20])

        print(f"\n   Using top {len(top_markets)} markets with positive edge")

        print("\n🎯 BEST DOUBLES (Top 10):")
        doubles = []
        for m1, m2 in combinations(top_markets.keys(), 2):
            combined_prob = avg_accs[m1] * avg_accs[m2]
            double_odds = avg_odds[m1] * avg_odds[m2]
            expected_roi = (combined_prob * double_odds - 1) * 100

            doubles.append({
                'combo': f"{m1} + {m2}",
                'hit_rate_%': round(combined_prob * 100, 1),
                'est_odds': round(double_odds, 2),
                'expected_roi_%': round(expected_roi, 1)
            })

        doubles_df = pd.DataFrame(doubles).sort_values('expected_roi_%', ascending=False).head(10)
        print(doubles_df.to_string(index=False))

        print("\n🎯 BEST TREBLES (Top 10):")
        trebles = []
        for m1, m2, m3 in combinations(top_markets.keys(), 3):
            combined_prob = avg_accs[m1] * avg_accs[m2] * avg_accs[m3]
            treble_odds = avg_odds[m1] * avg_odds[m2] * avg_odds[m3]
            expected_roi = (combined_prob * treble_odds - 1) * 100

            trebles.append({
                'combo': f"{m1} + {m2} + {m3}",
                'hit_rate_%': round(combined_prob * 100, 1),
                'est_odds': round(treble_odds, 2),
                'expected_roi_%': round(expected_roi, 1)
            })

        trebles_df = pd.DataFrame(trebles).sort_values('expected_roi_%', ascending=False).head(10)
        print(trebles_df.to_string(index=False))

        doubles_df.to_csv(OUTPUT_DIR / "backtest_best_doubles.csv", index=False)
        trebles_df.to_csv(OUTPUT_DIR / "backtest_best_trebles.csv", index=False)

        print("\n[OK] Saved combination analysis to outputs/backtest_best_*.csv")
    
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