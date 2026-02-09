# market_backtest.py
"""
Multi-Market Backtesting - Analyze all Tier 1+2 markets over 4-week period
Identifies which markets have the best accuracy and profitability
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys

from config import OUTPUT_DIR, DATA_DIR

# Tier 1+2 Market Configurations
MARKET_CONFIGS = {
    # TIER 1: Core Markets
    '1X2': {
        'actual_col': 'FTR',
        'pred_cols': ['P_1X2_H', 'P_1X2_D', 'P_1X2_A'],
        'outcomes': ['H', 'D', 'A'],
        'type': 'multiclass'
    },
    'BTTS': {
        'actual_col': 'y_BTTS',
        'pred_cols': ['P_BTTS_Y', 'P_BTTS_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },
    'OU_0_5': {
        'actual_col': 'y_OU_0_5',
        'pred_cols': ['P_OU_0_5_O', 'P_OU_0_5_U'],
        'outcomes': ['O', 'U'],
        'type': 'binary'
    },
    'OU_1_5': {
        'actual_col': 'y_OU_1_5',
        'pred_cols': ['P_OU_1_5_O', 'P_OU_1_5_U'],
        'outcomes': ['O', 'U'],
        'type': 'binary'
    },
    'OU_2_5': {
        'actual_col': 'y_OU_2_5',
        'pred_cols': ['P_OU_2_5_O', 'P_OU_2_5_U'],
        'outcomes': ['O', 'U'],
        'type': 'binary'
    },
    'OU_3_5': {
        'actual_col': 'y_OU_3_5',
        'pred_cols': ['P_OU_3_5_O', 'P_OU_3_5_U'],
        'outcomes': ['O', 'U'],
        'type': 'binary'
    },
    'OU_4_5': {
        'actual_col': 'y_OU_4_5',
        'pred_cols': ['P_OU_4_5_O', 'P_OU_4_5_U'],
        'outcomes': ['O', 'U'],
        'type': 'binary'
    },
    'DC_1X': {
        'actual_col': 'y_DC_1X',
        'pred_cols': ['P_DC_1X_Y', 'P_DC_1X_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },
    'DC_12': {
        'actual_col': 'y_DC_12',
        'pred_cols': ['P_DC_12_Y', 'P_DC_12_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },
    'DC_X2': {
        'actual_col': 'y_DC_X2',
        'pred_cols': ['P_DC_X2_Y', 'P_DC_X2_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },
    'DNB_H': {
        'actual_col': 'y_DNB_H',
        'pred_cols': ['P_DNB_H_Y', 'P_DNB_H_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },
    'DNB_A': {
        'actual_col': 'y_DNB_A',
        'pred_cols': ['P_DNB_A_Y', 'P_DNB_A_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },
    'HomeToScore': {
        'actual_col': 'y_HomeToScore',
        'pred_cols': ['P_HomeToScore_Y', 'P_HomeToScore_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },
    'AwayToScore': {
        'actual_col': 'y_AwayToScore',
        'pred_cols': ['P_AwayToScore_Y', 'P_AwayToScore_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },

    # TIER 2: Value Markets
    'HomeTG_0_5': {
        'actual_col': 'y_HomeTG_0_5',
        'pred_cols': ['P_HomeTG_0_5_O', 'P_HomeTG_0_5_U'],
        'outcomes': ['O', 'U'],
        'type': 'binary'
    },
    'HomeTG_1_5': {
        'actual_col': 'y_HomeTG_1_5',
        'pred_cols': ['P_HomeTG_1_5_O', 'P_HomeTG_1_5_U'],
        'outcomes': ['O', 'U'],
        'type': 'binary'
    },
    'AwayTG_0_5': {
        'actual_col': 'y_AwayTG_0_5',
        'pred_cols': ['P_AwayTG_0_5_O', 'P_AwayTG_0_5_U'],
        'outcomes': ['O', 'U'],
        'type': 'binary'
    },
    'AwayTG_1_5': {
        'actual_col': 'y_AwayTG_1_5',
        'pred_cols': ['P_AwayTG_1_5_O', 'P_AwayTG_1_5_U'],
        'outcomes': ['O', 'U'],
        'type': 'binary'
    },
    'AH_-0_5': {
        'actual_col': 'y_AH_-0_5',
        'pred_cols': ['P_AH_-0_5_H', 'P_AH_-0_5_A', 'P_AH_-0_5_P'],
        'outcomes': ['H', 'A', 'P'],
        'type': 'multiclass'
    },
    'AH_+0_5': {
        'actual_col': 'y_AH_+0_5',
        'pred_cols': ['P_AH_+0_5_H', 'P_AH_+0_5_A', 'P_AH_+0_5_P'],
        'outcomes': ['H', 'A', 'P'],
        'type': 'multiclass'
    },
    'AH_-1_0': {
        'actual_col': 'y_AH_-1_0',
        'pred_cols': ['P_AH_-1_0_H', 'P_AH_-1_0_A', 'P_AH_-1_0_P'],
        'outcomes': ['H', 'A', 'P'],
        'type': 'multiclass'
    },
    'AH_+1_0': {
        'actual_col': 'y_AH_+1_0',
        'pred_cols': ['P_AH_+1_0_H', 'P_AH_+1_0_A', 'P_AH_+1_0_P'],
        'outcomes': ['H', 'A', 'P'],
        'type': 'multiclass'
    },
    'AH_0_0': {
        'actual_col': 'y_AH_0_0',
        'pred_cols': ['P_AH_0_0_H', 'P_AH_0_0_A', 'P_AH_0_0_P'],
        'outcomes': ['H', 'A', 'P'],
        'type': 'multiclass'
    },
    'HomeWin_BTTS_Y': {
        'actual_col': 'y_HomeWin_BTTS_Y',
        'pred_cols': ['P_HomeWin_BTTS_Y_Y', 'P_HomeWin_BTTS_Y_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },
    'AwayWin_BTTS_Y': {
        'actual_col': 'y_AwayWin_BTTS_Y',
        'pred_cols': ['P_AwayWin_BTTS_Y_Y', 'P_AwayWin_BTTS_Y_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },
    'HomeWin_BTTS_N': {
        'actual_col': 'y_HomeWin_BTTS_N',
        'pred_cols': ['P_HomeWin_BTTS_N_Y', 'P_HomeWin_BTTS_N_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },
    'AwayWin_BTTS_N': {
        'actual_col': 'y_AwayWin_BTTS_N',
        'pred_cols': ['P_AwayWin_BTTS_N_Y', 'P_AwayWin_BTTS_N_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },
    'DC1X_O25': {
        'actual_col': 'y_DC1X_O25',
        'pred_cols': ['P_DC1X_O25_Y', 'P_DC1X_O25_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },
    'DCX2_O25': {
        'actual_col': 'y_DCX2_O25',
        'pred_cols': ['P_DCX2_O25_Y', 'P_DCX2_O25_N'],
        'outcomes': ['Y', 'N'],
        'type': 'binary'
    },
}


class MarketBacktester:
    """Analyze all markets using walk-forward backtest on historical data"""

    def __init__(self, weeks: int = 4, min_confidence: float = 0.90, use_historical: bool = True):
        """
        Args:
            weeks: Number of weeks to analyze
            min_confidence: Minimum confidence threshold (default 90%)
            use_historical: If True, run walk-forward backtest on historical data
                          If False, analyze recent predictions (for live tracking)
        """
        self.weeks = weeks
        self.min_confidence = min_confidence
        self.use_historical = use_historical
        self.results = []

    def load_weekly_predictions(self, week_folders: List[Path]) -> pd.DataFrame:
        """Load predictions from multiple week folders"""
        all_preds = []

        for folder in week_folders:
            csv_file = folder / "weekly_bets.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                df['Week'] = folder.name
                all_preds.append(df)
                print(f"[OK] Loaded {len(df)} predictions from {folder.name}")
            else:
                print(f"[WARN] No predictions found in {folder.name}")

        if not all_preds:
            raise FileNotFoundError("No prediction files found")

        combined = pd.concat(all_preds, ignore_index=True)
        print(f"\n[OK] Total predictions: {len(combined)} across {len(all_preds)} weeks")
        return combined

    def load_actual_results(self, start_date=None, end_date=None) -> pd.DataFrame:
        """Load actual match results from API-Football database and calculate targets"""
        from api_football_adapter import get_fixtures_from_db

        # Get all completed matches from database
        df = get_fixtures_from_db(status='FT')

        # Filter completed matches only (must have result)
        df = df[df['FTR'].notna()].copy()
        df['Date'] = pd.to_datetime(df['Date'])

        # Filter by date range if specified
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]

        # Calculate target columns from raw match data
        df = self._calculate_target_columns(df)

        print(f"[OK] Loaded {len(df)} completed matches with results")
        return df

    def _calculate_target_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all target columns (y_*) from raw match data"""
        import numpy as np

        # Total goals
        total = df['FTHG'] + df['FTAG']

        # 1X2 - already have FTR
        df['y_1X2'] = df['FTR'].astype(str)

        # BTTS
        btts = (df['FTHG'] > 0) & (df['FTAG'] > 0)
        df['y_BTTS'] = np.where(btts, 'Y', 'N')

        # Over/Under lines
        for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            line_str = str(line).replace('.', '_')
            df[f'y_OU_{line_str}'] = np.where(total > line, 'O', 'U')

        # Double Chance
        df['y_DC_1X'] = np.where(df['FTR'].isin(['H', 'D']), 'Y', 'N')
        df['y_DC_12'] = np.where(df['FTR'].isin(['H', 'A']), 'Y', 'N')
        df['y_DC_X2'] = np.where(df['FTR'].isin(['D', 'A']), 'Y', 'N')

        # Draw No Bet
        df['y_DNB_H'] = np.where(df['FTR'] == 'H', 'Y', np.where(df['FTR'] == 'D', 'P', 'N'))
        df['y_DNB_A'] = np.where(df['FTR'] == 'A', 'Y', np.where(df['FTR'] == 'D', 'P', 'N'))

        # Team to Score
        df['y_HomeToScore'] = np.where(df['FTHG'] > 0, 'Y', 'N')
        df['y_AwayToScore'] = np.where(df['FTAG'] > 0, 'Y', 'N')

        # Home Team Goals O/U
        df['y_HomeTG_0_5'] = np.where(df['FTHG'] > 0.5, 'O', 'U')
        df['y_HomeTG_1_5'] = np.where(df['FTHG'] > 1.5, 'O', 'U')

        # Away Team Goals O/U
        df['y_AwayTG_0_5'] = np.where(df['FTAG'] > 0.5, 'O', 'U')
        df['y_AwayTG_1_5'] = np.where(df['FTAG'] > 1.5, 'O', 'U')

        # Asian Handicap (simplified - just goal difference)
        goal_diff = df['FTHG'] - df['FTAG']
        df['y_AH_-0_5'] = np.where(goal_diff > 0.5, 'H', np.where(goal_diff < -0.5, 'A', 'P'))
        df['y_AH_+0_5'] = np.where(goal_diff > -0.5, 'H', np.where(goal_diff < 0.5, 'A', 'P'))
        df['y_AH_-1_0'] = np.where(goal_diff > 1, 'H', np.where(goal_diff < -1, 'A', 'P'))
        df['y_AH_+1_0'] = np.where(goal_diff > -1, 'H', np.where(goal_diff < 1, 'A', 'P'))
        df['y_AH_0_0'] = np.where(goal_diff > 0, 'H', np.where(goal_diff < 0, 'A', 'P'))

        # Result + BTTS Combinations
        df['y_HomeWin_BTTS_Y'] = np.where((df['FTR'] == 'H') & btts, 'Y', 'N')
        df['y_AwayWin_BTTS_Y'] = np.where((df['FTR'] == 'A') & btts, 'Y', 'N')
        df['y_HomeWin_BTTS_N'] = np.where((df['FTR'] == 'H') & ~btts, 'Y', 'N')
        df['y_AwayWin_BTTS_N'] = np.where((df['FTR'] == 'A') & ~btts, 'Y', 'N')

        # DC + O/U Combinations
        df['y_DC1X_O25'] = np.where((df['FTR'].isin(['H', 'D'])) & (total > 2.5), 'Y', 'N')
        df['y_DCX2_O25'] = np.where((df['FTR'].isin(['D', 'A'])) & (total > 2.5), 'Y', 'N')

        return df

    def run_historical_backtest(self) -> pd.DataFrame:
        """
        Run walk-forward backtest on historical data

        This generates predictions for historical matches and evaluates them
        against actual results we already know.
        """
        from datetime import timedelta

        print("="*60)
        print(f"WALK-FORWARD HISTORICAL BACKTEST ({self.weeks} weeks)")
        print("="*60)
        print(f"Minimum confidence: {self.min_confidence:.0%}")
        print(f"Markets analyzed: {len(MARKET_CONFIGS)}")
        print()

        # Load all historical data
        print("[1/5] Loading historical match results...")
        all_results = self.load_actual_results()

        # Get the most recent completed matches for our test period
        all_results = all_results.sort_values('Date', ascending=False)

        # Calculate date range for test period (last N weeks of completed matches)
        latest_date = all_results['Date'].max()
        test_start_date = latest_date - timedelta(weeks=self.weeks)

        print(f"[INFO] Test period: {test_start_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")

        # Get test matches
        test_matches = all_results[
            (all_results['Date'] >= test_start_date) &
            (all_results['Date'] <= latest_date)
        ].copy()

        print(f"[OK] Found {len(test_matches)} completed matches in test period")

        if len(test_matches) == 0:
            print("[ERROR] No completed matches in test period")
            return pd.DataFrame()

        # Generate predictions for these historical matches
        print("\n[2/5] Generating predictions for historical matches...")
        predictions_df = self.generate_historical_predictions(test_matches)

        if predictions_df is None or len(predictions_df) == 0:
            print("[ERROR] Failed to generate predictions")
            return pd.DataFrame()

        print(f"[OK] Generated predictions for {len(predictions_df)} matches")

        # Merge predictions with actual results
        print("\n[3/5] Merging predictions with actual results...")
        merged = self.merge_predictions_with_results(predictions_df, test_matches)

        if len(merged) == 0:
            print("[ERROR] No matches found between predictions and results")
            return pd.DataFrame()

        print(f"[OK] Matched {len(merged)} predictions with results")

        # Merge bookmaker odds for realistic ROI
        try:
            from odds_utils import merge_odds_into_df
            merged = merge_odds_into_df(merged)
        except Exception as e:
            print(f"[INFO] Odds not available for backtest: {e}")

        # Evaluate each market
        print("\n[4/5] Evaluating market performance...")
        print("="*60)

        market_results = []
        for market_name, config in MARKET_CONFIGS.items():
            result = self.evaluate_market(merged, market_name, config)
            market_results.append(result)

            if 'error' in result:
                print(f"  {market_name:20s} [SKIP] {result['error']}")
            else:
                roi_str = f"Fair: {result['roi_fair']:+6.1f}%"
                if result.get('roi_actual') is not None:
                    roi_str += f" | Actual: {result['roi_actual']:+6.1f}% ({result['odds_coverage']:.0%} coverage)"
                print(f"  {market_name:20s} [OK] {result['total_predictions']:4d} preds, {result['accuracy']:5.1%} acc, {roi_str}")

        # Convert to DataFrame
        df_results = pd.DataFrame(market_results)

        # Filter out markets with errors or no predictions
        df_results = df_results[df_results['total_predictions'] > 0].copy()

        # Sort by accuracy (descending)
        df_results = df_results.sort_values('accuracy', ascending=False)

        # Save results
        print("\n[5/5] Saving analysis...")
        from config import OUTPUT_DIR
        output_dir = OUTPUT_DIR.parent if OUTPUT_DIR.name.startswith('20') else OUTPUT_DIR

        csv_path = output_dir / "market_backtest_analysis.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"[OK] Saved analysis to {csv_path}")

        # Generate HTML report
        self.generate_html_report(df_results, output_dir)

        return df_results

    def generate_historical_predictions(self, test_matches: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for historical matches using trained models

        Args:
            test_matches: DataFrame of historical matches we want to evaluate

        Returns:
            DataFrame with predictions (P_ columns) for the test matches
        """
        import tempfile
        import os
        from predict import predict_week
        from config import OUTPUT_DIR

        # Create temporary fixtures file from historical matches
        # This is the format that predict_week expects
        temp_fixtures = test_matches[['Date', 'League', 'HomeTeam', 'AwayTeam']].copy()

        # Add Time column if it doesn't exist (set to default 15:00 for historical matches)
        if 'Time' not in temp_fixtures.columns:
            temp_fixtures['Time'] = '15:00'

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            temp_path = f.name

        # Write to CSV
        temp_fixtures.to_csv(temp_path, index=False)
        print(f"  Created temporary fixtures file with {len(temp_fixtures)} matches")

        try:
            # Generate predictions using the trained models
            print(f"  Generating predictions using trained models...")
            predict_week(temp_path)

            # Load the generated predictions
            predictions_file = OUTPUT_DIR / "weekly_bets_full.csv"
            if not predictions_file.exists():
                predictions_file = OUTPUT_DIR / "weekly_bets.csv"

            if not predictions_file.exists():
                print(f"  [ERROR] Predictions not generated at {OUTPUT_DIR}")
                return None

            predictions = pd.read_csv(predictions_file)
            predictions['Date'] = pd.to_datetime(predictions['Date'])

            print(f"  [OK] Generated predictions for {len(predictions)} matches")
            return predictions

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def merge_predictions_with_results(self,
                                       predictions: pd.DataFrame,
                                       results: pd.DataFrame) -> pd.DataFrame:
        """Merge predictions with actual results"""
        # Ensure date columns are datetime
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        results['Date'] = pd.to_datetime(results['Date'])

        # Merge on match identifiers
        merged = predictions.merge(
            results,
            on=['Date', 'League', 'HomeTeam', 'AwayTeam'],
            how='inner',
            suffixes=('_pred', '_actual')
        )

        # Preserve fixture_id for odds lookup (may come from results side as _actual suffix)
        if 'fixture_id' not in merged.columns:
            if 'fixture_id_actual' in merged.columns:
                merged['fixture_id'] = merged['fixture_id_actual']
            elif 'fixture_id_pred' in merged.columns:
                merged['fixture_id'] = merged['fixture_id_pred']

        print(f"[OK] Matched {len(merged)} predictions with actual results")
        return merged

    def evaluate_market(self, df: pd.DataFrame, market_name: str, config: dict) -> dict:
        """Evaluate single market accuracy"""
        actual_col = config['actual_col']
        pred_cols = config['pred_cols']
        outcomes = config['outcomes']

        # Check if columns exist
        missing_cols = [col for col in pred_cols if col not in df.columns]
        if missing_cols or actual_col not in df.columns:
            return {
                'market': market_name,
                'error': f'Missing columns: {missing_cols}',
                'total_predictions': 0
            }

        # Filter by minimum confidence
        df_market = df.copy()
        df_market['MaxProb'] = df_market[pred_cols].max(axis=1)
        df_market = df_market[df_market['MaxProb'] >= self.min_confidence]

        if len(df_market) == 0:
            return {
                'market': market_name,
                'total_predictions': 0,
                'error': f'No predictions above {self.min_confidence:.0%} confidence'
            }

        # Get predicted outcome (highest probability)
        df_market['PredOutcome'] = df_market[pred_cols].idxmax(axis=1)
        df_market['PredOutcome'] = df_market['PredOutcome'].str.split('_').str[-1]

        # Calculate accuracy
        df_market['Correct'] = df_market['PredOutcome'] == df_market[actual_col].astype(str)

        accuracy = df_market['Correct'].mean()
        total = len(df_market)
        correct = df_market['Correct'].sum()

        # Calculate calibration (Brier score)
        brier_scores = []
        for idx, row in df_market.iterrows():
            actual_outcome = str(row[actual_col])
            for i, outcome in enumerate(outcomes):
                predicted_prob = row[pred_cols[i]]
                actual_prob = 1.0 if outcome == actual_outcome else 0.0
                brier_scores.append((predicted_prob - actual_prob) ** 2)

        brier_score = np.mean(brier_scores) if brier_scores else np.nan

        # Calculate ROI using actual bookmaker odds where available
        roi_result = self.calculate_roi(df_market, pred_cols, actual_col, outcomes, market_name=market_name)

        # Confidence distribution
        conf_bins = {
            '60-70%': len(df_market[(df_market['MaxProb'] >= 0.60) & (df_market['MaxProb'] < 0.70)]),
            '70-80%': len(df_market[(df_market['MaxProb'] >= 0.70) & (df_market['MaxProb'] < 0.80)]),
            '80-90%': len(df_market[(df_market['MaxProb'] >= 0.80) & (df_market['MaxProb'] < 0.90)]),
            '90%+': len(df_market[df_market['MaxProb'] >= 0.90]),
        }

        return {
            'market': market_name,
            'total_predictions': total,
            'correct_predictions': int(correct),
            'accuracy': accuracy,
            'brier_score': brier_score,
            'roi_fair': roi_result['roi_fair'],
            'roi_actual': roi_result['roi_actual'],
            'odds_coverage': roi_result['odds_coverage'],
            'avg_confidence': df_market['MaxProb'].mean(),
            'confidence_distribution': conf_bins,
            'type': config['type']
        }

    def calculate_roi(self, df: pd.DataFrame, pred_cols: List[str],
                     actual_col: str, outcomes: List[str],
                     market_name: str = None) -> dict:
        """
        Calculate ROI using actual bookmaker odds where available,
        falling back to fair odds (1/probability) otherwise.

        Returns dict with roi_actual (bookmaker odds), roi_fair (model odds),
        and odds_coverage (fraction with real odds).
        """
        from odds_utils import get_odds_for_prediction

        total_stake_actual = 0
        total_return_actual = 0
        total_stake_fair = 0
        total_return_fair = 0
        bets_with_odds = 0

        for idx, row in df.iterrows():
            # Get predicted outcome and its probability
            pred_probs = [row[col] for col in pred_cols]
            pred_idx = np.argmax(pred_probs)
            pred_outcome = outcomes[pred_idx]
            pred_prob = pred_probs[pred_idx]

            if pred_prob <= 0:
                continue

            actual_outcome = str(row[actual_col])
            fair_odds = 1.0 / pred_prob

            # Look up actual bookmaker odds
            actual_odds = None
            if market_name is not None:
                actual_odds = get_odds_for_prediction(row, market_name, pred_outcome)

            # Fair odds ROI (always calculated)
            total_stake_fair += 1.0
            if pred_outcome == actual_outcome:
                total_return_fair += fair_odds

            # Actual odds ROI (when available)
            if actual_odds is not None:
                bets_with_odds += 1
                total_stake_actual += 1.0
                if pred_outcome == actual_outcome:
                    total_return_actual += actual_odds

        roi_fair = ((total_return_fair - total_stake_fair) / total_stake_fair * 100) if total_stake_fair > 0 else 0.0
        roi_actual = ((total_return_actual - total_stake_actual) / total_stake_actual * 100) if total_stake_actual > 0 else None
        coverage = bets_with_odds / total_stake_fair if total_stake_fair > 0 else 0.0

        return {
            'roi_fair': roi_fair,
            'roi_actual': roi_actual,
            'odds_coverage': coverage,
            'bets_with_odds': bets_with_odds,
        }

    def run_analysis(self, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
        """Run complete multi-market analysis"""
        print("="*60)
        print(f"MULTI-MARKET BACKTEST ANALYSIS ({self.weeks} weeks)")
        print("="*60)
        print(f"Minimum confidence: {self.min_confidence:.0%}")
        print(f"Markets analyzed: {len(MARKET_CONFIGS)}")
        print()

        # If output_dir is a dated folder, get its parent (outputs/)
        if output_dir.name.startswith('20'):
            base_output_dir = output_dir.parent
            print(f"[INFO] Using base output directory: {base_output_dir}")
        else:
            base_output_dir = output_dir

        # Find week folders (dated folders like 2026-01-02)
        try:
            week_folders = sorted([f for f in base_output_dir.iterdir() if f.is_dir() and f.name.startswith('20')])
        except Exception as e:
            print(f"[ERROR] Could not read output directory: {e}")
            return pd.DataFrame()

        if len(week_folders) == 0:
            print(f"[ERROR] No week folders found in {output_dir}")
            print(f"[INFO] Looking for folders named like: 2026-01-02")
            print(f"[INFO] Make sure predictions have been generated first")
            return pd.DataFrame()

        week_folders = week_folders[-self.weeks:]  # Last N weeks

        print(f"Analyzing weeks: {[f.name for f in week_folders]}\n")

        # Load data
        predictions = self.load_weekly_predictions(week_folders)
        results = self.load_actual_results()

        # Merge
        merged = self.merge_predictions_with_results(predictions, results)

        if len(merged) == 0:
            print("\n[ERROR] No matches found between predictions and results")
            print("[INFO] This happens when:")
            print("  - Predictions are for upcoming matches (not yet played)")
            print("  - Match dates/teams don't exactly match between predictions and database")
            print("  - No completed matches in the selected time period")
            print("\n[TIP] Wait for matches to complete, then run backtest again")
            print("[TIP] Or use older prediction folders that have completed matches")
            return pd.DataFrame()

        # Merge bookmaker odds for realistic ROI
        try:
            from odds_utils import merge_odds_into_df
            merged = merge_odds_into_df(merged)
        except Exception as e:
            print(f"[INFO] Odds not available: {e}")

        # Evaluate each market
        print("\n" + "="*60)
        print("EVALUATING MARKETS")
        print("="*60)

        market_results = []
        for market_name, config in MARKET_CONFIGS.items():
            print(f"\nAnalyzing {market_name}...", end=" ")
            result = self.evaluate_market(merged, market_name, config)
            market_results.append(result)

            if 'error' in result:
                print(f"[SKIP] {result['error']}")
            else:
                print(f"[OK] {result['total_predictions']} predictions, {result['accuracy']:.1%} accuracy")

        # Convert to DataFrame
        df_results = pd.DataFrame(market_results)

        # Filter out markets with errors or no predictions
        df_results = df_results[df_results['total_predictions'] > 0].copy()

        # Sort by accuracy (descending)
        df_results = df_results.sort_values('accuracy', ascending=False)

        # Save results to base output directory
        csv_path = base_output_dir / "market_backtest_analysis.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"\n[OK] Saved analysis to {csv_path}")

        # Generate HTML report
        self.generate_html_report(df_results, base_output_dir)

        return df_results

    def generate_html_report(self, df: pd.DataFrame, output_dir: Path):
        """Generate HTML report with visual analysis"""
        if len(df) == 0:
            return

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Multi-Market Backtest Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 20px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .acc-excellent {{ color: #22c55e; font-weight: bold; }}
        .acc-good {{ color: #3b82f6; font-weight: bold; }}
        .acc-fair {{ color: #f59e0b; }}
        .acc-poor {{ color: #ef4444; }}
        .roi-positive {{ color: #22c55e; font-weight: bold; }}
        .roi-negative {{ color: #ef4444; }}
        .rank {{
            background: #667eea;
            color: white;
            padding: 4px 8px;
            border-radius: 50%;
            font-weight: bold;
            display: inline-block;
            min-width: 30px;
            text-align: center;
        }}
        .recommendations {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .rec-good {{
            color: #22c55e;
            font-weight: bold;
        }}
        .rec-avoid {{
            color: #ef4444;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Multi-Market Backtest Analysis</h1>
        <p>Analysis Period: {self.weeks} weeks | Min Confidence: {self.min_confidence:.0%} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="summary">
        <div class="stat-card">
            <div class="stat-value">{len(df)}</div>
            <div class="stat-label">Markets Analyzed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{df['total_predictions'].sum():.0f}</div>
            <div class="stat-label">Total Predictions</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{df['accuracy'].mean():.1%}</div>
            <div class="stat-label">Average Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(df[df['accuracy'] >= 0.60])}</div>
            <div class="stat-label">Markets >= 60% Acc</div>
        </div>
    </div>

    <h2>Market Performance Ranking</h2>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Market</th>
                <th>Type</th>
                <th>Predictions</th>
                <th>Accuracy</th>
                <th>Brier Score</th>
                <th>ROI</th>
                <th>Avg Confidence</th>
            </tr>
        </thead>
        <tbody>
"""

        # Add rows
        for idx, row in df.iterrows():
            rank = list(df.index).index(idx) + 1
            market = row['market']
            market_type = row['type']
            total = int(row['total_predictions'])
            accuracy = row['accuracy']
            brier = row['brier_score']
            roi = row.get('roi_actual') if pd.notna(row.get('roi_actual')) else row.get('roi_fair', 0)
            avg_conf = row['avg_confidence']

            # Accuracy class
            if accuracy >= 0.70:
                acc_class = 'acc-excellent'
            elif accuracy >= 0.60:
                acc_class = 'acc-good'
            elif accuracy >= 0.55:
                acc_class = 'acc-fair'
            else:
                acc_class = 'acc-poor'

            # ROI class
            roi_class = 'roi-positive' if roi > 0 else 'roi-negative'

            html += f"""
            <tr>
                <td><span class="rank">{rank}</span></td>
                <td><strong>{market}</strong></td>
                <td>{market_type}</td>
                <td>{total}</td>
                <td class="{acc_class}">{accuracy:.1%}</td>
                <td>{brier:.3f}</td>
                <td class="{roi_class}">{roi:+.1f}%</td>
                <td>{avg_conf:.1%}</td>
            </tr>
"""

        html += """
        </tbody>
    </table>

    <div class="recommendations">
        <h2>Recommendations</h2>
        <h3 class="rec-good">Markets to Focus On (Accuracy >= 60%)</h3>
        <ul>
"""

        good_markets = df[df['accuracy'] >= 0.60]
        for idx, row in good_markets.iterrows():
            roi_val = row.get('roi_actual') if pd.notna(row.get('roi_actual')) else row.get('roi_fair', 0)
            html += f"<li><strong>{row['market']}</strong>: {row['accuracy']:.1%} accuracy, {int(row['total_predictions'])} predictions, ROI: {roi_val:+.1f}%</li>\n"

        html += """
        </ul>
        <h3 class="rec-avoid">Markets to Avoid (Accuracy < 55%)</h3>
        <ul>
"""

        poor_markets = df[df['accuracy'] < 0.55]
        for idx, row in poor_markets.iterrows():
            html += f"<li><strong>{row['market']}</strong>: {row['accuracy']:.1%} accuracy, {int(row['total_predictions'])} predictions</li>\n"

        html += """
        </ul>
    </div>
</body>
</html>
"""

        html_path = output_dir / "market_backtest_analysis.html"
        html_path.write_text(html, encoding='utf-8')
        print(f"[OK] Saved HTML report to {html_path}")


def run_market_backtest(weeks: int = 4, min_confidence: float = 0.90, use_historical: bool = True):
    """
    Run multi-market backtest analysis

    Args:
        weeks: Number of weeks to analyze
        min_confidence: Minimum confidence threshold (0.0-1.0)
        use_historical: If True, run walk-forward backtest on historical data (default)
                       If False, analyze recent prediction folders
    """
    backtester = MarketBacktester(weeks=weeks, min_confidence=min_confidence, use_historical=use_historical)

    if use_historical:
        # Run walk-forward backtest on historical data
        results = backtester.run_historical_backtest()
    else:
        # Analyze recent prediction folders
        results = backtester.run_analysis()

    if len(results) > 0:
        print("\n" + "="*60)
        print("TOP 10 MARKETS BY ACCURACY")
        print("="*60)
        for idx, row in results.head(10).iterrows():
            roi_val = row.get('roi_actual') if pd.notna(row.get('roi_actual')) else row.get('roi_fair', 0)
            print(f"{list(results.index).index(idx)+1:2d}. {row['market']:20s} - {row['accuracy']:.1%} ({int(row['total_predictions'])} predictions, ROI: {roi_val:+.1f}%)")

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Files generated:")
        print("  - market_backtest_analysis.csv")
        print("  - market_backtest_analysis.html")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Market Backtest Analysis')
    parser.add_argument('--weeks', type=int, default=4, help='Number of weeks to analyze (default: 4)')
    parser.add_argument('--min-confidence', type=float, default=0.90, help='Minimum confidence threshold 0.0-1.0 (default: 0.90)')
    parser.add_argument('--no-historical', action='store_true', help='Use recent predictions instead of historical backtest')
    args = parser.parse_args()

    use_historical = not args.no_historical
    run_market_backtest(weeks=args.weeks, min_confidence=args.min_confidence, use_historical=use_historical)
