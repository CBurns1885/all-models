#!/usr/bin/env python3
"""
World Cup Prediction Runner
Separate pipeline for international tournament predictions.

Loads international results data, converts to the system's format,
trains models, backtests on historical tournament data, and predicts
upcoming World Cup fixtures.

Usage:
    python run_worldcup.py --data path/to/archive
    python run_worldcup.py --data ./data_worldcup/raw --fixtures worldcup_fixtures.csv
    python run_worldcup.py --backtest-only
"""

import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path isolation: keep World Cup data/models separate from league pipeline
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
WC_DATA_DIR = BASE_DIR / "data_worldcup"
WC_PROCESSED = WC_DATA_DIR / "processed"
WC_MODELS_DIR = BASE_DIR / "models_worldcup"
WC_OUTPUT_DIR = BASE_DIR / "outputs" / "worldcup"

for d in [WC_DATA_DIR, WC_DATA_DIR / "raw", WC_DATA_DIR / "interim",
          WC_PROCESSED, WC_MODELS_DIR, WC_OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

WC_HISTORICAL = WC_PROCESSED / "historical_matches.parquet"
WC_FEATURES = WC_PROCESSED / "features.parquet"

# Patch config BEFORE importing pipeline modules
import config
_ORIG_CONFIG = {
    'DATA_DIR': config.DATA_DIR,
    'RAW_DIR': config.RAW_DIR,
    'INTERIM_DIR': config.INTERIM_DIR,
    'PROCESSED_DIR': config.PROCESSED_DIR,
    'FEATURES_PARQUET': config.FEATURES_PARQUET,
    'HISTORICAL_PARQUET': config.HISTORICAL_PARQUET,
    'MODEL_ARTIFACTS_DIR': config.MODEL_ARTIFACTS_DIR,
    'MODELS_DIR': config.MODELS_DIR,
    'OUTPUT_DIR': config.OUTPUT_DIR,
    'BLEND_WEIGHTS_JSON': config.BLEND_WEIGHTS_JSON,
}

def _patch_config():
    """Redirect all paths to World Cup directories."""
    config.DATA_DIR = WC_DATA_DIR
    config.RAW_DIR = WC_DATA_DIR / "raw"
    config.INTERIM_DIR = WC_DATA_DIR / "interim"
    config.PROCESSED_DIR = WC_PROCESSED
    config.FEATURES_PARQUET = WC_FEATURES
    config.HISTORICAL_PARQUET = WC_HISTORICAL
    config.MODEL_ARTIFACTS_DIR = WC_MODELS_DIR
    config.MODELS_DIR = WC_MODELS_DIR
    config.OUTPUT_DIR = WC_OUTPUT_DIR
    config.BLEND_WEIGHTS_JSON = WC_MODELS_DIR / "blend_weights.json"

# Thorough training settings for overnight runs
import os
os.environ['OPTUNA_TRIALS'] = os.environ.get('OPTUNA_TRIALS', '50')
os.environ['N_ESTIMATORS'] = os.environ.get('N_ESTIMATORS', '500')
os.environ['FORCE_RETRAIN'] = '1'
config.OPTUNA_TRIALS = int(os.environ['OPTUNA_TRIALS'])
config.N_ESTIMATORS = int(os.environ['N_ESTIMATORS'])
config.FORCE_RETRAIN = True

_patch_config()

# NOW import pipeline modules — they'll pick up config values at import time
import features as features_mod
import models as models_mod
import predict as predict_mod
import blending as blending_mod
from backtest import BacktestEngine

# Patch module-level bindings that were copied at import via `from config import`
def _patch_modules():
    """Override local name bindings in modules that used `from config import`."""
    patch_map = {
        'FEATURES_PARQUET': WC_FEATURES,
        'HISTORICAL_PARQUET': WC_HISTORICAL,
        'DATA_DIR': WC_DATA_DIR,
        'PROCESSED_DIR': WC_PROCESSED,
        'MODEL_ARTIFACTS_DIR': WC_MODELS_DIR,
        'MODELS_DIR': WC_MODELS_DIR,
        'OUTPUT_DIR': WC_OUTPUT_DIR,
    }
    for mod in [features_mod, models_mod, predict_mod, blending_mod]:
        for attr, val in patch_map.items():
            if hasattr(mod, attr):
                setattr(mod, attr, val)
    # Blend weights JSON lives in WC models dir
    if hasattr(blending_mod, 'BLEND_WEIGHTS_JSON'):
        blending_mod.BLEND_WEIGHTS_JSON = WC_MODELS_DIR / "blend_weights.json"
    if hasattr(predict_mod, 'BLEND_WEIGHTS_JSON'):
        predict_mod.BLEND_WEIGHTS_JSON = WC_MODELS_DIR / "blend_weights.json"

_patch_modules()

# ---------------------------------------------------------------------------
# International-specific configuration
# ---------------------------------------------------------------------------

INTL_LEAGUE_PROFILE = {
    'quality': 'elite',
    'home_advantage': 0.05,
    'draw_tendency': 0.22,
}

# Tournaments to treat as competitive (vs friendlies)
COMPETITIVE_TOURNAMENTS = {
    'FIFA World Cup', 'FIFA World Cup qualification',
    'UEFA Euro', 'UEFA Euro qualification',
    'Copa América', 'Copa America',
    'African Cup of Nations', 'AFC Asian Cup',
    'CONCACAF Gold Cup', 'UEFA Nations League',
    'Confederations Cup', 'African Nations Championship',
    'AFF Championship', 'CONCACAF Nations League',
}

# Minimum year for training data — 2022 captures the last World Cup cycle
# Use --min-year 2020 if you want even more (adds noise but more data)
MIN_YEAR = 2022


# ============================================================================
# Data loading & conversion
# ============================================================================

def load_international_data(archive_path: str, min_year: int = MIN_YEAR) -> pd.DataFrame:
    """
    Load Kaggle-format international results and convert to system format.

    Expected CSV columns: date, home_team, away_team, home_score, away_score,
                          tournament, city, country, neutral
    """
    archive = Path(archive_path)

    csv_path = None
    for name in ['results.csv', 'international_results.csv', 'matches.csv']:
        candidate = archive / name
        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        csvs = sorted(archive.glob('*.csv'))
        if csvs:
            csv_path = csvs[0]
        else:
            raise FileNotFoundError(
                f"No CSV found in {archive_path}. "
                "Expected results.csv from Kaggle international football dataset."
            )

    print(f"\n Loading international results from {csv_path}")
    raw = pd.read_csv(csv_path)
    print(f"   Raw rows: {len(raw):,}")

    # Normalise column names
    col_map = {}
    for col in raw.columns:
        key = col.strip().lower().replace(' ', '_')
        mapping = {
            'date': 'Date',
            'home_team': 'HomeTeam',
            'away_team': 'AwayTeam',
            'home_score': 'FTHG',
            'away_score': 'FTAG',
            'tournament': 'Tournament',
            'neutral': 'Neutral',
            'city': 'City',
            'country': 'Country',
        }
        if key in mapping:
            col_map[col] = mapping[key]
    raw = raw.rename(columns=col_map)

    required = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {list(raw.columns)}"
        )

    # Parse dates and filter
    raw['Date'] = pd.to_datetime(raw['Date'], dayfirst=True, errors='coerce')
    raw = raw.dropna(subset=['Date', 'FTHG', 'FTAG'])
    raw['FTHG'] = raw['FTHG'].astype(int)
    raw['FTAG'] = raw['FTAG'].astype(int)

    # Filter to recent years
    raw = raw[raw['Date'].dt.year >= min_year].copy()
    print(f"   After filtering to {min_year}+: {len(raw):,} matches")

    # Derive full-time result
    raw['FTR'] = np.where(
        raw['FTHG'] > raw['FTAG'], 'H',
        np.where(raw['FTHG'] < raw['FTAG'], 'A', 'D')
    )

    # Use tournament as League column (grouping for Elo/form)
    if 'Tournament' in raw.columns:
        raw['League'] = raw['Tournament'].fillna('International')
    else:
        raw['League'] = 'International'

    # Tag competitive vs friendly
    if 'Tournament' in raw.columns:
        raw['is_competitive'] = raw['Tournament'].isin(COMPETITIVE_TOURNAMENTS)
    else:
        raw['is_competitive'] = True

    # Handle neutral venue flag
    if 'Neutral' in raw.columns:
        raw['Neutral'] = raw['Neutral'].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
    else:
        raw['Neutral'] = False

    # No half-time data in Kaggle set — set to NaN
    raw['HTHG'] = np.nan
    raw['HTAG'] = np.nan
    raw['HTR'] = np.nan

    # Synthetic odds — better than nothing for ROI approximation
    # Use slightly home-biased for non-neutral, even for neutral
    raw['B365H'] = np.where(raw['Neutral'], 2.70, 2.40)
    raw['B365D'] = 3.20
    raw['B365A'] = np.where(raw['Neutral'], 2.70, 3.10)

    # Pinnacle columns (used by some features) — mirror B365
    raw['PSCH'] = raw['B365H']
    raw['PSCD'] = raw['B365D']
    raw['PSCA'] = raw['B365A']

    # Average/Max odds columns
    raw['AvgH'] = raw['B365H']
    raw['AvgD'] = raw['B365D']
    raw['AvgA'] = raw['B365A']
    raw['MaxH'] = raw['B365H']
    raw['MaxD'] = raw['B365D']
    raw['MaxA'] = raw['B365A']

    # Shots / cards / corners — not available, set to NaN
    for col in ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
                'HY', 'AY', 'HR', 'AR']:
        raw[col] = np.nan

    # Sort by date
    raw = raw.sort_values('Date').reset_index(drop=True)

    # Print summary
    n_comp = raw['is_competitive'].sum()
    n_friendly = len(raw) - n_comp
    n_neutral = raw['Neutral'].sum()
    print(f"   Competitive: {n_comp:,} | Friendlies: {n_friendly:,}")
    print(f"   Neutral venue: {n_neutral:,} ({n_neutral/len(raw)*100:.0f}%)")
    print(f"   Teams: {raw['HomeTeam'].nunique()}")
    print(f"   Date range: {raw['Date'].min().date()} to {raw['Date'].max().date()}")

    if 'Tournament' in raw.columns:
        wc_matches = raw[raw['Tournament'].str.contains('World Cup', case=False, na=False)]
        print(f"   World Cup matches: {len(wc_matches):,}")

    return raw


def save_historical(df: pd.DataFrame):
    """Save converted data as historical_matches.parquet."""
    print(f"\n Saving {len(df):,} matches to {WC_HISTORICAL}")
    df.to_parquet(WC_HISTORICAL, index=False)


# ============================================================================
# Pipeline steps
# ============================================================================

def step_build_features():
    """Build features from international historical data."""
    print("\n" + "="*60)
    print("STEP 2: Building features")
    print("="*60)

    features_mod.build_features(force=True)
    print(f"   Features saved to {WC_FEATURES}")


def step_train_models():
    """Train models on international data."""
    print("\n" + "="*60)
    print("STEP 3: Training models on international data")
    print("="*60)

    models = models_mod.train_all_targets(WC_MODELS_DIR)
    print(f"   Trained {len(models)} models")
    return models


def step_learn_blend_weights(df: pd.DataFrame):
    """Learn blend weights using temporal validation."""
    print("\n" + "="*60)
    print("STEP 4: Learning blend weights")
    print("="*60)

    try:
        latest_date = df['Date'].max()
        val_end = latest_date.strftime('%Y-%m-%d')
        blending_mod.learn_blend_weights_temporal(val_end)
        print("   Blend weights learned successfully")
    except Exception as e:
        print(f"   [WARN] Blend weight learning failed: {e}")
        print("   Falling back to default weights")
        try:
            blending_mod.learn_blend_weights()
        except Exception as e2:
            print(f"   [WARN] Fallback also failed: {e2}")


def step_backtest(backtest_months: int = 18):
    """
    Backtest on recent international matches.
    Uses 14-day windows for granular results; min 50 training matches.
    """
    print("\n" + "="*60)
    print(f"STEP 5: Backtesting (last {backtest_months} months)")
    print("="*60)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=backtest_months * 30)

    engine = BacktestEngine(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        test_window_days=14,
        min_training_matches=30,
    )

    # Patch engine's config references
    engine_config_attrs = ['DATA_DIR', 'OUTPUT_DIR', 'FEATURES_PARQUET', 'MODEL_ARTIFACTS_DIR']
    # BacktestEngine uses module-level imports from config, already patched

    summary_df = engine.run_backtest()
    engine.export_detailed_results()

    return summary_df


def step_predict(fixtures_path: str):
    """Generate predictions for upcoming World Cup fixtures."""
    print("\n" + "="*60)
    print("STEP 6: Generating World Cup predictions")
    print("="*60)

    fixtures = Path(fixtures_path)
    if not fixtures.exists():
        print(f"   [ERROR] Fixtures file not found: {fixtures}")
        print("   Create a CSV with columns: Date, League, HomeTeam, AwayTeam")
        print("   Example:")
        print("   Date,League,HomeTeam,AwayTeam")
        print("   2026-06-22,FIFA World Cup,Brazil,Germany")
        return

    print(f"   Loading fixtures from {fixtures}")
    fixtures_df = pd.read_csv(fixtures)
    print(f"   {len(fixtures_df)} matches to predict")

    # Set League to World Cup if not specified
    if 'League' not in fixtures_df.columns:
        fixtures_df['League'] = 'FIFA World Cup'

    # Save temp fixtures for predict_week
    temp_fixtures = WC_OUTPUT_DIR / "worldcup_fixtures.csv"
    fixtures_df.to_csv(temp_fixtures, index=False)

    predict_mod.predict_week(temp_fixtures)

    # Copy results
    predictions_file = WC_OUTPUT_DIR / "weekly_bets.csv"
    if predictions_file.exists():
        print(f"\n   Predictions saved to {predictions_file}")
        preds = pd.read_csv(predictions_file)
        _print_predictions_summary(preds)
    else:
        print("   [WARN] No predictions generated")


def _print_predictions_summary(df: pd.DataFrame):
    """Print a clean summary of World Cup predictions."""
    print("\n" + "="*60)
    print(" WORLD CUP PREDICTIONS")
    print("="*60)

    blend_cols_1x2 = ['BLEND_1X2_H', 'BLEND_1X2_D', 'BLEND_1X2_A']
    blend_btts = ['BLEND_BTTS_Y', 'BLEND_BTTS_N']
    blend_ou = ['BLEND_OU_2_5_O', 'BLEND_OU_2_5_U']

    for _, row in df.iterrows():
        home = row.get('HomeTeam', '?')
        away = row.get('AwayTeam', '?')
        date = row.get('Date', '?')

        print(f"\n   {home} vs {away}  ({date})")

        # 1X2
        if all(c in df.columns for c in blend_cols_1x2):
            h = row.get('BLEND_1X2_H', 0)
            d = row.get('BLEND_1X2_D', 0)
            a = row.get('BLEND_1X2_A', 0)
            if pd.notna(h):
                pick = 'HOME' if h > max(d, a) else ('DRAW' if d > a else 'AWAY')
                print(f"      1X2:  H={h:.0%}  D={d:.0%}  A={a:.0%}  -> {pick}")

        # BTTS
        if all(c in df.columns for c in blend_btts):
            y = row.get('BLEND_BTTS_Y', 0)
            n = row.get('BLEND_BTTS_N', 0)
            if pd.notna(y):
                print(f"      BTTS: Y={y:.0%}  N={n:.0%}  -> {'YES' if y > n else 'NO'}")

        # OU 2.5
        if all(c in df.columns for c in blend_ou):
            o = row.get('BLEND_OU_2_5_O', 0)
            u = row.get('BLEND_OU_2_5_U', 0)
            if pd.notna(o):
                print(f"      O/U:  O={o:.0%}  U={u:.0%}  -> {'OVER 2.5' if o > u else 'UNDER 2.5'}")

    print()


# ============================================================================
# Main orchestrator
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="World Cup Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full overnight run (load data, train, backtest):
    python run_worldcup.py --data "C:\\Users\\Chris\\Downloads\\archive"

  Include only competitive matches (no friendlies):
    python run_worldcup.py --data "C:\\Users\\Chris\\Downloads\\archive" --competitive-only

  Backtest only (after initial training):
    python run_worldcup.py --backtest-only

  Predict specific World Cup fixtures:
    python run_worldcup.py --predict-only --fixtures worldcup_fixtures.csv

  Full pipeline with predictions:
    python run_worldcup.py --data "C:\\Users\\Chris\\Downloads\\archive" --fixtures wc_fixtures.csv
        """
    )

    parser.add_argument(
        '--data', type=str,
        help='Path to folder containing results.csv (Kaggle international results)'
    )
    parser.add_argument(
        '--fixtures', type=str,
        help='CSV with upcoming fixtures (Date, HomeTeam, AwayTeam)'
    )
    parser.add_argument(
        '--backtest-only', action='store_true',
        help='Skip training, run backtest with existing models'
    )
    parser.add_argument(
        '--predict-only', action='store_true',
        help='Skip training/backtest, just predict fixtures'
    )
    parser.add_argument(
        '--backtest-months', type=int, default=12,
        help='Months of history to backtest (default: 12)'
    )
    parser.add_argument(
        '--min-year', type=int, default=MIN_YEAR,
        help=f'Earliest year to include in training (default: {MIN_YEAR})'
    )
    parser.add_argument(
        '--competitive-only', action='store_true',
        help='Exclude friendlies from training data'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print(" WORLD CUP PREDICTION PIPELINE")
    print("="*60)
    print(f"   Data dir:    {WC_DATA_DIR}")
    print(f"   Models dir:  {WC_MODELS_DIR}")
    print(f"   Output dir:  {WC_OUTPUT_DIR}")
    print("="*60)

    # ------------------------------------------------------------------
    # Predict-only mode
    # ------------------------------------------------------------------
    if args.predict_only:
        if not args.fixtures:
            print("[ERROR] --predict-only requires --fixtures")
            sys.exit(1)
        if not WC_FEATURES.exists():
            print("[ERROR] No trained models found. Run full pipeline first.")
            sys.exit(1)
        step_predict(args.fixtures)
        return

    # ------------------------------------------------------------------
    # Backtest-only mode
    # ------------------------------------------------------------------
    if args.backtest_only:
        if not WC_FEATURES.exists():
            print("[ERROR] No features found. Run full pipeline first.")
            sys.exit(1)
        step_backtest(args.backtest_months)
        return

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    if not args.data:
        # Check if data already exists
        if WC_HISTORICAL.exists():
            print("   Using existing international data (pass --data to reload)")
            df = pd.read_parquet(WC_HISTORICAL)
        else:
            print("[ERROR] No data. Use --data to point to folder with results.csv")
            print("   Download from: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017")
            sys.exit(1)
    else:
        # STEP 1: Load and convert data
        print("\n" + "="*60)
        print("STEP 1: Loading international results")
        print("="*60)

        df = load_international_data(args.data, min_year=args.min_year)

        if args.competitive_only:
            before = len(df)
            df = df[df['is_competitive']].copy()
            print(f"   Filtered to competitive only: {before:,} -> {len(df):,}")

        save_historical(df)

    # STEP 2: Build features
    step_build_features()

    # STEP 3: Train models
    step_train_models()

    # STEP 4: Learn blend weights
    df_hist = pd.read_parquet(WC_HISTORICAL)
    step_learn_blend_weights(df_hist)

    # STEP 5: Backtest
    step_backtest(args.backtest_months)

    # STEP 6: Predict (if fixtures provided)
    if args.fixtures:
        step_predict(args.fixtures)
    else:
        print("\n" + "-"*60)
        print("   No --fixtures provided. To predict upcoming matches:")
        print(f"   python run_worldcup.py --fixtures worldcup_fixtures.csv")
        print()
        print("   Create a CSV like:")
        print("   Date,League,HomeTeam,AwayTeam")
        print("   2026-06-22,FIFA World Cup,Brazil,Germany")
        print("   2026-06-23,FIFA World Cup,Argentina,France")
        print("-"*60)

    print("\n" + "="*60)
    print(" PIPELINE COMPLETE")
    print("="*60)
    print(f"   Models:     {WC_MODELS_DIR}")
    print(f"   Backtest:   {WC_OUTPUT_DIR / 'backtest_summary.csv'}")
    print(f"   Detailed:   {WC_OUTPUT_DIR / 'backtest_detailed.csv'}")
    if args.fixtures:
        print(f"   Predictions: {WC_OUTPUT_DIR / 'weekly_bets.csv'}")
    print("="*60)


if __name__ == "__main__":
    main()
