#!/usr/bin/env python3
"""
RUN_ALL.py - Complete Football Prediction Pipeline
==================================================
Master script that runs the entire pipeline:
1. Update API database with latest match data
2. Update results for accuracy tracking
3. Run predictions for upcoming matches
4. Generate all reports and analysis

Usage:
    python run_all.py                    # Interactive mode (recommended)
    python run_all.py --quick            # Quick mode (fast predictions only)
    python run_all.py --full             # Full mode (all models, best accuracy)
    python run_all.py --skip-update      # Skip API update (use existing data)
    python run_all.py --predictions-only # Only run predictions (skip updates)
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import time

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(
    description='Complete Football Prediction Pipeline',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python run_all.py                  # Full pipeline with prompts
  python run_all.py --quick          # Quick predictions (~5 min)
  python run_all.py --full           # Maximum accuracy (~2+ hours)
  python run_all.py --skip-update    # Skip API update step
  python run_all.py --predictions-only  # Only predictions, no updates
  python run_all.py -y               # Non-interactive mode
"""
)

parser.add_argument('--quick', action='store_true',
                    help='Quick mode: fast predictions with single model')
parser.add_argument('--full', action='store_true',
                    help='Full mode: all models, maximum accuracy')
parser.add_argument('--skip-update', action='store_true',
                    help='Skip API database update')
parser.add_argument('--skip-results', action='store_true',
                    help='Skip results update')
parser.add_argument('--predictions-only', action='store_true',
                    help='Only run predictions (skip all updates)')
parser.add_argument('--backtest', action='store_true',
                    help='Run backtest after predictions')
parser.add_argument('-y', '--non-interactive', action='store_true',
                    help='Non-interactive mode (no prompts)')
parser.add_argument('--days-ahead', type=int, default=7,
                    help='Days ahead to fetch fixtures (default: 7)')
parser.add_argument('--seasons', nargs='+', type=int, default=None,
                    help='Seasons to update (e.g., 2024 2025)')

args = parser.parse_args()

# Derive speed mode from flags
if args.quick:
    speed_mode = 'fast'
elif args.full:
    speed_mode = 'full'
else:
    speed_mode = None  # Will prompt user

# Set predictions-only mode
if args.predictions_only:
    args.skip_update = True
    args.skip_results = True

# ============================================================================
# HEADER
# ============================================================================

def print_header():
    """Print startup header"""
    print("\n" + "=" * 70)
    print("  FOOTBALL PREDICTION SYSTEM - COMPLETE PIPELINE")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'Non-interactive' if args.non_interactive else 'Interactive'}")
    print("=" * 70)

def print_step(step_num, total, description):
    """Print step header"""
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}/{total}: {description}")
    print(f"{'='*70}\n")

def print_success(message):
    """Print success message"""
    print(f"[OK] {message}")

def print_warning(message):
    """Print warning message"""
    print(f"[WARN] {message}")

def print_error(message):
    """Print error message"""
    print(f"[ERROR] {message}")

# ============================================================================
# STEP 1: UPDATE API DATABASE
# ============================================================================

def step1_update_api_database():
    """Update API-Football database with latest match data"""
    print_step(1, 5, "UPDATE API DATABASE")

    if args.skip_update:
        print("[SKIP] Skipping API database update (--skip-update)")
        return True

    try:
        from api_client import (
            test_api_connection,
            get_database_stats,
            populate_historical_data,
            download_missing_leagues
        )
        from config import LEAGUE_CODES, API_LEAGUE_MAP

        # Test connection
        print("[TEST] Testing API-Football connection...")
        if not test_api_connection():
            print_warning("API connection failed - using existing data")
            return True  # Continue with existing data

        # Check database status
        print("\n[CHECK] Checking database status...")
        stats = get_database_stats()

        if stats.get('exists'):
            print(f"   Total fixtures: {stats['total_fixtures']}")
            print(f"   Completed matches: {stats['completed_fixtures']}")
            print(f"   Leagues: {len(stats['leagues'])}")
            print(f"   Date range: {stats['date_range']}")
        else:
            print("   Database not found - will be created")

        # Determine seasons to update
        current_year = datetime.now().year
        current_month = datetime.now().month
        current_season = current_year if current_month >= 7 else current_year - 1

        if args.seasons:
            seasons = args.seasons
        else:
            # Update current and previous season
            seasons = [current_season - 1, current_season]

        print(f"\n[UPDATE] Updating seasons: {seasons}")

        # Check for missing leagues
        existing_leagues = set(stats.get('leagues', []))
        configured_leagues = set(lg for lg in LEAGUE_CODES if lg in API_LEAGUE_MAP)
        missing_leagues = configured_leagues - existing_leagues

        if missing_leagues:
            print(f"\n[DOWNLOAD] Downloading missing leagues: {sorted(missing_leagues)}")
            download_missing_leagues(seasons=seasons)

        # Update existing leagues with recent data
        print(f"\n[UPDATE] Updating league data...")
        total = populate_historical_data(
            leagues=list(configured_leagues),
            seasons=seasons,
            fetch_stats=False  # Skip detailed stats for speed
        )

        print_success(f"Database updated with {total} fixtures")
        return True

    except ImportError as e:
        print_warning(f"API client not available: {e}")
        print("   Continuing with existing data...")
        return True
    except Exception as e:
        print_error(f"API update failed: {e}")
        print("   Continuing with existing data...")
        return True

# ============================================================================
# STEP 2: UPDATE RESULTS
# ============================================================================

def step2_update_results():
    """Update accuracy database with latest match results"""
    print_step(2, 5, "UPDATE RESULTS DATABASE")

    if args.skip_results:
        print("[SKIP] Skipping results update (--skip-results)")
        return True

    try:
        from update_results import (
            fetch_latest_results,
            prepare_results_for_update,
            check_pending_predictions
        )
        from accuracy_tracker import AccuracyTracker

        # Show pending predictions
        print("[CHECK] Checking pending predictions...")
        check_pending_predictions()

        # Fetch latest results
        print("\n[FETCH] Fetching latest results...")
        results_df = fetch_latest_results(days_back=60)

        if results_df.empty:
            print_warning("No results found - skipping update")
            return True

        print(f"   Found {len(results_df)} matches")

        # Prepare for update
        print("\n[PREPARE] Preparing results for update...")
        prepared_df = prepare_results_for_update(results_df)
        print(f"   Prepared {len(prepared_df)} matches with all market outcomes")

        # Update database
        print("\n[UPDATE] Updating accuracy database...")
        tracker = AccuracyTracker()
        tracker.update_results(prepared_df)

        # Calculate accuracy for recent weeks
        unique_weeks = prepared_df['Date'].apply(lambda x: x.strftime('%Y-W%W')).unique()
        for week in unique_weeks[-4:]:  # Last 4 weeks
            tracker.calculate_weekly_accuracy(week)

        print_success("Results database updated")
        return True

    except ImportError as e:
        print_warning(f"Results updater not available: {e}")
        return True
    except Exception as e:
        print_error(f"Results update failed: {e}")
        import traceback
        traceback.print_exc()
        return True  # Continue with predictions

# ============================================================================
# STEP 3: RUN PREDICTIONS
# ============================================================================

def step3_run_predictions(speed: str = 'balanced'):
    """Run the main prediction pipeline"""
    print_step(3, 5, "GENERATE PREDICTIONS")

    # Set environment variables for run_weekly.py
    os.environ["SPEED_MODE"] = speed
    os.environ["USE_API_FOOTBALL"] = "1"
    os.environ["USE_XG_FEATURES"] = "1"

    # Map speed to estimators
    estimator_counts = {"fast": "100", "balanced": "150", "full": "300"}
    os.environ["N_ESTIMATORS"] = estimator_counts.get(speed, "150")

    # No tuning for faster runs
    os.environ["OPTUNA_TRIALS"] = "0"

    print(f"[CONFIG] Speed mode: {speed}")
    print(f"[CONFIG] Estimators: {os.environ['N_ESTIMATORS']}")

    try:
        # Import and run the prediction pipeline
        from data_ingest import build_historical_results
        from features import build_features
        from predict import predict_week
        from config import OUTPUT_DIR

        # Step 3a: Build historical database
        print("\n[3a] Building historical database...")
        build_historical_results(force=False)
        print_success("Historical database ready")

        # Step 3b: Build features
        print("\n[3b] Building features...")
        build_features(force=False)
        print_success("Features ready")

        # Step 3c: Train/load models
        print("\n[3c] Loading/training models...")
        from incremental_trainer import smart_train_or_load
        models = smart_train_or_load()
        print_success(f"Models ready ({len(models) if models else 0} models)")

        # Step 3d: Download fixtures
        print("\n[3d] Downloading upcoming fixtures...")
        try:
            from api_client import download_upcoming_fixtures, fetch_live_data_for_upcoming
            from config import LEAGUE_CODES

            fixtures_df = download_upcoming_fixtures(LEAGUE_CODES, days_ahead=args.days_ahead)

            if not fixtures_df.empty:
                # Fetch live data (injuries)
                print("   Fetching injuries and lineups...")
                fixtures_df = fetch_live_data_for_upcoming(fixtures_df)

                fixtures_path = OUTPUT_DIR / "upcoming_fixtures.csv"
                fixtures_df.to_csv(fixtures_path, index=False)
                print_success(f"Downloaded {len(fixtures_df)} fixtures")
            else:
                fixtures_path = OUTPUT_DIR / "upcoming_fixtures.csv"
                if not fixtures_path.exists():
                    print_error("No fixtures available!")
                    return False
        except Exception as e:
            print_warning(f"Could not download fixtures: {e}")
            fixtures_path = OUTPUT_DIR / "upcoming_fixtures.csv"

        # Step 3e: Generate predictions
        print("\n[3e] Generating predictions...")
        predict_week(fixtures_path)
        print_success("Predictions generated")

        return True

    except Exception as e:
        print_error(f"Prediction pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# STEP 4: GENERATE REPORTS
# ============================================================================

def step4_generate_reports():
    """Generate analysis reports"""
    print_step(4, 5, "GENERATE REPORTS")

    from config import OUTPUT_DIR

    errors = []

    # 4a: Weighted Top 50
    print("[4a] Generating weighted Top 50...")
    try:
        from weighted_top50 import generate_weighted_top50
        csv_path = OUTPUT_DIR / "weekly_bets.csv"
        if csv_path.exists():
            generate_weighted_top50(csv_path)
            print_success("Top 50 generated")
        else:
            print_warning("weekly_bets.csv not found")
    except Exception as e:
        errors.append(f"Top 50: {e}")
        print_warning(f"Top 50 failed: {e}")

    # 4b: O/U Analysis
    print("\n[4b] Generating O/U analysis...")
    try:
        from ou_analyzer import analyze_ou_predictions
        df_ou = analyze_ou_predictions(min_confidence=0.90)
        if df_ou is not None and not df_ou.empty:
            print_success(f"O/U analysis: {len(df_ou)} predictions")
        else:
            print("[INFO] No high-confidence O/U predictions")
    except Exception as e:
        errors.append(f"O/U: {e}")
        print_warning(f"O/U analysis failed: {e}")

    # 4c: Accumulators
    print("\n[4c] Building accumulators...")
    try:
        from acc_builder import AccumulatorBuilder
        csv_path = OUTPUT_DIR / "weekly_bets.csv"

        if csv_path.exists():
            builder = AccumulatorBuilder(str(csv_path))

            strategies = {
                'safe': ('Conservative 4-Fold', 4),
                'mixed': ('Balanced 5-Fold', 5),
                'aggressive': ('High-Risk 6-Fold', 6)
            }

            for strategy_name, (display_name, num_legs) in strategies.items():
                acca_html = builder.generate_report(strategy=strategy_name, num_legs=num_legs)
                acca_path = OUTPUT_DIR / f"accumulators_{strategy_name}.html"
                with open(acca_path, 'w', encoding='utf-8') as f:
                    f.write(acca_html)

            print_success("Accumulators generated")
    except Exception as e:
        errors.append(f"Accumulators: {e}")
        print_warning(f"Accumulators failed: {e}")

    # 4d: Market split
    print("\n[4d] Splitting by market...")
    try:
        from market_splitter import split_predictions
        csv_path = OUTPUT_DIR / "weekly_bets.csv"
        if csv_path.exists():
            split_predictions(csv_path, OUTPUT_DIR)
            print_success("Market files generated")
    except Exception as e:
        errors.append(f"Market split: {e}")
        print_warning(f"Market split failed: {e}")

    # 4e: Log predictions
    print("\n[4e] Logging predictions...")
    try:
        from accuracy_tracker import log_weekly_predictions
        import datetime as dt

        week_id = dt.datetime.now().strftime('%Y-W%W')
        csv_path = OUTPUT_DIR / "weekly_bets.csv"

        if csv_path.exists():
            log_weekly_predictions(csv_path, week_id)
            print_success(f"Predictions logged (Week {week_id})")
    except Exception as e:
        errors.append(f"Logging: {e}")
        print_warning(f"Logging failed: {e}")

    if errors:
        print(f"\n[WARN] {len(errors)} report(s) had issues")
    else:
        print_success("All reports generated successfully")

    return True

# ============================================================================
# STEP 5: BACKTEST (OPTIONAL)
# ============================================================================

def step5_run_backtest():
    """Run backtest on recent predictions"""
    print_step(5, 5, "BACKTEST RESULTS")

    if not args.backtest:
        print("[SKIP] Backtest not requested (use --backtest to enable)")
        return True

    try:
        from backtest_results import main as run_backtest
        run_backtest()
        return True
    except Exception as e:
        print_warning(f"Backtest failed: {e}")
        return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete pipeline"""
    global speed_mode

    print_header()

    # Interactive speed selection if not specified
    if speed_mode is None and not args.non_interactive:
        print("\n  Select prediction speed:")
        print("    1. FAST     (~5-10 min)  - Single model, quick results")
        print("    2. BALANCED (~20-30 min) - RF + LightGBM, good accuracy")
        print("    3. FULL     (~2-3 hours) - All models, best accuracy")
        print()

        while True:
            choice = input("  Enter choice (1-3, default=2): ").strip() or "2"
            if choice == "1":
                speed_mode = "fast"
                break
            elif choice == "2":
                speed_mode = "balanced"
                break
            elif choice == "3":
                speed_mode = "full"
                break
            print("  Please enter 1, 2, or 3")

    if speed_mode is None:
        speed_mode = "balanced"

    print(f"\n[CONFIG] Running with speed mode: {speed_mode}")

    # Track timing
    start_time = time.time()

    # Run pipeline
    success = True

    # Step 1: Update API database
    if not step1_update_api_database():
        print_warning("API update had issues - continuing...")

    # Step 2: Update results
    if not step2_update_results():
        print_warning("Results update had issues - continuing...")

    # Step 3: Run predictions
    if not step3_run_predictions(speed_mode):
        print_error("Predictions failed!")
        success = False

    # Step 4: Generate reports
    if success:
        step4_generate_reports()

    # Step 5: Backtest (optional)
    step5_run_backtest()

    # Final summary
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 70)
    if success:
        print("  PIPELINE COMPLETE!")
    else:
        print("  PIPELINE COMPLETED WITH ERRORS")
    print("=" * 70)
    print(f"  Total time: {minutes}m {seconds}s")

    from config import OUTPUT_DIR
    print(f"\n  Output folder: {OUTPUT_DIR}")
    print("\n  Key files:")
    print("    * weekly_bets.csv       - All predictions")
    print("    * top50_weighted.html   - Best bets ranked")
    print("    * ou_analysis.html      - Over/Under analysis")
    print("    * accumulators_*.html   - Accumulator suggestions")

    print("\n  Next steps:")
    print("    1. Review top50_weighted.html for best individual bets")
    print("    2. Check ou_analysis.html for O/U opportunities")
    print("    3. After matches: run 'python update_results.py'")
    print("=" * 70)

    # Open outputs folder
    if not args.non_interactive:
        try:
            import subprocess
            import platform

            if platform.system() == "Windows":
                subprocess.run(["explorer", str(OUTPUT_DIR)], check=False)
            elif platform.system() == "Darwin":
                subprocess.run(["open", str(OUTPUT_DIR)], check=False)
        except:
            pass

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
