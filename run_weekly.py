#!/usr/bin/env python3
"""
RUN_WEEKLY.py - Complete Weekly Runner with API-Football Integration
Includes: O/U Analyzer + Accumulator Builder + Enhanced Data Pipeline

MAXIMUM ACCURACY VERSION
- API-Football for rich match statistics
- Fallback to football-data.co.uk if API fails
- Enhanced feature engineering (200+ features)
- Calibrated predictions with league-specific adjustments
"""

import sys
import os
from pathlib import Path
import datetime
import argparse

# ============================================================================
# PARSE ARGUMENTS FIRST
# ============================================================================

parser = argparse.ArgumentParser(description='Football Prediction System')
parser.add_argument('--mode', type=int, default=4, choices=[1, 2, 3, 4],
                    help='Training mode: 1=Full (50 trials), 2=Quick (25), 3=Fast (10), 4=No tuning')
parser.add_argument('--speed', type=str, default=None, choices=['fast', 'balanced', 'full'],
                    help='Speed mode: fast (~5min), balanced (~20min), full (~2hrs)')
parser.add_argument('--non-interactive', action='store_true', default=False,
                    help='Run without interactive prompts')
parser.add_argument('--use-sample-data', action='store_true',
                    help='Generate sample data if API unavailable')
args, _ = parser.parse_known_args()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Interactive speed mode selection if not specified via command line
if args.speed is None:
    if not args.non_interactive:
        print("="*60)
        print("SPEED MODE SELECTION")
        print("="*60)
        print("\n  1. FAST     (~5-10 min)  - RF only, core markets")
        print("  2. BALANCED (~20-30 min) - RF + LightGBM, more markets")
        print("  3. FULL     (~2-3 hours) - All models, all markets, best accuracy")
        print("\n  Note: If you've run FULL before, FAST/BALANCED will reuse those models!")

        while True:
            choice = input("\nChoose speed mode (1-3, default=2): ").strip() or "2"
            if choice == "1":
                args.speed = "fast"
                break
            elif choice == "2":
                args.speed = "balanced"
                break
            elif choice == "3":
                args.speed = "full"
                break
            print("Please enter 1, 2, or 3")
    else:
        args.speed = "balanced"  # Default for non-interactive

# Set speed mode FIRST (controls everything else)
os.environ["SPEED_MODE"] = args.speed

# Legacy mode settings (for backward compatibility)
trial_counts = {1: "50", 2: "25", 3: "10", 4: "0"}
os.environ["DISABLE_XGB"] = "0"  # Enable XGBoost for better accuracy
os.environ["OPTUNA_TRIALS"] = trial_counts.get(args.mode, "0")

# Adjust N_ESTIMATORS based on speed mode
estimator_counts = {"fast": "100", "balanced": "150", "full": "300"}
os.environ["N_ESTIMATORS"] = estimator_counts.get(args.speed, "150")

os.environ["USE_API_FOOTBALL"] = "1"  # Use API-Football for data
os.environ["USE_XG_FEATURES"] = "1"   # Use expected goals features
os.environ["API_FOOTBALL_KEY"] = "0f17fdba78d15a625710f7244a1cc770"

# Email configuration (optional)
os.environ["EMAIL_SMTP_SERVER"] = "smtp-mail.outlook.com"
os.environ["EMAIL_SMTP_PORT"] = "587"
os.environ["EMAIL_SENDER"] = "christopher_burns@live.co.uk"
os.environ["EMAIL_PASSWORD"] = ""
os.environ["EMAIL_RECIPIENT"] = "christopher_burns@live.co.uk"

TRAINING_START_YEAR = 2023  # More recent data = better accuracy
NON_INTERACTIVE = args.non_interactive

DEFAULT_LEAGUES = [
    # European Competitions (Champions League, Europa League, Conference League)
    "UCL", "UEL", "UECL",
    # England (leagues + cups)
    "E0", "E1", "E2", "E3", "EC", "FAC",
    # Germany (leagues + cup)
    "D1", "D2", "DFB",
    # Spain (leagues + cup)
    "SP1", "SP2", "CDR",
    # Italy (leagues + cup)
    "I1", "I2", "CIT",
    # France (leagues + cup)
    "F1", "F2", "CDF",
    # Netherlands (league + cup)
    "N1", "KNVB",
    # Belgium (league + cup)
    "B1", "BEC",
    # Portugal (league + cup)
    "P1", "TCP",
    # Scotland (leagues + cup)
    "SC0", "SC1", "SFC",
    # Turkey (league + cup)
    "T1", "TFC",
    # Additional European leagues
    "G1", "A1", "SWZ", "POL", "DEN", "NOR", "SWE", "CZE", "CRO",
]

# Import config after setting env vars
from config import OUTPUT_DIR, log_header, log_step

print("="*60)
print("FOOTBALL PREDICTION SYSTEM - WEEKLY RUN")
print("="*60)
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Output: {OUTPUT_DIR}")
print(f"Mode: {'Non-interactive' if NON_INTERACTIVE else 'Interactive'}")

# ============================================================================
# STEP 0: DOWNLOAD FIXTURES (API-Football or Fallback)
# ============================================================================

print("\n[DOWNLOAD] STEP 0: Downloading Fixtures")
print("="*40)

fixtures_file = None

# Try API-Football first
try:
    from api_client import download_upcoming_fixtures, test_api_connection, fetch_live_data_for_upcoming

    print("Testing API-Football connection...")
    if test_api_connection():
        print("\n[OK] API-Football connected!")

        # Download fixtures
        import pandas as pd
        fixtures_df = download_upcoming_fixtures(DEFAULT_LEAGUES, days_ahead=7)

        if not fixtures_df.empty:
            # Fetch live data (injuries & lineups) for upcoming fixtures
            print("\n[LIVE] Fetching injuries and lineups...")
            fixtures_df = fetch_live_data_for_upcoming(fixtures_df)

            # Save to outputs
            fixtures_file = OUTPUT_DIR / "upcoming_fixtures.csv"
            fixtures_df.to_csv(fixtures_file, index=False)
            print(f"[OK] Downloaded {len(fixtures_df)} fixtures via API (with live data)")
        else:
            print("[WARN] No fixtures from API, trying fallback...")
    else:
        print("[WARN] API connection failed, trying fallback...")
        
except ImportError:
    print("[WARN] api_client.py not found, using fallback...")
except Exception as e:
    print(f"[WARN] API error: {e}, trying fallback...")

# Fallback to football-data.co.uk
if not fixtures_file:
    try:
        from simple_fixture_downloader import download_upcoming_fixtures as download_csv
        print("\nUsing football-data.co.uk fallback...")
        fixtures_file = download_csv()

        if fixtures_file and fixtures_file.exists():
            print(f"[OK] Downloaded via fallback: {fixtures_file}")
    except Exception as e:
        print(f"[WARN] Fallback download failed: {e}")

# Manual file check
if not fixtures_file or not Path(fixtures_file).exists():
    fixtures_xlsx = Path("upcoming_fixtures.xlsx")
    fixtures_csv = Path("upcoming_fixtures.csv")
    outputs_csv = OUTPUT_DIR / "upcoming_fixtures.csv"

    if outputs_csv.exists():
        fixtures_file = outputs_csv
        print(f"[OK] Using existing: {fixtures_file}")
    elif fixtures_xlsx.exists():
        fixtures_file = fixtures_xlsx
        print(f"[OK] Using manual: {fixtures_file}")
    elif fixtures_csv.exists():
        fixtures_file = fixtures_csv
        print(f"[OK] Using manual: {fixtures_file}")
    else:
        # Try generating sample data as last resort
        print("[WARN] No fixtures file found, generating sample data...")
        try:
            from sample_data_generator import generate_upcoming_fixtures_file, initialize_database_with_sample_data
            from api_football_adapter import check_api_football_db

            # Initialize database if needed
            if not check_api_football_db():
                print("[INFO] Initializing database with sample data...")
                initialize_database_with_sample_data(
                    leagues=['E0', 'E1', 'D1', 'SP1', 'I1', 'F1', 'N1', 'T1'],
                    seasons=[2023, 2024]
                )

            # Generate upcoming fixtures
            fixtures_file = generate_upcoming_fixtures_file(
                leagues=['E0', 'E1', 'D1', 'SP1', 'I1', 'F1'],
                output_path=OUTPUT_DIR / "upcoming_fixtures.csv"
            )

            if fixtures_file and fixtures_file.exists():
                print(f"[OK] Generated sample fixtures: {fixtures_file}")
            else:
                raise RuntimeError("Failed to generate fixtures")

        except Exception as e:
            print(f"[ERROR] ERROR: No fixtures file found and sample generation failed: {e}")
            print("[INFO] Options:")
            print("   1. Check API-Football connection")
            print("   2. Check internet connection")
            print("   3. Manually download from football-data.co.uk/matches.php")
            if not NON_INTERACTIVE:
                input("Press Enter to exit...")
            sys.exit(1)

# ============================================================================
# VALIDATE FIXTURES FILE
# ============================================================================

print("\n[SEARCH] Validating fixtures file...")
try:
    import pandas as pd
    
    if str(fixtures_file).endswith('.xlsx'):
        df = pd.read_excel(fixtures_file)
    else:
        df = pd.read_csv(fixtures_file)
    
    # Fix common column issues
    if "Div" in df.columns and "League" not in df.columns:
        df = df.rename(columns={"Div": "League"})
        print("[FIX] Fixed: Renamed 'Div' -> 'League'")
        df.to_csv(fixtures_file, index=False)
    
    required = ["Date", "League", "HomeTeam", "AwayTeam"]
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        if not NON_INTERACTIVE:
            input("Press Enter to exit...")
        sys.exit(1)
    
    print(f"[OK] Validated: {len(df)} matches found")
    print(f"   Leagues: {df['League'].unique().tolist()}")

except Exception as e:
    print(f"[ERROR] Error reading fixtures: {e}")
    if not NON_INTERACTIVE:
        input("Press Enter to exit...")
    sys.exit(1)

# ============================================================================
# USER OPTIONS
# ============================================================================

print("\n" + "="*60)
print("[CONFIG] CONFIGURATION")
print("="*60)

# Use command-line argument for mode (default: no tuning)
choice = str(args.mode)
mode_trial_counts = {"1": "50", "2": "25", "3": "10", "4": "0"}
os.environ["OPTUNA_TRIALS"] = mode_trial_counts[choice]

print(f"\n[OK] Configuration set:")
print(f"   Speed mode: {args.speed}")
print(f"   Tuning: {mode_trial_counts[choice]} trials")
print(f"   Data source: API-Football (enhanced)" if os.environ.get("USE_API_FOOTBALL") == "1" else "   Data source: football-data.co.uk")
print(f"   Training period: {TRAINING_START_YEAR}-{datetime.datetime.now().year}")

# ============================================================================
# RUN PIPELINE WITH ERROR RECOVERY
# ============================================================================

TOTAL_STEPS = 12  # Updated to include market splitting step
errors = []

def run_step(step_num, step_name, func, *args, **kwargs):
    """Run a step with error recovery"""
    log_step(step_num, TOTAL_STEPS, step_name)
    
    try:
        result = func(*args, **kwargs)
        print(f"[OK] Step {step_num} complete")
        return result, None
    except Exception as e:
        error_msg = f"Step {step_num} ({step_name}): {str(e)}"
        errors.append(error_msg)
        print(f"[WARN] Step {step_num} failed: {e}")
        print("   Continuing to next step...")
        import traceback
        traceback.print_exc()
        return None, error_msg

try:
    from data_ingest import build_historical_results
    from features import build_features
    from predict import predict_week
    import datetime as dt

    # Step 1: Build historical database (uses shared API-Football DB)
    def step1():
        build_historical_results(force=False)  # Don't force rebuild unless needed

    run_step(1, "BUILD HISTORICAL DATABASE", step1)

    # Step 2: Build features
    def step2():
        build_features(force=False)  # Don't force rebuild unless needed

    run_step(2, "BUILD FEATURES", step2)

    # Step 3: Train/load models (with intelligent caching)
    def step3():
        from incremental_trainer import smart_train_or_load
        print("Checking if models need retraining...")
        print(f"  Speed mode: {args.speed}")
        print(f"  Set FORCE_RETRAIN=1 to force full retraining")
        return smart_train_or_load()

    models, err = run_step(3, "TRAIN/LOAD MODELS", step3)

    # Step 4: Generate predictions
    def step4():
        # Convert fixtures file to CSV if needed
        import pandas as pd

        if str(fixtures_file).endswith('.xlsx'):
            df = pd.read_excel(fixtures_file)
            csv_path = OUTPUT_DIR / "upcoming_fixtures.csv"
            df.to_csv(csv_path, index=False)
            predict_week(csv_path)
        else:
            predict_week(fixtures_file)

    run_step(4, "GENERATE PREDICTIONS", step4)

    # Step 5: Log predictions
    def step5():
        from accuracy_tracker import log_weekly_predictions
        week_id = dt.datetime.now().strftime('%Y-W%W')
        csv_path = OUTPUT_DIR / "weekly_bets.csv"

        if csv_path.exists():
            log_weekly_predictions(csv_path, week_id)
            print(f"[OK] Logged predictions (Week {week_id})")
        else:
            raise FileNotFoundError("weekly_bets.csv not found")

    run_step(5, "LOG PREDICTIONS", step5)

    # Step 6: Generate weighted Top 50
    def step6():
        from weighted_top50 import generate_weighted_top50
        csv_path = OUTPUT_DIR / "weekly_bets.csv"

        if csv_path.exists():
            generate_weighted_top50(csv_path)
            print("[OK] Weighted Top 50 generated")
        else:
            raise FileNotFoundError("weekly_bets.csv not found")

    run_step(6, "GENERATE WEIGHTED TOP 50", step6)

    # Step 7: O/U Analysis
    def step7():
        from ou_analyzer import analyze_ou_predictions
        csv_path = OUTPUT_DIR / "weekly_bets.csv"

        if csv_path.exists():
            df_ou = analyze_ou_predictions(min_confidence=0.90)

            if df_ou is not None and not df_ou.empty:
                print(f"[OK] O/U Analysis: {len(df_ou)} predictions")
                elite = len(df_ou[df_ou['Best_Prob'] >= 0.95])
                high = len(df_ou[df_ou['Best_Prob'] >= 0.92])
                print(f"   Elite (95%+): {elite}, High (92%+): {high}")
            else:
                print("[INFO] No high-confidence O/U predictions")
        else:
            raise FileNotFoundError("weekly_bets.csv not found")

    run_step(7, "O/U ANALYSIS", step7)

    # Step 8: Build Accumulators
    def step8():
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

                acca_count = 0
                for strategy_name, (display_name, num_legs) in strategies.items():
                    acca_html = builder.generate_report(strategy=strategy_name, num_legs=num_legs)
                    acca_path = OUTPUT_DIR / f"accumulators_{strategy_name}.html"

                    with open(acca_path, 'w', encoding='utf-8') as f:
                        f.write(acca_html)

                    acca_count += 1

                print(f"[OK] Generated {acca_count} accumulator strategies")
            else:
                raise FileNotFoundError("weekly_bets.csv not found")
        except ImportError:
            print("[WARN] Accumulator builder not available - skipping")

    run_step(8, "BUILD ACCUMULATORS", step8)

    # Step 9: Split by Market
    def step9():
        from market_splitter import split_predictions
        csv_path = OUTPUT_DIR / "weekly_bets_full.csv"

        if csv_path.exists():
            split_predictions(csv_path, OUTPUT_DIR)
            print("[OK] Market-specific files generated")
        else:
            # Try alternate file name
            csv_path = OUTPUT_DIR / "weekly_bets.csv"
            if csv_path.exists():
                split_predictions(csv_path, OUTPUT_DIR)
                print("[OK] Market-specific files generated")
            else:
                raise FileNotFoundError("weekly_bets*.csv not found")

    run_step(9, "SPLIT BY MARKET", step9)

    # Step 10: Update accuracy database
    def step10():
        try:
            from accuracy_tracker import update_accuracy_database
            update_accuracy_database()
            print("[OK] Accuracy database updated")
        except Exception as e:
            print(f"[WARN] Accuracy update skipped: {e}")

    run_step(10, "UPDATE ACCURACY DB", step10)

    # Step 11: Archive outputs
    def step11():
        import shutil
        from datetime import datetime

        date_str = datetime.now().strftime('%Y-%m-%d')
        archive_dir = Path("archives") / date_str
        archive_dir.mkdir(parents=True, exist_ok=True)

        files_to_archive = [
            "weekly_bets.csv",
            "top50_weighted.html",
            "top50_weighted.csv",
            "ou_analysis.html",
            "ou_analysis.csv",
            "accumulators_safe.html",
            "accumulators_mixed.html",
            "accumulators_aggressive.html",
        ]

        archived_count = 0
        for filename in files_to_archive:
            source = OUTPUT_DIR / filename
            if source.exists():
                name_parts = filename.rsplit('.', 1)
                if len(name_parts) == 2:
                    archived_name = f"{name_parts[0]}_{date_str}.{name_parts[1]}"
                else:
                    archived_name = f"{filename}_{date_str}"

                dest = archive_dir / archived_name
                shutil.copy2(source, dest)
                archived_count += 1

        print(f"[OK] Archived {archived_count} files to {archive_dir}")

    run_step(11, "ARCHIVE OUTPUTS", step11)

    # Step 12: Open outputs folder
    def step12():
        import subprocess
        import platform

        if platform.system() == "Windows":
            subprocess.run(["explorer", str(OUTPUT_DIR)], check=False)
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(OUTPUT_DIR)], check=False)

    run_step(12, "OPEN OUTPUTS FOLDER", step12)
    
    # ========================================================================
    # SUCCESS SUMMARY
    # ========================================================================
    
    print("\n" + "="*60)
    print("[SUCCESS] PIPELINE COMPLETE!")
    print("="*60)
    
    if errors:
        print(f"\n[WARN] {len(errors)} step(s) had errors:")
        for error in errors:
            print(f"   * {error}")
        print("\n[INFO] Check outputs folder - some files may still be generated")
    else:
        print("\n[OK] All steps completed successfully!")
    
    print("\n[DATA] Main Files:")
    print("   * weekly_bets.csv - All predictions")
    print("   * top50_weighted.html - Top picks (weighted)")
    
    print("\n[FOOTBALL] Specialized Reports:")
    print("   * ou_analysis.html - Over/Under analysis")
    print("   * accumulators_safe.html - Conservative 4-fold")
    print("   * accumulators_mixed.html - Balanced 5-fold")
    print("   * accumulators_aggressive.html - High-risk 6-fold")
    
    print("\n" + "="*60)
    print("[INFO] Next Steps:")
    print("   1. Review top50_weighted.html for best individual bets")
    print("   2. Check ou_analysis.html for O/U opportunities")
    print("   3. Review accumulator files for multi-leg options")
    print("   4. After matches: run update_results.py")
    print("="*60)

except ImportError as e:
    print(f"[ERROR] Missing required file: {e}")
    print("[FIX] Ensure all Python files are in the project folder")
    print("[PACKAGE] Run: pip install -r requirements.txt")

except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n[DONE] Pipeline finished.")
