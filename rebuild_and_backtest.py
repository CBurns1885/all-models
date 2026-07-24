"""
Post-training pipeline: rebuild features with H2H/referee/injury, retrain,
learn blend weights, run backtest, then auto-tune overnight.

Run after run_weekly.py has completed its first pass.
"""
import os, sys
from pathlib import Path

# Ensure we're running from all_models/ directory
_SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(_SCRIPT_DIR)

# Match run_weekly.py --speed balanced --mode 4 settings
os.environ["SPEED_MODE"] = "balanced"
os.environ["N_ESTIMATORS"] = "150"
os.environ["OPTUNA_TRIALS"] = "0"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tune", action="store_true", help="Skip auto_tune overnight run")
    parser.add_argument("--skip-blend", action="store_true", help="Skip learn_blend_weights step")
    args = parser.parse_args()

    print("=" * 60)
    print("POST-TRAINING REBUILD PIPELINE")
    print("H2H + referee + injury -> retrain -> blend -> backtest -> tune")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1: Rebuild features with new feature set
    # ------------------------------------------------------------------ #
    print("\n[1/5] Rebuilding features.parquet with H2H / referee / injury...")
    from features import build_features
    feat_path = build_features(force=True)
    import pandas as pd
    df_feat = pd.read_parquet(feat_path)
    print(f"[OK] Features: {len(df_feat):,} rows × {len(df_feat.columns)} cols")
    for prefix in ('H2H_', 'Ref_', 'Home_InjuryCount', 'Away_InjuryCount'):
        cols = [c for c in df_feat.columns if c.startswith(prefix) or c == prefix.rstrip('_')]
        if cols:
            print(f"     {prefix}: {cols}")
    del df_feat

    # ------------------------------------------------------------------ #
    # Step 2: Retrain models on new features
    # ------------------------------------------------------------------ #
    print("\n[2/5] Retraining all models on new features...")
    from config import MODEL_ARTIFACTS_DIR
    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # Delete stale models so train_all_targets always retrains
    deleted = 0
    for f in MODEL_ARTIFACTS_DIR.glob("y_*.joblib"):
        f.unlink()
        deleted += 1
    if deleted:
        print(f"   Cleared {deleted} stale model files")
    from models import train_all_targets
    models = train_all_targets()
    print(f"[OK] Trained {len(models)} models")

    # ------------------------------------------------------------------ #
    # Step 3: Learn blend weights
    # ------------------------------------------------------------------ #
    if not args.skip_blend:
        print("\n[3/5] Learning blend weights (DC price caching active)...")
        from blending import learn_blend_weights
        weights = learn_blend_weights()
        print(f"[OK] Weights learned for {len(weights)} targets:")
        for k, v in sorted(weights.items()):
            print(f"      {k}: alpha={v:.4f}  ({v*100:.1f}% ML / {(1-v)*100:.1f}% DC)")
    else:
        print("\n[3/5] SKIPPED blend weights (--skip-blend)")

    # ------------------------------------------------------------------ #
    # Step 4: Run backtest (BLEND_ columns preferred when available)
    # ------------------------------------------------------------------ #
    print("\n[4/5] Running 4-week market backtest at 70% min confidence...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "market_backtest.py", "--weeks", "4", "--min-confidence", "0.70"],
        capture_output=False,
    )
    if result.returncode == 0:
        print("[OK] Backtest complete — see outputs/market_backtest_analysis.csv")
    else:
        print(f"[WARN] Backtest exited with code {result.returncode}")

    # ------------------------------------------------------------------ #
    # Step 5: Auto-tune overnight (optimise dc_temperature + blends)
    # ------------------------------------------------------------------ #
    if not args.skip_tune:
        print("\n[5/5] Launching auto_tune.py overnight (both phases)...")
        result2 = subprocess.run(
            [sys.executable, "auto_tune.py"],
            capture_output=False,
        )
        if result2.returncode == 0:
            print("[OK] Auto-tune complete — see outputs/tuning_best_params.json")
        else:
            print(f"[WARN] Auto-tune exited with code {result2.returncode}")
    else:
        print("\n[5/5] SKIPPED auto_tune (--skip-tune)")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
