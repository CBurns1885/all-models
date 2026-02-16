#incremental_trainer

# incremental_trainer.py
import hashlib
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from models import train_all_targets, load_trained_targets, _load_features, _feature_columns
from config import MODEL_ARTIFACTS_DIR, log_header


def _compute_feature_hash(df: pd.DataFrame) -> str:
    """Compute a hash of the feature column names used for training.

    If the feature set changes (e.g. leakage columns removed, new features
    added), the hash will differ and we force a retrain rather than silently
    using incompatible models.
    """
    num_cols, cat_cols = _feature_columns(df)
    all_cols = sorted(num_cols + cat_cols)
    col_str = "|".join(all_cols)
    return hashlib.md5(col_str.encode()).hexdigest()


def _verify_model_compatibility(models_dir: Path) -> bool:
    """Check if saved models are compatible with the current feature set.

    Returns True if compatible, False if models need retraining.
    """
    settings_file = models_dir / "training_settings.json"
    if not settings_file.exists():
        return False

    try:
        old_settings = json.loads(settings_file.read_text())
        old_hash = old_settings.get("feature_hash")
        if not old_hash:
            print("[COMPAT] No feature hash in saved settings — models may be stale")
            return False

        df = _load_features()
        current_hash = _compute_feature_hash(df)
        if current_hash != old_hash:
            print(f"[COMPAT] Feature set has changed (old={old_hash[:8]}… new={current_hash[:8]}…)")
            print("         Models were trained on different features — retraining required")
            return False

        return True
    except Exception as e:
        print(f"[COMPAT] Could not verify compatibility: {e}")
        return False


def needs_retraining(models_dir: Path = MODEL_ARTIFACTS_DIR, days_threshold: int = 7) -> bool:
    """Check if models need retraining based on new data OR changed training settings"""
    if os.environ.get("FORCE_RETRAIN") == "1":
        print("Force retraining requested")
        return True

    if not models_dir.exists():
        print("No models directory found, training from scratch")
        return True

    # CRITICAL: Check if saved models are compatible with current features
    if not _verify_model_compatibility(models_dir):
        print("[RETRAIN] Models incompatible with current feature set — retraining")
        return True

    # Check if training settings have changed
    settings_file = models_dir / "training_settings.json"

    # Get current leagues from features data
    try:
        df = _load_features()
        current_leagues = sorted(df['League'].unique().tolist()) if 'League' in df.columns else []
    except Exception:
        current_leagues = []

    current_settings = {
        "optuna_trials": os.environ.get("OPTUNA_TRIALS", "0"),
        "n_estimators": os.environ.get("N_ESTIMATORS", "300"),
        "models_only": os.environ.get("MODELS_ONLY", ""),
        "speed_mode": os.environ.get("SPEED_MODE", "balanced"),
        "leagues": current_leagues
    }

    if settings_file.exists():
        try:
            old_settings = json.loads(settings_file.read_text())

            if "leagues" not in old_settings:
                print("Old training settings format detected (no leagues field), retraining to update...")
                return True

            # Check if leagues changed
            old_leagues = set(old_settings.get("leagues", []))
            new_leagues = set(current_settings["leagues"])

            if old_leagues != new_leagues:
                new_league_only = new_leagues - old_leagues
                if new_league_only:
                    print(f"New leagues detected: {new_league_only}")
                    print("Will train only new leagues (incremental training)...")
                    return True
                else:
                    print(f"Removed leagues: {old_leagues - new_leagues}")
                    print("No new leagues, using existing models for current leagues...")
                    return False

            # Check if other settings changed
            settings_to_check = ["optuna_trials", "n_estimators", "models_only"]
            for key in settings_to_check:
                if old_settings.get(key) != current_settings.get(key):
                    print(f"Setting '{key}' changed: {old_settings.get(key)} -> {current_settings.get(key)}")
                    print("Retraining...")
                    return True

            # Special handling for speed_mode: only retrain when UPGRADING quality
            speed_quality = {"fast": 1, "balanced": 2, "full": 3}
            old_speed = old_settings.get("speed_mode", "balanced")
            new_speed = current_settings.get("speed_mode", "balanced")

            if old_speed != new_speed:
                old_quality = speed_quality.get(old_speed, 2)
                new_quality = speed_quality.get(new_speed, 2)

                if new_quality > old_quality:
                    print(f"Speed mode UPGRADED: {old_speed} → {new_speed}")
                    print("Retraining with better models...")
                    return True
                else:
                    print(f"Speed mode changed: {old_speed} → {new_speed}")
                    print("Keeping existing higher-quality models (no retraining needed)")
        except Exception:
            print("Could not read previous training settings, retraining...")
            return True
    else:
        print("No previous training settings found, retraining...")
        return True

    # Check if manifest exists
    manifest_file = models_dir / "manifest.json"
    if not manifest_file.exists():
        print("No model manifest found, retraining...")
        return True

    # Check model age
    try:
        model_age = datetime.now() - datetime.fromtimestamp(manifest_file.stat().st_mtime)
        if model_age.days > days_threshold:
            print(f"Models are {model_age.days} days old, retraining...")
            return True
    except Exception:
        print("Could not check model age, retraining...")
        return True

    # Check for new data
    try:
        df = _load_features()
        if df.empty:
            print("No features data available")
            return True

        latest_data = df['Date'].max()
        cutoff_date = latest_data - timedelta(days=days_threshold)
        new_data_count = len(df[df['Date'] > cutoff_date])

        if new_data_count > 500:
            print(f"Found {new_data_count} new matches, retraining...")
            return True
        else:
            print(f"Found {new_data_count} new matches (threshold: 500), no retraining needed")
    except Exception as e:
        print(f"Could not check for new data: {e}, retraining...")
        return True

    print(f"Models are compatible and recent, using existing models")
    return False


def smart_train_or_load():
    """Train models if needed, otherwise load existing ones"""
    if needs_retraining():
        log_header("TRAINING MODELS (clean retrain)")

        # Delete old incompatible models to avoid loading stale files
        if MODEL_ARTIFACTS_DIR.exists():
            stale_count = 0
            for old_model in MODEL_ARTIFACTS_DIR.glob("y_*.joblib"):
                old_model.unlink()
                stale_count += 1
            if stale_count:
                print(f"[CLEAN] Removed {stale_count} old model files")

        models = train_all_targets()

        # Save training settings + feature hash for future compatibility checks
        try:
            df = _load_features()
            current_leagues = sorted(df['League'].unique().tolist()) if 'League' in df.columns else []
            feature_hash = _compute_feature_hash(df)

            settings = {
                "optuna_trials": os.environ.get("OPTUNA_TRIALS", "0"),
                "n_estimators": os.environ.get("N_ESTIMATORS", "300"),
                "models_only": os.environ.get("MODELS_ONLY", ""),
                "speed_mode": os.environ.get("SPEED_MODE", "balanced"),
                "leagues": current_leagues,
                "feature_hash": feature_hash,
                "trained_at": datetime.now().isoformat()
            }
            settings_file = MODEL_ARTIFACTS_DIR / "training_settings.json"
            MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            settings_file.write_text(json.dumps(settings, indent=2))
            print(f"Saved training settings to {settings_file}")
            print(f"  Feature hash: {feature_hash[:8]}…")
            print(f"  Leagues trained: {current_leagues}")
        except Exception as e:
            print(f"Warning: Could not save training settings: {e}")

        return models
    else:
        log_header("LOADING EXISTING MODELS")
        models = load_trained_targets()
        if not models:
            log_header("NO MODELS FOUND - TRAINING FROM SCRATCH")
            return train_all_targets()
        print(f"Loaded {len(models)} existing models")
        return models