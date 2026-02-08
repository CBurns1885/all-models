#incremental_trainer

# incremental_trainer.py
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from models import train_all_targets, load_trained_targets, _load_features
from config import MODEL_ARTIFACTS_DIR, log_header

def needs_retraining(models_dir: Path = MODEL_ARTIFACTS_DIR, days_threshold: int = 7) -> bool:
    """Check if models need retraining based on new data OR changed training settings"""
    if os.environ.get("FORCE_RETRAIN") == "1":
        print("Force retraining requested")
        return True
        
    if not models_dir.exists():
        print("No models directory found, training from scratch")
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
        "speed_mode": os.environ.get("SPEED_MODE", "balanced"),  # Track speed mode
        "leagues": current_leagues  # Track which leagues models were trained on
    }
    
    if settings_file.exists():
        try:
            old_settings = json.loads(settings_file.read_text())

            # If old settings don't have leagues key, this is an old format - retrain once to save proper settings
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
                    # Only removed leagues, no new ones - no retraining needed
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
            # Quality order: fast < balanced < full
            # Downgrading (full→fast) keeps better models, upgrading (fast→full) retrains
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
                    # Don't retrain - existing models are better
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

        # Only retrain if we have A LOT of new data (e.g., 500+ matches in 7 days = major data update)
        # This prevents retraining on regular weekly updates with new fixtures
        if new_data_count > 500:  # Increased threshold from 50 to 500
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
        log_header("TRAINING MODELS")
        models = train_all_targets()
        
        # Save training settings for future comparison
        try:
            # Get current leagues from features data
            df = _load_features()
            current_leagues = sorted(df['League'].unique().tolist()) if 'League' in df.columns else []

            settings = {
                "optuna_trials": os.environ.get("OPTUNA_TRIALS", "0"),
                "n_estimators": os.environ.get("N_ESTIMATORS", "300"),
                "models_only": os.environ.get("MODELS_ONLY", ""),
                "speed_mode": os.environ.get("SPEED_MODE", "balanced"),
                "leagues": current_leagues,  # Track which leagues models were trained on
                "trained_at": datetime.now().isoformat()
            }
            settings_file = MODEL_ARTIFACTS_DIR / "training_settings.json"
            MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            settings_file.write_text(json.dumps(settings, indent=2))
            print(f"Saved training settings to {settings_file}")
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