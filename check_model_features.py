#!/usr/bin/env python3
"""
Check which features and model types are being used
"""

import joblib
import pandas as pd
from pathlib import Path

# Check a few representative models
models_to_check = ['y_1X2', 'y_BTTS', 'y_OU_2_5', 'y_AH_-0_5']

print("="*80)
print("MODEL & FEATURE ANALYSIS")
print("="*80)

for target in models_to_check:
    model_path = Path(f'models/{target}.joblib')

    if not model_path.exists():
        print(f"\n{target}: Model not found")
        continue

    print(f"\n{target}:")
    print("-" * 80)

    model = joblib.load(model_path)
    print(f"  Model type: {type(model).__name__}")

    # Check for feature names
    if hasattr(model, 'feature_name_'):
        features = model.feature_name_
        print(f"  Total features: {len(features)}")

        # Categorize features
        xg_features = [f for f in features if 'xg' in f.lower()]
        injury_features = [f for f in features if 'inj' in f.lower() or 'injury' in f.lower()]
        elo_features = [f for f in features if 'elo' in f.lower()]
        form_features = [f for f in features if any(x in f.lower() for x in ['_ma', 'ewm', 'rolling'])]
        possession_features = [f for f in features if 'possession' in f.lower()]
        shots_features = [f for f in features if 'shot' in f.lower()]

        print(f"  xG features: {len(xg_features)} - {xg_features[:3] if xg_features else 'None'}")
        print(f"  Injury features: {len(injury_features)} - {injury_features[:3] if injury_features else 'None'}")
        print(f"  Elo features: {len(elo_features)} - {elo_features[:3] if elo_features else 'None'}")
        print(f"  Form/rolling features: {len(form_features)}")
        print(f"  Possession features: {len(possession_features)}")
        print(f"  Shots features: {len(shots_features)}")

        print(f"\n  First 15 features:")
        for i, feat in enumerate(features[:15], 1):
            print(f"    {i:2d}. {feat}")

    # Check model parameters
    if hasattr(model, 'get_params'):
        params = model.get_params()
        print(f"\n  Key parameters:")
        for key in ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves']:
            if key in params:
                print(f"    {key}: {params[key]}")

print("\n" + "="*80)
print("CHECKING FEATURE PARQUET")
print("="*80)

df = pd.read_parquet('../data/processed/features.parquet')
print(f"\nTotal columns in features.parquet: {len(df.columns)}")
print(f"Total rows (matches): {len(df)}")

# Check for specific feature types
xg_cols = [c for c in df.columns if 'xg' in c.lower()]
injury_cols = [c for c in df.columns if 'inj' in c.lower() or 'injury' in c.lower()]
elo_cols = [c for c in df.columns if 'elo' in c.lower()]

print(f"\nxG columns in parquet: {len(xg_cols)}")
if xg_cols:
    print(f"  {xg_cols}")

print(f"\nInjury columns in parquet: {len(injury_cols)}")
if injury_cols:
    print(f"  {injury_cols}")
else:
    print("  ❌ NO INJURY FEATURES FOUND")

print(f"\nElo columns in parquet: {len(elo_cols)}")
if elo_cols:
    print(f"  {elo_cols}")
else:
    print("  ❌ NO ELO FEATURES FOUND")

print("\n" + "="*80)
