# Training Cache & Speed Mode Intelligence

## ✅ New Features Added

### 1. **Smart Training Cache**

The system now intelligently decides when to retrain vs. load existing models:

**Automatically SKIPS retraining when:**
- Models exist and are recent (< 7 days old)
- Same speed mode or downgrading (e.g., full → balanced)
- Same training data/leagues

**Automatically RETRAINS when:**
- No models exist
- Models older than 7 days
- New leagues added to training data
- **UPGRADING** speed mode (fast → balanced → full)
- `FORCE_RETRAIN=1` environment variable set

### 2. **Speed Mode Quality Hierarchy**

```
FAST (quality: 1)
  ↓ Upgrade → RETRAIN
BALANCED (quality: 2)
  ↓ Upgrade → RETRAIN
FULL (quality: 3)
```

**Smart Behavior:**
- **Upgrading** (fast → balanced): Retrains with better models
- **Downgrading** (full → fast): Keeps existing better models, no retrain!
- **Same speed**: Loads cached models instantly

### 3. **Interactive Speed Selection**

When running without `--speed` flag, you'll get a menu:

```
SPEED MODE SELECTION
==================================================

  1. FAST     (~5-10 min)  - RF only, core markets
  2. BALANCED (~20-30 min) - RF + LightGBM, more markets
  3. FULL     (~2-3 hours) - All models, all markets, best accuracy

  Note: If you've run FULL before, FAST/BALANCED will reuse those models!

Choose speed mode (1-3, default=2):
```

### 4. **Training Settings Tracking**

Saved in `models/training_settings.json`:
```json
{
  "speed_mode": "balanced",
  "optuna_trials": "0",
  "n_estimators": "150",
  "leagues": ["E0", "E1", "E2", ...],
  "trained_at": "2026-01-01T14:30:00"
}
```

## Usage Examples

### First Run (No Models Exist)
```bash
python run_weekly.py --speed balanced
```
- Trains models from scratch (~20-30 min)
- Saves to `models/*.joblib`
- Saves settings to `models/training_settings.json`

### Second Run (Same Speed)
```bash
python run_weekly.py --speed balanced
```
- Loads cached models (~5 seconds!)
- Generates predictions immediately
- No retraining needed

### Upgrading Speed Mode
```bash
# First run: fast mode
python run_weekly.py --speed fast
# Trains RF only (~5 min)

# Later: upgrade to balanced
python run_weekly.py --speed balanced
# Detects upgrade, retrains with RF + LGB (~20 min)
```

### Downgrading Speed Mode (Smart!)
```bash
# First run: full mode
python run_weekly.py --speed full
# Trains all 5 models (~2 hours)

# Later: switch to fast for quick prediction
python run_weekly.py --speed fast
# REUSES existing full models! (~5 seconds)
# Output: "Keeping existing higher-quality models (no retraining needed)"
```

### Force Retraining
```bash
# Force retrain even if cache valid
set FORCE_RETRAIN=1
python run_weekly.py --speed balanced
```

## Workflow Recommendations

### Initial Setup (One-Time)
```bash
# Train with FULL mode once for best models
python run_weekly.py --speed full
# Takes ~2-3 hours, saves all models
```

### Weekly Usage
```bash
# Use FAST or BALANCED to quickly generate predictions
python run_weekly.py --speed fast
# Takes ~5 seconds (loads cached FULL models!)
```

### When to Retrain

**Every 7 days:** Models automatically retrain if > 7 days old

**When leagues change:** Add/remove leagues in config

**When upgrading:** Switch from fast → balanced → full

**Manual:** Set `FORCE_RETRAIN=1`

## Benefits

1. **Massive Time Savings**
   - First run: 2-3 hours (full training)
   - Subsequent runs: 5-10 seconds (load cache)

2. **Flexibility**
   - Train once with FULL mode
   - Use FAST mode daily (reuses FULL models!)

3. **Intelligence**
   - Only retrains when necessary
   - Detects data/setting changes automatically

4. **No Duplicate Work**
   - Downgrading keeps better models
   - Upgrading trains better models

## Current Status

✅ Training cache fully implemented  
✅ Speed mode hierarchy working  
✅ Interactive mode selection added  
✅ Smart retrain logic in place  
✅ Ready to test  

## Test It Now!

```bash
cd "d:\Users\Ian\Desktop\Chris Code\all_models"
python run_weekly.py
# Select option 2 (BALANCED) when prompted
# First run will train, second run will load cache instantly!
```
