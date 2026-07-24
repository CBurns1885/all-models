# Feature & Model Analysis Report

## Executive Summary

**CRITICAL FINDINGS:**
1. ✅ **xG Features**: PRESENT (2 features: Home_xG, Away_xG)
2. ❌ **Injury Features**: MISSING (0 features)
3. ❌ **Elo Ratings**: MISSING (0 features)
4. ⚠️ **Rolling Form**: LIMITED (only ma3 and ma5, missing ma10 and ma20)
5. ✅ **Advanced Stats**: PRESENT (Shots, Possession, Corners, etc.)
6. 🔴 **Model Architecture**: ENSEMBLE (7 models) - NOT single LightGBM as configured!

## Models Being Used

### Current Configuration (speed_config.py)
- **Configured**: Single LightGBM model only
- **Speed Mode**: BALANCED
- **Expected**: models=["lgb"]

### Actual Models Trained (from y_1X2.joblib)
- **Base models**: 7 models running in ensemble!
  1. RandomForest (rf)
  2. ExtraTrees (et)
  3. LogisticRegression (lr)
  4. XGBoost (xgb)
  5. CatBoost (cat)
  6. BayesianNN (bnn)
  7. Dixon-Coles (dc)

**DISCREPANCY**: The system is training a FULL ENSEMBLE despite configuration saying single model.

## Features Available

### Total Features
- **Parquet file**: 239 columns (107 targets + 132 features)
- **Model uses**: 129 features (after preprocessing/filtering)

### Feature Categories

#### ✅ xG Features (2)
- Home_xG
- Away_xG

**Status**: Working correctly from API-Football database

#### ✅ Match Statistics (10 per team = 20 total)
- Shots, ShotsT (on target)
- Corners
- Yellow/Red Cards
- Possession
- ShotsInBox
- BigChances
- PassAcc (passing accuracy)

**Status**: Rich statistics from API-Football

#### ⚠️ Rolling Form Features (LIMITED)
**Available**:
- ma3 (3-match average): GF, GA, GD, PPG, Shots, ShotsT, Corners, Cards, etc.
- ma5 (5-match average): Same metrics
- Clean sheet rates, FTS rates, BTTS rates

**Missing**:
- ma10 (10-match average)
- ma20 (20-match average)
- EWM (Exponential Weighted Moving averages)

**Impact**: Model lacks medium/long-term form trends

#### ❌ Injury Features (0)
**Expected**: Player injury counts, key player absences, injury severity
**Actual**: None
**Reason**: Injury data requires premium API tier or not implemented

**Impact**: Model cannot account for team weaknesses due to injuries

#### ❌ Elo Rating Features (0)
**Expected**: Team Elo ratings (strength metric)
**Actual**: None
**Reason**: Elo calculation not implemented in feature engineering

**Impact**: Model lacks fundamental team strength indicator

## Why Performance is Poor

### 1. Model Overconfidence (Calibration Issues)
- **1X2**: 37.4% accuracy at 80%+ confidence → predicting with 80% confidence but only right 37% of time
- **Asian Handicaps**: 18-40% accuracy at 80%+ confidence → massively overconfident
- **Root cause**: Ensemble is producing overconfident probabilities that don't match reality

### 2. Missing Critical Features
**Elo Ratings**: Without Elo, model doesn't know team strength
- Can't distinguish Manchester City vs Luton Town strength properly
- Relies only on recent form (ma3, ma5) which is noisy

**Injury Impact**: Model blind to team weaknesses
- Liverpool missing Salah/Van Dijk → model treats them as full strength
- Can't adjust predictions for key absences

**Long-term Form**: Only 3-5 match windows
- Misses seasonal trends (team improving/declining over 10-20 games)
- Over-weights recent results

### 3. Wrong Model Architecture
**Configured**: Single LightGBM (fast, efficient)
**Actually Running**: 7-model ensemble (slow, complex, prone to overconfidence)

**Problem**: Ensembles can amplify overconfidence when all models are trained on same flawed features.

## Recommendations (Priority Order)

### CRITICAL (Do First)
1. **Fix Model Architecture**
   - Verify why ensemble is running despite speed_config saying single LightGBM
   - Force switch to single LightGBM as intended
   - OR: If keeping ensemble, add proper probability calibration

2. **Add Elo Ratings**
   - Implement team Elo calculation from historical results
   - Update features.py to include Home_Elo, Away_Elo, Elo_Diff
   - Rebuild features.parquet

3. **Recalibrate Probabilities**
   - Models are predicting 80%+ confidence but only achieving 37-74% accuracy
   - Apply temperature scaling or isotonic regression calibration
   - Test on validation set to ensure calibration works

### HIGH PRIORITY
4. **Add Missing Rolling Windows**
   - Implement ma10 (10-match average)
   - Implement ma20 (20-match average)
   - Implement EWM (exponential weighted) with span=10

5. **Add Injury Features** (if API tier allows)
   - Check if API-Football free tier includes injury data
   - Add injury_count, key_player_out features
   - Weight by player importance

### MEDIUM PRIORITY
6. **Feature Importance Analysis**
   - Check which features are actually useful (LightGBM has feature_importances_)
   - Remove noise features
   - Focus on high-signal features

7. **Market-Specific Feature Engineering**
   - 1X2 needs different features than Asian Handicaps
   - BTTS cares about attack/defense, not just result
   - Customize features per market type

## Next Steps

**Immediate Actions**:
1. Run feature importance check on existing models
2. Determine why ensemble is running (check models.py)
3. Add Elo ratings to feature engineering
4. Implement calibration fix
5. Re-run backtest to validate improvements

**File to Modify**:
- `features.py` - Add Elo calculation and extended rolling windows
- `models.py` - Verify model selection logic
- `calibration.py` - Improve probability calibration
- `predict.py` - Ensure calibrated probabilities are used
