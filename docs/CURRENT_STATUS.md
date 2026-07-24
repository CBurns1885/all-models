# Current System Status - 2026-01-02

## What Was Just Fixed

### ✅ Elo Ratings - RESTORED
**Problem**: Elo ratings existed in `historical_matches.parquet` but weren't making it into `features.parquet` or trained models.

**Root Cause**: The `_build_side_features()` function reshaped data and only kept specific columns, dropping the Elo columns during the process.

**Solution**: Modified `features.py` to:
1. Rename existing Elo columns from historical data
2. Preserve them before the reshape operation
3. Re-merge after pivot completes

**Result**:
- features.parquet now has **242 columns** (was 239)
- New Elo features: `Elo_Home`, `Elo_Away`, `Elo_Diff`
- Models will now have team strength ratings when retrained

### ✅ Backtest Confidence Threshold - UPDATED
Changed from 80% to 65% in market_backtest.py

## Current Model Architecture

### Configuration Says:
- **Speed Mode**: BALANCED
- **Models**: Single LightGBM only (`models=["lgb"]`)
- **Tuning**: Pre-tuned parameters (no Optuna)
- **CV Folds**: 3

### Actually Trained Models (from Jan 2 12:09 AM):
- **Ensemble of 7 models**: RF, ET, LR, XGBoost, CatBoost, BNN, Dixon-Coles
- **Features**: 129 features (WITHOUT Elo - trained before fix)
- **Calibration**: Using Dirichlet/Temperature scaling

**DISCREPANCY**: Models were trained with old configuration (full ensemble) before speed_config changes were applied.

## Features Now Available (Post-Fix)

### ✅ Working Features (242 total):
1. **Elo Ratings** (3): `Elo_Home`, `Elo_Away`, `Elo_Diff` ✅ JUST ADDED
2. **xG Features** (2): `Home_xG`, `Away_xG`
3. **Match Stats** (20): Shots, Possession, Corners, Cards, etc.
4. **Rolling Form** (~60): ma3, ma5 averages for goals, shots, clean sheets, BTTS, etc.
5. **Target Variables** (103): All Tier 1+2 markets

### ⚠️ Limited Features:
- **Rolling windows**: Only ma3 and ma5 (missing ma10, ma20, EWM spans)

### ❌ Missing Features:
- **Injury data**: Not available (requires premium API tier or not implemented)

## Calibration Status

### Current Approach:
Models use `DirichletCalibrator` and `TemperatureScaler` for probability calibration.

### Issue Found:
The backtest results showed **severe overconfidence**:
- 1X2: Predicting 80%+ confidence → only 37.4% accurate
- Asian Handicaps: 80%+ confidence → 18-40% accurate

This indicates the **calibration is not working properly** or the base model probabilities are too confident.

## Next Steps for You

### 1. **RETRAIN MODELS** (Recommended)
The current models:
- Don't have Elo features (trained before fix)
- Use full ensemble (7 models) instead of configured single LightGBM
- Have calibration issues

**To retrain**:
```bash
cd "d:\Users\Ian\Desktop\Chris Code\all_models"
python run_weekly.py
```

This will:
- Use new features with Elo ratings
- Train with single LightGBM (faster - 5-10x speedup)
- Use Tier 1+2 markets only (28 markets)
- Apply fresh calibration

**Expected time**: ~30-60 minutes (vs 2-3 hours for full ensemble)

### 2. **Run Backtest at 65% Confidence**
After retraining, run the backtest:
```bash
python market_backtest.py --weeks 4
```

This will:
- Use 65% minimum confidence (was 80%)
- Test on last 4 weeks of completed matches
- Show accuracy and ROI for all Tier 1+2 markets

### 3. **Evaluate Results**
Look for:
- **Improved calibration**: Predictions at 65% confidence should be accurate ~65% of time
- **Better accuracy**: Elo features should improve 1X2 and Asian Handicap predictions
- **Positive ROI markets**: Identify which markets are actually profitable

## Model vs Configuration Comparison

| Aspect | Configured (speed_config.py) | Currently Trained Models |
|--------|------------------------------|--------------------------|
| Models | Single LightGBM | 7-model ensemble |
| Features | 139 (with Elo) | 129 (without Elo) |
| Training Time | ~30 min | ~2-3 hours |
| CV Folds | 3 | 5 |
| Tuning | Pre-tuned params | Optuna tuning |
| Calibration | Yes | Yes (but overconfident) |

## Files Modified Today

1. **features.py** - Fixed Elo preservation during feature building
2. **market_backtest.py** - Changed confidence threshold to 65%
3. **features.parquet** - Rebuilt with Elo columns (242 total)

## Technical Details

### Speed Config (BALANCED mode):
```python
SpeedMode.BALANCED: SpeedConfig(
    n_folds=3,
    n_estimators=200,
    use_tuning=False,
    tuning_trials=0,
    models=["lgb"],  # SINGLE MODEL
    use_specialized=False,
    use_dc=True,
    skip_rare_markets=True,
    max_depth=12,
    early_stopping=True,
    n_jobs=-1
)
```

### Markets Being Trained:
**Tier 1 (13)**: 1X2, BTTS, OU_0.5-4.5, DC_1X/12/X2, DNB_H/A
**Tier 2 (15)**: HomeTG_0.5/1.5, AwayTG_0.5/1.5, AH_±0.5/±1.0/0.0, Win+BTTS combos, DC+O2.5 combos

Total: **28 markets** (down from 103)

## Recommendations

### Immediate (Do Now):
1. ✅ Retrain models to get Elo features + single LightGBM
2. ✅ Run backtest at 65% to see if calibration improves
3. Check which markets are actually profitable

### Short-term (Next):
1. Add missing rolling windows (ma10, ma20, EWM)
2. Investigate calibration - may need different approach
3. Consider removing Asian Handicap markets if they remain unprofitable

### Medium-term (Later):
1. Add injury features (if API allows)
2. Market-specific feature engineering
3. Implement Kelly Criterion for bet sizing instead of flat ROI
