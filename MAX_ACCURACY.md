# Maximum Accuracy Guide

Practical tuning guide for squeezing the best accuracy out of the existing pipeline. No drastic changes -- just turning the right dials.

## Ground Rules

Every change to this project must follow three principles:

1. **Maximum accuracy** -- every change must demonstrably improve prediction accuracy or maintain it. No change should be merged that makes predictions worse.
2. **Minimal difficulty / runtime** -- prefer config tweaks and weight adjustments over architectural rewrites. If a change adds significant runtime, it must justify itself with measurable accuracy gains.
3. **Backtesting is mandatory** -- no parameter change, model tweak, or calibration update goes live without a before/after backtest comparison. Run `python backtest_config.py` (section 12 below) and record the results. If accuracy doesn't improve, the change doesn't ship.

---

## 1. Run in FULL Speed Mode

The single biggest accuracy lever. `run_weekly.py` has three speed modes:

| Mode | N_ESTIMATORS | Models used | Time |
|------|-------------|-------------|------|
| fast | 100 | RF only | ~5-10 min |
| balanced | 150 | RF + LightGBM | ~20-30 min |
| **full** | **300** | **RF, ET, XGB, LGB, CAT, LR** | ~2-3 hrs |

```bash
python run_weekly.py --speed full --mode 1
```

`--mode 1` gives 50 Optuna trials for hyperparameter tuning. Combined with `--speed full` and 300 estimators, this is the maximum accuracy configuration. Once trained, faster modes reuse the saved models, so you only pay the cost once.

**Where it lives:** `config.py:237-238`, `run_weekly.py:68-74`

---

## 2. Blend Weights (ML vs Dixon-Coles)

The blend weights control how much the ML models vs the Dixon-Coles statistical model contribute to the final probability. They're stored in `models/blend_weights.json` and learned automatically by `blending.py`.

Key behaviour in `predict.py:879-964`:

- Base alpha comes from the JSON file (per-target)
- League quality boosts ML weight: elite +0.15, high +0.10, medium +0.05
- Alpha is capped at 0.85 (DC always gets at least 15%)

**What to check:**
- If `blend_weights.json` doesn't exist, blending is skipped entirely and you only get raw ML predictions
- Re-run `blending.py` after retraining to recalculate optimal weights
- For 1X2 and O/U 2.5, DC is usually strong -- if ML accuracy is poor on those, increase DC weight (lower alpha in the JSON)

---

## 3. League Profiles

`predict.py:35-136` has hardcoded league profiles with `avg_goals`, `btts_rate`, `over25_rate`, `home_adv`, `style`, etc.

These drive:
- **League calibration** (`apply_league_calibration`) -- nudges raw probabilities toward league averages
- **Poisson adjustments** (`apply_poisson_adjustment`) -- uses league avg goals as expected goals baseline
- **Blend weight boost** -- elite leagues trust ML more

**How to improve:**
- The function `calculate_league_profiles` at line 145 can compute these from your actual data. Currently the hardcoded values are used as defaults when data is missing.
- After each season, update the hardcoded profiles with fresh numbers from `calculate_league_profiles()`. The rates shift year to year (e.g. Bundesliga O/U 2.5 rate was 60% last season, could be 55% this season).
- Pay special attention to `home_adv` -- post-COVID home advantage dropped in several leagues and hasn't fully recovered.

---

## 4. Poisson Blend Weights

`apply_poisson_adjustment` (predict.py:390-437) blends ML probabilities with Poisson-derived ones. The blend weights per line:

| Line | Poisson weight | Notes |
|------|---------------|-------|
| O/U 0.5 | 0.30 | Nearly always over, less Poisson influence |
| O/U 1.5 | 0.40 | |
| O/U 2.5 | 0.50 | Equal blend -- this is the key market |
| O/U 3.5 | 0.40 | |
| O/U 4.5 | 0.30 | |
| BTTS | 0.35 | (line 436) |

**Tuning tip:** If your O/U 2.5 accuracy is poor, try increasing the Poisson weight to 0.55-0.60. Poisson is particularly strong for this line. If your ML model is well-calibrated, reduce it toward 0.40.

---

## 5. Cross-Market Constraints

`enforce_cross_market_constraints` (predict.py:323-388) enforces mathematical consistency:

- O/U probabilities must be monotonically decreasing (Over 0.5 > Over 1.5 > Over 2.5 etc.)
- BTTS Yes + O/U 0.5 Under is impossible (if both teams score, there's at least 2 goals)
- BTTS Yes implies Over 1.5 with high probability
- 1X2 probabilities sum to 1.0
- Correct Score 0-0 cannot exceed Under 0.5

These are already enabled and shouldn't need changing. If you see O/U probabilities that violate monotonicity in output, the constraint isn't being applied -- check that `enforce_cross_market_constraints` is called after blending.

---

## 6. Training Data Depth

`config.py:234`:
```python
TRAIN_SEASONS_BACK = 8  # env: FOOTY_TRAIN_SEASONS_BACK
```

More seasons = more training data = better for rare events (exact scores, BTTS patterns). But older data may not reflect current league dynamics.

**Recommendation:**
- 5-6 seasons for core markets (1X2, O/U 2.5, BTTS) -- enough data without too much drift
- 8 seasons (current default) is fine for correct score and niche markets
- Set via environment variable: `FOOTY_TRAIN_SEASONS_BACK=6`

---

## 7. Feature Engineering

`config.py:224-232`:
```python
USE_ELO = True              # Elo ratings
USE_ROLLING_FORM = True     # Rolling form windows
USE_MARKET_FEATURES = True  # Market-derived features
USE_XG_FEATURES = True      # Expected goals (API-Football)
USE_ADVANCED_STATS = True   # Advanced stats (API-Football)
USE_ODDS_COMPARISON = True  # Odds comparison features
```

All of these should be **on** for maximum accuracy. `USE_XG_FEATURES` requires the API-Football data source -- if you're using football-data.co.uk CSV fallback only, these features won't be available.

**Rolling form windows** (`config.py:242`):
```python
FORM_WINDOWS = [3, 5, 10, 20]
EWM_SPAN = 10
```

The exponentially weighted mean span of 10 matches is a good default. Shorter (5-7) reacts faster to form changes, longer (15-20) is more stable. The time-weighted feature builder in `predict.py:448-450` uses a 180-day half-life, which is reasonable for a full season.

---

## 8. Optuna Hyperparameter Tuning

`config.py:237`:
```python
OPTUNA_TRIALS = 25  # env: OPTUNA_TRIALS
```

The tuning search space is in `tuning.py:96-150`:

| Model | Parameter | Search range |
|-------|-----------|-------------|
| RF | n_estimators | 200-800 |
| RF | max_depth | 6-20 |
| XGB | n_estimators | 200-800 |
| XGB | max_depth | 3-8 |
| XGB | learning_rate | 0.01-0.2 |
| LGB | n_estimators | 200-1000 |
| LGB | learning_rate | 0.01-0.2 |

**More trials = better hyperparameters.** 50 trials (mode 1) is a good balance. 100+ gives diminishing returns.

---

## 9. Market-Specific Model Config

`market_config.py` defines per-market settings. The ones that matter most for 1X2, O/U, and BTTS:

| Market | Strategy | DC Blend | Estimators boost | Notes |
|--------|----------|----------|-----------------|-------|
| y_1X2 | FULL_ENSEMBLE | Yes | 1.0x | All 6 model types |
| y_BTTS | FULL_ENSEMBLE | Yes | 1.0x | All 6 model types |
| y_OU_2_5 | TREE_ENSEMBLE | Yes | 1.0x | 4 tree models |
| y_OU_1_5 | TREE_ENSEMBLE | Yes | 1.0x | 4 tree models |
| y_CS | POISSON_BASED | Yes | 0.5x | DC-dominant |

**Quick win:** If you're getting poor BTTS accuracy, try changing its strategy from `FULL_ENSEMBLE` to `TREE_ENSEMBLE` (drop LR which can underfit on non-linear BTTS patterns). Or increase `n_estimators_boost` to 1.2 to give it more trees.

---

## 10. Calibration Weight

`apply_league_calibration` (predict.py:180-182):
```python
confidence = abs(prob - 0.5) * 2        # 0 to 1 scale
calibration_weight = 0.3 * (1 - confidence)  # More calibration when less confident
```

This means:
- Predictions near 50% get up to 30% pull toward the league average
- Predictions near 0% or 100% get almost no calibration (already confident)

**Tuning:** The `0.3` base weight controls calibration strength. Lower it to 0.15-0.20 if your ML model is already well-calibrated (check Brier scores from backtest). Increase to 0.35-0.40 if you're seeing overconfident wrong predictions.

---

## 11. Injury Adjustment Strength

`apply_injury_adjustments` (predict.py:795-876):

- Per-key-player impact: `0.02` per injured key player (max 5% shift per team)
- Goal reduction from total injuries: `0.01` per injury, max 5%

These are conservative by design. If you're finding injury data improves accuracy, you could increase the per-player factor to `0.03`.

Requires `home_injuries` and `away_injuries` columns in fixtures -- needs API-Football data or manual entry.

---

## 12. Backtesting (MANDATORY)

**Every change must be backtested.** No exceptions. This is the only way to know whether a tweak actually improves accuracy or just looks good on paper.

### Required workflow

```
1. Run baseline backtest           -->  record Brier / accuracy / ROI
2. Make ONE change                 -->  keep it small and isolated
3. Run the same backtest again     -->  compare numbers
4. Accuracy improved?  Ship it.
   Accuracy same/worse?  Revert it.
```

### How to run

```bash
# Interactive -- choose period
python backtest_config.py

# Non-interactive examples
python backtest_config.py  # then select option 1 (single period) or 2 (compare multiple)
```

Available periods: last 3 months, 6 months, full season, 2 seasons.

### What to look at

| Metric | Good | Bad | Action |
|--------|------|-----|--------|
| Brier score | < 0.20 | > 0.30 | Recalibrate (sections 4, 10) |
| 1X2 accuracy | > 55% | < 50% | Check blend weights, league profiles |
| O/U 2.5 accuracy | > 58% | < 52% | Tune Poisson weight (section 4) |
| BTTS accuracy | > 55% | < 50% | Check calibration weight (section 10) |
| ROI | > 0% | < -5% | Reduce stake on losing markets |

### When to backtest

- After changing any value in this guide
- After retraining models
- After updating league profiles
- After a new season's data is loaded
- Before deploying weekly predictions

If you can't backtest a change, don't make the change.

---

## Quick Reference: What To Change First

In order of expected impact:

1. **Run `--speed full --mode 1`** (biggest single improvement if you haven't)
2. **Ensure blend_weights.json exists** (run blending.py after training)
3. **Update league profiles** with current-season data
4. **Check Poisson weight on O/U 2.5** (0.50 is the default, tune based on backtest)
5. **Increase OPTUNA_TRIALS to 50** if using mode 1
6. **Run backtest** to confirm each change actually helps

Everything else is fine-tuning. The architecture is sound -- it's about getting the weights and calibration right for the current season.

**Remember: backtest before, backtest after, only ship improvements.**
