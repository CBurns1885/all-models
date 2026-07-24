# Football Prediction System — Project Reference

> Quick-reference for all key facts, files, configs, and levers. Kept concise.

---

## Data Available

| Source | Location | Shape | Date Range |
|--------|----------|-------|------------|
| `historical_matches.parquet` | `../data/processed/` | 31,172 × 52 | Apr 2023 – Feb 2026 |
| `features.parquet` | `../data/processed/` | 31,174 × 225 | Apr 2023 – Feb 2026 |
| `football_api.db` | `../data/` | 20 MB SQLite | — |

**Per year**: 2023: ~7,000 | 2024: ~11,500 | 2025: ~11,650 | 2026: ~1,000

> Config says `TRAIN_SEASONS_BACK = 8` but the DB was built from ~Apr 2023 onward.
> Effective backtest window: ~33 months across 40+ European leagues.

---

## Feature Groups (features.parquet — 225 columns)

| Group | Columns | Key Features |
|-------|---------|-------------|
| **Elo** | 6 | `Elo_Home`, `Elo_Away`, `Elo_Diff`, `Elo_Mom_*` |
| **Rolling Form** | ~60 | `Home/Away_GF_ma3/5/10/20`, `_GA_*`, `_GD_*`, `_PPG_*`, `_ewm` |
| **xG** | 2 | `Home_xG`, `Away_xG` |
| **BTTS Rate** | 8 | `Home/Away_BTTS_rate3/5/10/20` |
| **Clean Sheet Rate** | 8 | `Home/Away_CleanSheet_rate3/5/10/20` |
| **FTS Rate** | 8 | `Home/Away_FTS_rate3/5/10/20` |
| **Match Stats** | ~20 | `Shots`, `ShotsT`, `ShotsInBox`, `Possession`, `Corners`, `CardsY/R`, `PassAcc`, `BigChances` |
| **H2H** | 10 | `H2H_HomeWinRate`, `H2H_AvgGoals`, `H2H_BTTS_Rate`, `H2H_Over25_Rate`, etc. |
| **Odds** | 6 | `B365H/D/A`, `B365_Impl_H/D/A`, `B365_Overround` |
| **Contextual** | ~5 | `DayOfWeek`, `IsWeekend`, `Month`, `SeasonProgress`, `RestDiff` |
| **Targets (y_)** | 103 | All market labels — see below |

---

## Target Markets (y_ columns)

**Tier 1 — Core**
- `y_1X2` — Match result (H/D/A, 3-class)
- `y_BTTS` — Both teams to score (binary)
- `y_OU_0_5`, `y_OU_1_5`, `y_OU_2_5`, `y_OU_3_5`, `y_OU_4_5`, `y_OU_5_5` — Over/Under (binary each)
- `y_DC_1X`, `y_DC_12`, `y_DC_X2` — Double chance
- `y_DNB_H`, `y_DNB_A` — Draw no bet

**Tier 2 — Extended**
- `y_HomeTG_0_5/1_5/2_5/3_5`, `y_AwayTG_0_5/1_5/2_5/3_5`
- `y_HomeCS`, `y_AwayCS` — Clean sheet
- `y_HomeToScore`, `y_AwayToScore`
- Asian handicap: `y_AH_-2_0` through `y_AH_+2_0`, `y_AH_0_0`
- European handicap: `y_EH_m2/m1/p1/p2_H/D/A`
- Combos: `y_HomeWin_BTTS_Y`, `y_DC1X_O25`, `y_Draw_O25`, etc.
- Exact totals: `y_ExactTotal_0/1/2/3/4/5/6+`

---

## Key Files & Roles

```
run_weekly.py         Main pipeline entry point — runs full weekly update
config.py             Central config (seasons, features toggles, API key)
features.py           Feature engineering → features.parquet (225 cols)
models.py             Ensemble training (RF, ET, XGB, LGB, CatBoost + DC)
predict.py            Prediction generation + calibration + blending
models_dc.py          Dixon-Coles Poisson model
calibration.py        Dirichlet + Temperature scaling
blending.py           ML vs DC blend weight management
auto_tune.py          Optuna overnight tuning (saves best_params.json)
backtest.py           Walk-forward backtest engine
market_backtest.py    Multi-market ROI backtest
backtest_tuner.py     ← STANDALONE tweakable backtest (new — see below)
api_client.py         API-Football data fetcher
data_ingest.py        Data loading & processing
```

---

## Pipeline Flow

```
API-Football → football_api.db
                    ↓
            historical_matches.parquet
                    ↓
            features.py → features.parquet (225 cols)
                    ↓
            models.py → models/y_*.joblib (11–103 markets)
                    ↓
            predict.py → outputs/YYYY-MM-DD/*.csv + .html
                    ↓
            ou_analyzer / weighted_top50 → weekly_bets.csv, elite_picks.html
```

---

## Running the System

```bash
# Standard weekly prediction run
python run_weekly.py --speed balanced

# Full retrain with tuning (overnight)
python run_weekly.py --speed full --mode 1

# Overnight hyperparameter tuning only
python auto_tune.py

# Resume tuning from checkpoint
python auto_tune.py --resume

# Standalone backtest (no pipeline needed)
python backtest_tuner.py
# Edit CONFIG block at top to change weights, markets, models
```

**Speed Modes:**
| Mode | Models | Estimators | Approx Time |
|------|--------|-----------|-------------|
| `fast` | RF only | 100 | 5–10 min |
| `balanced` | LGB | 150 | 20–30 min |
| `full` | All 6 + tuning | 300 | 2–3 hrs |

---

## Model Ensemble

1. RandomForest
2. ExtraTreesClassifier
3. XGBoost
4. LightGBM
5. CatBoost
6. LogisticRegression
7. Dixon-Coles (Poisson statistical model)

**Blending**: ML ensemble prediction blended with Dixon-Coles Poisson at market-specific ratios.
**Calibration**: Dirichlet scaling + Temperature scaling post-training.

---

## Trained Models (last run: 2026-03-28)

11 markets saved to `models/y_*.joblib`:
`y_1X2`, `y_BTTS`, `y_OU_0_5` through `y_OU_4_5`, `y_HomeTG_0_5/1_5`, `y_AwayTG_0_5/1_5`

---

## Key Config Knobs (config.py)

```python
TRAIN_SEASONS_BACK = 8        # Max seasons back (DB limits to ~3 in practice)
FORM_WINDOWS = [3, 5, 10, 20] # Rolling average windows
EWM_SPAN = 10                 # Exponential weighted span
N_ESTIMATORS = 150            # Tree estimators
OPTUNA_TRIALS = 0             # Set >0 to enable Optuna tuning
N_FOLDS = 3                   # CV folds
USE_XG_FEATURES = True
USE_ELO = True
USE_ROLLING_FORM = True
USE_ADVANCED_STATS = True
```

---

## Auto-Tune Best Params (last run)

Saved in `outputs/tuning_best_params.json`:

```json
{
  "calibration_scalar": 0.5,
  "style_boost_magnitude": 0.1,
  "home_adv_1x2_h": 0.2, "home_adv_1x2_a": 0.2, "home_adv_1x2_d": 0.1,
  "poisson_blend_weights": {"0_5": 0.45, "1_5": 0.6, "2_5": 0.75, "3_5": 0.6, "4_5": 0.45},
  "btts_poisson_weight": 0.525,
  "time_half_life": 90,
  "ml_weight_cap": 0.7
}
```
**Last score**: 48.723 (combined 1X2 + BTTS + O/U 2.5 accuracy + Brier)

---

## Known Issues

| Issue | Status |
|-------|--------|
| Calibration overconfidence (80%+ preds → ~37% accurate) | Open |
| H2H features not yet materialized in parquet | Open — run `build_features(force=True)` |
| DC cache key collision | Fixed (MD5 hash) |
| Static league profiles | Fixed (dynamic calculation with fallback) |

---

## Improvement Backlog (ordered by expected impact)

1. Fix calibration with proper holdout cal fold (train | cal | test split)
2. Rebuild features with H2H materialized
3. Add referee features (data in DB)
4. Add injury impact scoring (data in DB)
5. Increase Optuna trials to 200+
6. SHAP feature importance to drop low-signal columns
7. League-specific models
8. Market-specific model architectures
9. Season phase features
10. Kelly Criterion bet sizing

---

## Standalone Backtest Tuner (`backtest_tuner.py`)

A single-file script to run walk-forward backtests with configurable:
- **Feature weights** per group (xG, Elo, rolling windows, H2H, etc.)
- **Feature exclusion** (set weight to 0.0)
- **Model choice** (LGB, XGB, RF, ET, Logistic, Ensemble)
- **Backtest parameters** (train window, test stride, min train size)
- **Markets** to evaluate

Edit the `CONFIG` block at the top and run `python backtest_tuner.py`.
Results saved to `backtest_results/` as CSV + summary print.

See comments in the CONFIG block for guidance on what each knob does.

---

## GitHub

https://github.com/CBurns1885/all-models
