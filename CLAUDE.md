# CLAUDE.md — Football Betting Model Project

Context for working on this prediction system. Read this first.

## What this project does

Trains ML models to predict football match outcomes across 70+ betting markets,
backtests them to find where genuine edge exists, and generates betting cards for
upcoming fixtures. The whole point is **finding markets where the model beats random
chance in backtesting, then only betting those** — not betting every market it can predict.

## Pipeline architecture

```
data_ingest.py  →  features.py  →  models.py  →  predict.py  →  blending.py
   (pull data)     (build 50+      (train ~70    (per-fixture    (ML + Dixon-Coles
                    y_* targets)    LightGBM       probabilities)   → BLEND_* cols)
                                    models)
                                         │
                                         ▼
                                    backtest.py  →  generate_predictions_html.py
                                  (find edge per    (HTML card of edge markets only)
                                   market)
```

## Data sources (NONE are committed — .gitignore excludes data/)

- **football-data.co.uk** (FREE) — `FOOTBALL_DATA_CSV_BASE` in config.py. Historic match
  results + closing odds for ~40 leagues/cups. This is the backtest backbone.
- **API-Football** (paid, key in config.py) — enhanced stats (shots/cards/corners/injuries)
  + live fixtures. Only needed for enhanced markets and upcoming fixtures, NOT historic backtesting.
- **Kaggle international results** (FREE) — World Cup / international, used by run_worldcup.py.

To pull free historic data: `python data_ingest.py`
Data lands in `data/processed/historical_matches.parquet` and `features.parquet` (gitignored).

## Backtest system (backtest.py) — KEY FEATURES

`evaluate_predictions()` **auto-discovers ALL evaluable markets** rather than a hardcoded few.
`_discover_markets()` scans for `y_*` actual columns and matches `BLEND_*` or `P_*` prediction
columns by prefix + outcome validation.

`generate_summary()` writes `backtest_summary.csv` with these columns (market = index):

| Column | Meaning |
|--------|---------|
| `Total_Matches`, `Correct` | sample size |
| `Accuracy_%` | raw hit rate |
| `Baseline_%` | random baseline = 100/n_outcomes (50% binary, 33% 3-way) |
| `Edge_%` | **Accuracy − Baseline — THE key metric.** Fair across binary/3-way/multiclass |
| `Brier_Score` | calibration (lower better; <0.20 good, <0.15 excellent) |
| `ROI_%`, `ROI_Type` | return; 'real' (odds-based) or 'approx' |
| `Outcomes`, `Source` | n outcomes; 'P' (ML) or 'BLEND' |

Edge tiers: Elite ≥15%, Strong 10–15%, Good 5–10%, Marginal 0–5%. Below baseline = skip.

## generate_predictions_html.py — the edge betting card

Standalone script (NOT part of the pipeline). Bridges the gap where existing HTML sheets
only show 1X2/BTTS/O-U 2.5 but the model predicts 350+ columns. It:
1. Reads `backtest_summary.csv`, filters to positive `Edge_%` markets with enough matches
2. Reads `weekly_bets.csv`, pulls BLEND_*/P_* columns for those markets
3. Builds a dark-themed mobile HTML card per fixture, ranked by edge

```bash
python generate_predictions_html.py                 # today's outputs/ folder
python generate_predictions_html.py --worldcup      # World Cup paths
python generate_predictions_html.py --min-edge 10   # only ≥10% edge markets
python generate_predictions_html.py --min-matches 30
python generate_predictions_html.py --backtest <path> --predictions <path> --output <path>
```

`_market_to_pred_columns()` maps backtest market names → prediction columns (handles
1X2, BTTS, O/U, AH, DC, DNB, clean sheets, HT/FT, combos, etc.). Extend it if you add markets.

## run_worldcup.py — separate World Cup pipeline

Isolated runner (own data_worldcup/, models_worldcup/, outputs/worldcup/). Patches config
+ module bindings so all paths redirect. Loads Kaggle international data, extracts upcoming
fixtures from blank-score / future-dated rows, trains thoroughly (50 Optuna trials, 500 est),
backtests, predicts.

```bash
python run_worldcup.py --data path/to/archive            # full overnight run
python run_worldcup.py --predict-only --claude-fixtures  # fetch knockout fixtures via Claude API
python run_worldcup.py --backtest-only
```

**Claude API fixtures** (`fetch_fixtures_via_claude`): uses web search + structured JSON to
pull upcoming knockout fixtures. Requires `ANTHROPIC_API_KEY`. `TEAM_NAME_ALIASES` +
`_normalise_team_name()` map API names → training-data names (USA→United States etc).
Tested in test_claude_fixtures.py (36 tests).

## Prediction column naming (predict.py)

- `P_*` — raw ML probabilities (`_collect_market_columns()`, ~350 columns)
- `DC_*` — Dixon-Coles probabilities
- `BLEND_*` — learned blend of ML + DC (`_apply_blend()`, only for markets in
  `pair_cols_for_target()`: 1X2, BTTS, GOAL_RANGE→GR, CS, OU_*, AH_*)
- Examples: `P_1X2_H/D/A`, `P_BTTS_Y/N`, `P_OU_2_5_O/U`, `P_AH_-0_5_H/A/P`, `P_HomeCS_Y/N`

## Config patching gotcha

Modules use `from config import X`, which binds at import time. Changing `config.X` later does
NOT propagate. run_worldcup.py solves this by patching BOTH `config` AND each module's local
binding via `setattr(mod, attr, val)`. Keep this in mind for any path-isolation work.

## Conventions

- Commit messages: clear and descriptive. Develop on feature branches.
- Data/models/CSVs are gitignored — never commit them.
- Backtest before trusting any market. Edge_% is the metric that matters, not raw accuracy.
