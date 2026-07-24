# Football Prediction System

High-accuracy football match prediction system using a 5-model ensemble (LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees) combined with a Dixon-Coles Poisson model.

## Quick Start

```bash
git clone https://github.com/CBurns1885/all-models.git
cd all-models
pip install -r requirements.txt

# Create .env with your API-Football key
echo API_FOOTBALL_KEY=your_key_here > .env

# Copy football_api.db to ../data/football_api.db (not in repo — see Setup below)

# Run the weekly pipeline
py run_weekly.py --speed full --non-interactive
```

## Setup (New Machine)

1. Clone repo and `pip install -r requirements.txt`
2. Create `.env` in the repo root with `API_FOOTBALL_KEY=<key from RapidAPI>`
3. Copy `data/football_api.db` to `<parent of repo>/data/football_api.db` — this file is not in git (too large). Transfer via USB or OneDrive.
4. Populate player stats (if not already done): `py fetch/fetch_player_stats.py` — run daily until complete (~4 days)
5. Rebuild features + retrain: `py features.py` → `py models.py --speed full`

## Repository Structure

```
all_models/
├── run_weekly.py          ← Main entry point (13-step pipeline)
├── config.py              ← All paths, league codes, model settings
│
├── Core pipeline (imported by run_weekly)
│   ├── features.py        ← Feature engineering (296 features, 447 cols)
│   ├── models.py          ← 5-model ensemble training
│   ├── predict.py         ← Prediction generation + calibration
│   ├── calibration.py     ← Dirichlet + temperature scaling
│   ├── blending.py        ← DC/ML blend weight learning
│   ├── dc_predict.py      ← Dixon-Coles Poisson model
│   ├── models_dc.py       ← DC model training
│   ├── market_config.py   ← Market definitions + confidence thresholds
│   ├── market_splitter.py ← Split predictions by market
│   ├── auto_tune.py       ← Greedy + joint parameter tuning
│   ├── backtest.py        ← Walk-forward backtesting engine
│   ├── market_backtest.py ← Per-market ROI backtest
│   ├── threshold_analysis.py ← Per-league confidence threshold sweep
│   ├── picks_page.py      ← HTML picks dashboard generator
│   ├── accuracy_tracker.py / acc_builder.py
│   ├── data_ingest.py / api_client.py / api_football_adapter.py
│   ├── model_binary.py / model_multiclass.py / model_ordinal.py
│   ├── tuning.py / odds_utils.py / blending.py
│   └── weighted_top50.py / ou_analyzer.py / sample_data_generator.py
│
├── fetch/                 ← Standalone data download scripts
│   ├── fetch_season_data.py    ← Phase 1 (fixtures) + Phase 2 (match_stats)
│   ├── fetch_player_stats.py   ← Player-level stats (/fixtures/players)
│   ├── fetch_standings_all.py  ← League table standings
│   ├── fetch_historical_odds.py
│   └── download_football_data.py / ingest_local_run.py
│
├── betfair/               ← Betfair Exchange integration
│   ├── betfair_auth.py    ← Auth (reads creds from env vars)
│   ├── betfair_ltd.py     ← Bet placement
│   ├── betfair_markets.py ← Market discovery
│   └── betfair_placer.py  ← Order management
│
├── tools/                 ← One-off analysis + utility scripts
│   ├── backtest_visualizer.py
│   ├── best_bets.py
│   ├── rebuild_and_backtest.py
│   └── ...
│
├── docs/                  ← Reference documents
│
├── models/                ← Trained model artifacts (.pkl, .joblib) — gitignored
├── outputs/               ← Weekly predictions, archives — gitignored
└── data/                  ← Symlink/pointer to ../data/ — gitignored
```

## Pipeline Steps (run_weekly.py)

| Step | Name |
|------|------|
| 0 | API health check |
| 1 | Download upcoming fixtures |
| 2 | Build historical dataset |
| 3 | Build features |
| 4 | Train / load models |
| 5 | Generate predictions |
| 6 | Dixon-Coles blend |
| 7 | Calibrate |
| 8 | Market backtest (thresholds) |
| 9 | Market split |
| 10 | **Generate picks page (HTML dashboard)** |
| 11 | Update accuracy DB |
| 12 | Archive outputs |
| 13 | Open output folder |

## Data Architecture

The DB lives **outside** the repo (shared, never committed):

```
Chris Code/
├── data/
│   └── football_api.db       ← 34,725 FT fixtures, 29,471 with match_stats
└── all_models/               ← This repo
    └── outputs/YYYY-MM-DD/   ← Weekly predictions
```

Tables in `football_api.db`: `fixtures`, `match_stats`, `player_fixture_stats`, `standings`, `fixture_odds`

## Key Numbers (as of 2026-07-24)

- **296 features**, 447 columns, ~34K training rows
- **Feature hash**: `6d2f183c`
- **auto_tune best score**: 63.558
- **Markets**: 41 trained; 14+ active with per-league confidence thresholds
- **API quota**: 7,500 calls/day (API-Football Pro, key in `.env`)
