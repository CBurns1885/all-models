# Claude Context - Football Prediction System

---

## ⚡ SESSION STATE — READ THIS FIRST (updated 2026-07-24)

### What was done this session (2026-07-24):

1. ✓ **`picks_page.py` created** — standalone HTML picks dashboard generator
   - Reads `outputs/<date>/weekly_bets_full.csv`, outputs `picks_page.html`
   - Groups best picks by market (15 markets), confidence bars, fallback detection (`!DATA` badge)
   - Fallback detection: `P_1X2_H ≈ 0.9047619` → red background + `!DATA` badge
   - Columns read: all `P_*` raw probability format (e.g. `P_1X2_H`, `P_BTTS_Y`, `P_OU_0_5_O`, etc.)

2. ✓ **`run_weekly.py` now 13 steps** (was 12) — Step 10 inserted:
   - Step 10: GENERATE PICKS PAGE (between market split and accuracy DB)
   - Old steps 10/11/12 → 11/12/13 (renumbered)
   - `picks_page.html` added to archive list (step 12)

3. ✓ **`fetch_player_stats.py` created** — populate `player_fixture_stats` from `/fixtures/players`
   - Resumable: skips fixture_ids already in table
   - Ordered `date DESC` (most recent first)
   - Quota-aware: stops when fewer than 50 requests remain
   - CLI: `--season N`, `--limit N`, `--dry-run`
   - Progress every 100 fixtures: `[N/total] pct% fetched=N players=N no_data=N`

4. ✓ **Data downloads kicked off (2026-07-24)**:
   - Phase 1 DONE: 11,327 fixtures updated across 41 leagues
   - Phase 2 DONE (eventually): 5,254 fixtures had match_stats fetched (PID 556)
   - Chain runner (PID 8604): auto-started `fetch_player_stats.py` after Phase 2 complete
   - Player stats: 29,471 total fixtures to fill → ~4 days at 7,500 API calls/day
   - Re-run `py fetch_player_stats.py` each day until `[OK] player_fixture_stats fully populated!`

5. ✓ **`.gitignore` updated**: `certs/` (Betfair SSL), `_ul*` (OneDrive), `pipeline_*.log`, `pipeline_*.err`, `chain_runner.py`
6. ✓ **`catboost_info/` untracked from git** (`git rm --cached catboost_info/ -r`)
7. ✓ **Repo pushed**: HEAD `aab99df`, working tree clean

### ⚠️ DB state after session (2026-07-24)
- `data/football_api.db`: 34,725 fixtures, latest FT 2026-06-20
- `match_stats`: ~34,725 rows (Phase 2 backfill complete)
- `player_fixture_stats`: 0 rows → being filled by `fetch_player_stats.py` over ~4 days
- `standings`: 2,200 rows

### ⚠️ Next steps after downloads complete
1. `py features.py build_features --force` — rebuild with new player-level card/fouls/rating features
2. `py models.py --speed full` — retrain all 5 models on new features
3. `py auto_tune.py` — re-tune calibration params on new model
4. `py backtest.py --weeks 52 --min-conf 0.01` — verify improvement
5. `py run_weekly.py --speed full --non-interactive` — generate fresh weekly picks

### ⚠️ Security notes (NEVER commit)
- `certs/` contains `client-2048.crt` and `client-2048.key` — Betfair SSL private key
- `.env` must contain `API_FOOTBALL_KEY=<your key>` — create on each new PC (key in RapidAPI dashboard)
- `betfair_auth.py` reads credentials from env vars only (safe to commit)

---

## Previous session (2026-07-14):

### All steps done this session (2026-07-14):
1. ✓ 5-bug pipeline audit + features rebuilt (296 features)
2. ✓ Full 5-model retrain (--speed full, 41 markets, 261 min)
3. ✓ auto_tune (score 63.5575, new params saved)
4. ✓ 52-week backtest (--min-conf 0.01) → see Step 1 results in "Current Backtest Results"
5. ✓ Threshold analysis --by-league → 379 per-league combos in league_thresholds.json
6. ✓ Per-league backtest (52 weeks) → see Step 3 results below
7. ✓ run_weekly.py --speed full --non-interactive → 66 fixtures, UCL/UEL/UECL/NOR/SWE

### ⚠️ Known issue: Unknown-team fallback in European qualifiers
Many UCL/UECL early-round teams (Atert Bissen, Tre Fiori, Vardar Skopje, etc.) show exact probability `0.9047619 / 0.047619 / 0.047619` — this is a home-heavy prior fallback when ML+DC have no training data. These predictions are unreliable for betting. NOR/SWE domestic picks are credible.

### What was done this session (2026-07-13/14) — Pipeline Audit
- **5 bugs fixed in features.py** — 31 features were silently missing + xG was all zeros. See "Bug Fixes" section.
- **Features rebuilt**: 447 cols, **296 features** (was 403/252, +44 features)
  - xG rolling: `Home_xG_ewm3`, `Away_xG_ewm3`, `Home_xG_ewm`, `Away_xG_ewm` (now working — was all zeros before)
  - Table position (20 cols): `Home_TablePos`, `Away_TablePos`, `SeasonPtsDiff`, `IsTopSix_Home/Away`, `IsBottom3_Home/Away`, etc.
  - Prev-season standings (11 cols): `PrevSeasonRankDiff`, `Home_PrevSeasonRank`, `Away_PrevSeasonPts`, etc.
- **Full 5-model retrain**: LGB + RF + ET + XGB + CatBoost, `--speed full --mode 4`, 3 CV folds, 300 trees, 41 markets, ~261 min. Feature hash: **6d2f183c** (was 04322c2f)
- **Blend weights re-learned** on last 20% (6,946 val rows from 2025-10-28+)
- **market_backtest.py ROI fix**: `fair_odds = (1/prob) * 0.9` (was `1/prob`, no bookmaker margin). Display label: "Cnsv" not "Fair".
- **Historical odds confirmed dead end**: API-Football doesn't return retroactive historical odds. 0/800 requests returned data. Only 653/34,993 fixtures have odds (fetched live pre-match). Build up organically week by week going forward.
- **auto_tune incumbent**: 63.557 (from previous run). New auto_tune running on improved 296-feature model.

### Previous session (2026-07-03)
- **API renewed** — same key `0f17fdba...`, Pro plan, 7500/day, active to 2026-08-03
- **dotenv fix** — config.py now auto-loads `.env`
- **3,534 new fixtures** fetched (Feb–Jun 2026, full 2025/26 season)
- **best params confirmed at 71.705** — restored to `outputs/tuning_best_params.json`

### What was done this session (2026-07-03)
- **API renewed** — same key `0f17fdba...`, Pro plan, 7500/day, active to 2026-08-03
- **dotenv fix** — config.py now auto-loads `.env` (key was not being read from env)
- **Phase 1 DONE** — fetched all FT fixtures for 2025/26 season across 40 leagues: **3,534 new matches added**, latest date 2026-06-20
- **Per-league threshold analysis** — `threshold_analysis.py --by-league` now generates `outputs/league_thresholds.json`
- **Per-league backtest breakdown** — `market_backtest.py --by-league` now prints per-league accuracy/ROI and saves `league_breakdown.csv`
- **LEAGUE_MARKET_CONF** — market_backtest.py auto-loads `league_thresholds.json` and uses per-league confidence floors for evaluate_market (overrides global MARKET_MIN_CONF)
- **best params confirmed at 71.705** — restored to `outputs/tuning_best_params.json`

### Core objective (2026-07-03)
- **API key**: in `.env` as `API_FOOTBALL_KEY` (pass as env var when running scripts)
- **3,534 new fixtures**: Feb 2026 – Jun 2026 (full end of 2025/26 season)
- **Stats backfill needed**: Run `py fetch_season_data.py` in user terminal (resumable, 2 days)
- **Per-league tuning**: Implemented via per-league confidence thresholds (not separate models — global model is still used)
- **Betfair**: Not yet — do after retrain + auto_tune

### Changes made for temperature_binary / scoring (2026-07-01)
- **predict.py**: Per-market DC temperature scaling — `dc_temperature_1x2` overrides global for 1X2 blending. Also `ml_weight_cap_1x2` to reduce ML contribution on 1X2 (DC is purpose-built for match results).
- **auto_tune.py**: Added Groups 8+9 to search space: `dc_temperature_1x2` [1.0-3.0] and `ml_weight_cap_1x2` [0.40-0.80]. Also added neutral overrides so existing cache remains valid.
- **Next auto_tune run** will find optimal 1X2-specific temperature + ML cap.

### What was completed this session (2026-07-01)
- **Fixed temperature_binary bug** — predict.py checked `market.startswith('TotalCorners_')` but market arrives as `P_TotalCorners_O9_5_Y` (with `P_` prefix). Fixed to `P_TotalCorners_` etc. temperature_binary had NEVER worked before this fix.
- **Fixed in-process scorer** (for testing/future use) — now computes real Brier scores (was `1-accuracy`). Added temperature_binary logistic scaling, dc_temperature_1x2 + ml_weight_cap_1x2 per-market 1X2 blend, BLEND_1X2 in MARKET_PAIRS, HomeTG/AwayTG/Corners/YC in scoring.
- **Expanded temperature_binary search space** — [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0] (corners near-random, needs heavy compression)
- **Reset score guard baseline** to 32.0 (old 49.91 was from incompatible 5-market fake-Brier scoring)
- **auto_tune #5 Phase 1 DONE** — best: temperature_binary=10.0, score=56.308 (up from 49.868)
  - All other Phase 1 params confirmed: cs=0.3, style=0.08, time=120, ml_cap=0.7, dc_temp=1.25, dc_temp_1x2=1.25, ml_cap_1x2=0.4
  - Brier 0.2543→0.1981, ROI -22.5%→-9.5% (corners/YC/HomeTG/AwayTG compressed toward 50%, fall below 70% threshold → removed from active bets)
- **auto_tune #5 COMPLETE** — final score 56.355, log: `outputs/auto_tune_2026-07-01.log`
  - Best params: temperature_binary=10.0, calibration_scalar=0.3, ml_weight_cap=0.7, dc_temp=1.25, dc_temp_1x2=1.25, ml_cap_1x2=0.4, time=120, poisson_scale=1.5
  - style_boost=0.05 (Phase 2 reverted from 0.08 found in Phase 1 greedy — joint combo with cs=0.3 preferred 0.05)
- **4-week backtest RUNNING** — log: `outputs/backtest_2026-07-01.log`

### What was completed last session (2026-06-30)
- **Fixed tuning_best_params.json** — auto_tune checkpoint bug saved wrong params. Correct best score: **54.780**
- **Feature rebuild DONE** — 403 cols (252 features + 151 targets), 31,174 rows. New: Fouls/EWM/referee/corners/cards targets, 20 league table position features
- **Model retrain DONE** — 11 markets, 35.4 min, LightGBM balanced, feature hash 04322c2f
- **Fixed is_cup/is_european suffix collision** in predict.py + cleared future_frame_cache
- **Backtest DONE** — see results table above. Major wins: 1X2 +19.4pp, BTTS +18.1pp, 4 markets now positive ROI
- **auto_tune DONE** — best score **60.844** (up from 54.780). Verified via tuning_log.csv trial 58 (joint_dc_temp×ml). Params: dc_temperature=1.1, ml_weight_cap=0.7, calibration_scalar=0.4, time_half_life=365, home_adv reduced slightly. JSON saved correctly.
- **Threshold analysis DONE** — per-market confidence floors applied to MARKET_CONFIGS. 6 markets now positive ROI: OU_0_5 (96%,+2.8%), OU_4_5 (95%,+4.6%), OU_1_5 (89%,+10.9%), OU_3_5 (86%,+14.5%), OU_2_5 (70%,+32.7%), BTTS (71%,+14.2%). 1X2/HomeTG/AwayTG have no +ROI threshold — pending FULL retrain fix.
- **FULL retrain DONE** — 12 markets trained (added y_HTFT vs 11 before). All models updated 17:54-18:20. Corners/cards skipped — insufficient stats coverage (49.4% of matches). Optuna tuning completed fast (possibly loaded cached hyperparams). Blend weights kicked off next.
- **Blend weights (2nd run) DONE**
- **Threshold analysis DONE (2nd run)** — same thresholds confirmed post-FULL retrain. 6 markets active, all +ROI.
- **auto_tune (2nd run) kicked off** (see "currently running")

---

## Project Goal
Build the most **accurate** football match prediction system possible.
- Accuracy is the primary objective over speed or simplicity
- Features should be included or dropped based purely on predictive logic/evidence
- ROI is secondary — accuracy first, betting edge follows from that

## System Overview
Football betting prediction system using ensemble ML + Dixon-Coles Poisson model.
- GitHub: https://github.com/CBurns1885/all-models
- Entry point: `run_weekly.py`
- Main pipeline: data_ingest → features → models → predict → outputs

## Key Files
| File | Purpose |
|------|---------|
| `run_weekly.py` | Main pipeline orchestrator |
| `config.py` | Central configuration |
| `features.py` | Feature engineering (~280+ columns after rebuild) |
| `models.py` | Ensemble model training (RF, ET, XGB, LGB, CatBoost, DC) |
| `predict.py` | Prediction generation + calibration |
| `backtest.py` / `backtest_engine.py` | Walk-forward backtesting |
| `auto_tune.py` | Greedy+joint predict-time parameter tuning |
| `calibration.py` | Dirichlet + Temperature scaling |
| `market_backtest.py` | Market-specific ROI backtest |
| `blending.py` | Learn DC/ML blend weights per market |
| `threshold_analysis.py` | Per-market confidence threshold sweep → find min threshold for 95% accuracy |
| `picks_page.py` | HTML picks dashboard generator — reads `weekly_bets_full.csv`, outputs `picks_page.html` grouped by market |
| `fetch_player_stats.py` | Populate `player_fixture_stats` from `/fixtures/players` — resumable, quota-aware, run daily until complete |
| `fetch_season_data.py` | Phase 1: fixture list refresh; Phase 2: match_stats backfill |

## Data Architecture
```
Chris Code/
├── data/                    ← SHARED (with dc_laptop)
│   ├── football_api.db      ← API-Football SQLite DB
│   └── processed/
│       ├── features.parquet ← rebuilt by features.py
│       └── historical_matches.parquet
├── all_models/              ← This repo
│   └── outputs/             ← Predictions, tuning results, model PKLs
└── dc_laptop/               ← Quick DC-only predictions
```

## Current Feature Set (rebuilt 2026-07-13: 447 cols total, 296 features + 151 targets)
Previous: 403/252 (June 2026). Now includes all of the below (+44 new features from bug fixes):
- Elo ratings: `Elo_Home`, `Elo_Away`, `Elo_Diff`
- Rolling form: ma3, ma5, ma10, ma20 windows for goals, shots, clean sheets, BTTS, corners, cards, fouls
- EWM rolling: span=3 (burst trend) and span=10 (stable trend) for all stats including Fouls, Corners, Cards, xG
- **xG rolling** (NOW WORKING — was all zeros before July 2026 bug fix): `Home_xG_ewm3`, `Away_xG_ewm3`, `Home_xG_ewm`, `Away_xG_ewm` (50.5% fixture coverage)
- **Live table position** (NOW INCLUDED — was silently dropped before): `Home_TablePos`, `Away_TablePos`, `SeasonPtsDiff`, `IsTopSix_Home/Away`, `IsBottom3_Home/Away`, + 14 more (100% coverage for active seasons)
- **Prev-season standings** (NOW INCLUDED — was silently dropped before): `PrevSeasonRankDiff`, `Home/Away_PrevSeasonRank/Pts/GD/RankPct`, `PrevSeasonPtsDiff/GDDiff` (70.8% coverage)
- H2H features: 10 columns
- Referee features: `Ref_AvgCards`, `Ref_AvgFouls`
- Target variables: Tier 1+2 (28 markets) + cards/corners (y_TotalYC, y_BookingPts, y_TotalCorners, y_HomeCorners, y_AwayCorners)

## Pipeline Bug Fixes (2026-07-13) — Critical Fixes That Unlocked 44 New Features

| Bug | Root Cause | Impact | Fix |
|-----|-----------|--------|-----|
| **xG all zeros** | `historical_matches.parquet` stores `home_xG` (lowercase) but `_add_team_side()` looks for `Home_xG` (uppercase H). `_ensure_cols()` fills missing with NaN → `notna().any()=False` → no rolling computed. | xG EWM features = 0 for all 34k matches | Rename at load in `build_features()`: `df.rename(columns={'home_xG': 'Home_xG', 'away_xG': 'Away_xG'})` |
| **31 features silently missing** | `_pivot_back()` only keeps `base_cols`. Table position (step 1b) + prev-season (step 1c) cols added BEFORE pivot are not in `base_cols` → dropped. | 31 features (20 table pos + 11 prev-season) never reached model | Added all 31 to `preserve_cols`; merge back after `_pivot_back()` |
| **Season_x/Season_y collision** | Both `_pivot_back` (via `base_cols`) AND `preserve_data` included `Season`. After merge: `Season_x`/`Season_y`, no plain `Season` → steps 1b/1c skip with "missing Season". | Both feature groups silently fail | Excluded `Season` from `preserve_data`; add cleanup after merge |
| **Season_x/Season_y as features** | `get_feature_columns()` excluded `Season` but not `Season_x`/`Season_y` artifacts | Raw season year used as numeric feature (data leak) | Added to exclusion set in `get_feature_columns()` |
| **ROI ~10% too optimistic** | `fair_odds = 1/prob` has no bookmaker margin | All ROI figures overstated by ~10pp | Changed to `(1/prob) * 0.9` in `market_backtest.py` |

## Autotune
`auto_tune.py` — two-phase: greedy sequential (Phase 1, ~40 trials) then joint refinement (Phase 2, ~100 trials).
- **Architecture**: Writes overrides to `tuning_best_params.json`, calls `market_backtest.py --weeks 1` per trial
- **Cache file**: `outputs/tuning_preds_cache.parquet` — DELETE before re-running after a model retrain
- **Best params**: `outputs/tuning_best_params.json`
- **Score function**: `(1 - brier)*100*0.6 + max(roi,-50)*0.25 + accuracy*0.15` — higher is better
- TUNING_OVERRIDES only affect post-training calibration and blending, NOT model weights
- **Known bug**: `tuning_report.json` `best_score` field can be wrong if run resumes from checkpoint. Always verify against `tuning_log.csv` — find best score with `df["score"].max()` grouped to June run.

## Model State (2026-07-14)
- **Models**: LGB + RF + ET + XGB + CatBoost (5-model ensemble), `--speed full`, 3 CV folds, 300 trees, 41 markets
- **Feature hash**: `6d2f183c` (296 features, 447 cols total)
- **Blend weights**: Learned on last 20%, 6,946 val rows (2025-10-28 onward)
- **auto_tune**: RUNNING (incumbent 63.557, building prediction cache as of 04:09 Jul 14)

## Best Params (incumbent 63.557, loaded by current auto_tune run)
Last updated by auto_tune run before the 5-bug fix retrain. Will be updated when new auto_tune completes.
Params in `outputs/tuning_best_params.json` — do not edit manually.
Key values from last run: temperature_binary=10.0, calibration_scalar=0.3, ml_weight_cap=0.7, dc_temp=1.25, dc_temp_1x2=1.25, ml_cap_1x2=0.4, time=120, poisson_scale=1.5, style_boost=0.05

## Known Issues / History
- **DC cache key bug** (FIXED): was `len(train_df)`, now MD5 hash of date range + leagues + count
- **Static league profiles** (FIXED): `calculate_league_profiles()` now computes style/quality/clean_sheet_rate dynamically from last 3 seasons, falls back to static for <50 matches
- **H2H features** (FIXED 2026-06-28): run_weekly.py --clean now materializes these via build_features(force=True)
- **BLEND predictions = 0 (FIXED 2026-06-28)**: Two bugs — (1) blend_weights.json missing from models/ dir (copied from outputs/), (2) pair_cols_for_target() didn't handle keys without "y_" prefix
- **Calibration overconfidence (FIXED 2026-06-28)**: models.py now uses holdout cal split — calibrator fitted on LAST FOLD OOF only (true out-of-sample), not all OOF. Fallback to full OOF if last fold <100 samples.
- **DC overconfidence (PARTIALLY FIXED 2026-06-28)**: Added `dc_temperature` TUNING_OVERRIDE (default 1.5, now tuned to 2.0) applies temperature scaling to DC probs before blending.
- **Backtest Date type**: predictions CSV loads Date as string, must cast to datetime before merge
- **API key expired (2026-06-28)**: API-Football key returns 403. DB has results only through 2026-02-04. Needs renewal via RapidAPI before fresh data can be fetched. player_fixture_stats table also empty (cards/fouls data).
- **_build_future_frame row-by-row O(N×K) (FIXED 2026-06-29)**: Replaced with vectorized merge+groupby approach (~5s vs 10 min). Future frame disk cache added.
- **DC2 no disk cache (FIXED 2026-06-29)**: dc_predict.py now saves/loads DC2 params from outputs/dc2_params_cache/. Auto_tune trial 3+ now instant.
- **EWM span-1 only (FIXED 2026-06-29)**: Added span=3 EWM alongside span=10 for burst vs stable trend. Extended to Fouls, Corners, Cards.
- **Fouls not rolling feature (FIXED 2026-06-29)**: Added Fouls to _add_team_side() and rolling stats (ma3/5/10/20, ewm3, ewm).
- **Ref_AvgFouls missing (FIXED 2026-06-29)**: Added to _add_referee_features().
- **Ref in backtest fixtures (FIXED 2026-06-29)**: market_backtest.py now passes referee column in temp fixtures.
- **DC merge positional bug (FIXED 2026-06-29)**: Changed to key-based merge on (League,Date,HomeTeam,AwayTeam) — safe regardless of row ordering.
- **Poisson xG hardcoded (FIXED 2026-06-29)**: apply_poisson_adjustment() now uses per-team Home_xG_ewm/Away_xG_ewm from the future frame when available.
- **pair_cols_for_target() missing cards/corners (FIXED 2026-06-29)**: Added handlers for TotalYC, BookingPts, HomeTeam_Card, AwayTeam_Card, TotalCorners, HomeCorners, AwayCorners.
- **auto_tune checkpoint bug (IDENTIFIED 2026-06-30)**: tuning_report.json best_score can be corrupted by resume from checkpoint. Fix: verify best params by reading tuning_log.csv and finding max score row per run date.
- **models.py UnicodeEncodeError (FIXED 2026-06-30)**: `→` arrow (U+2192) in `_importance_prune` print at line 404 crashes on Windows cp1252. Replaced with `->`. Also set `$env:PYTHONIOENCODING='utf-8'` before running py commands in PowerShell to prevent similar issues.
- **predict.py is_cup/is_european suffix collision (FIXED 2026-06-30)**: `_build_future_frame()` added `is_cup`/`is_european` BEFORE merging home_form/away_form. Since base rolling stats also carry these columns, pandas renamed them to `is_cup_x`/`is_european_x`, so the preprocessor couldn't find them. Fix: moved assignment to AFTER the merges (predict.py ~line 614). Also cleared `outputs/future_frame_cache/` to purge stale cached frames. Affected all 11 models — backtest was DC-only before fix.
- **xG all zeros in features.py (FIXED 2026-07-13)**: `historical_matches.parquet` stores `home_xG` (lowercase) but code expects `Home_xG` (uppercase). `_ensure_cols()` silently fills with NaN → no rolling computed → xG EWM = 0 for all matches. Fixed: renamed at load in `build_features()`. See "Pipeline Bug Fixes" section above.
- **31 features silently dropped by _pivot_back (FIXED 2026-07-13)**: `_pivot_back()` only keeps `base_cols`. Table position (step 1b) + prev-season (step 1c) cols are not in `base_cols` → silently dropped. Fixed: added `preserve_cols` list, merge back after pivot. See "Pipeline Bug Fixes" section.
- **Season_x/Season_y collision (FIXED 2026-07-13)**: `Season` in both `base_cols` and `preserve_data` → pandas creates `Season_x`/`Season_y` on merge → functions at 1b/1c skip with "missing Season" warning. Fixed: exclude `Season` from `preserve_data`; add cleanup after merge.
- **ROI overstated ~10pp (FIXED 2026-07-13)**: `fair_odds = 1/prob` has no bookmaker margin. Fixed: `(1/prob) * 0.9` in `market_backtest.py`. All historical backtest ROI figures before this fix were ~10pp too optimistic.
- **Historical odds dead end (2026-07-13)**: API-Football Pro plan does NOT store retroactive historical odds. Only 653/34,993 fixtures have real odds (fetched live pre-match). `fetch_historical_odds.py` returns 0 results for all historical fixtures. Real odds will build up organically as weekly predictions are run.

## Blend Weight Notes
- Blend weights stored at `models/blend_weights.json` (keys with OR without "y_" prefix both work)
- `blending.py:learn_blend_weights()` now caches DC prices per unique match (was O(N×T) calls, now O(N))
- Run after retraining: `py -c "from blending import learn_blend_weights; learn_blend_weights()"`
- Current alphas are tiny (1X2: 0.03, OU: 0.05-0.08) meaning DC dominates 95%+. Re-run auto_tune after new models are trained.
- **Must re-learn blend weights after every model retrain** — old weights will be from old OOF predictions

## Backtest Results — BASELINE (2026-06-28, pre-retrain, ML-only P_ cols)
At 70% min confidence — kept for comparison:
| Market | Preds | Accuracy | Avg Conf | ROI (fair) |
|--------|-------|----------|----------|------------|
| OU_0_5 | 873 | 94.4% | 95.9% | -1.6% |
| OU_4_5 | 864 | 85.3% | 89.8% | -4.8% |
| OU_1_5 | 675 | 76.9% | 77.5% | -0.7% |
| OU_3_5 | 699 | 70.0% | 76.8% | -8.8% |
| HomeTG_0_5 | 763 | 75.9% | 86.5% | -11.7% |
| AwayTG_0_5 | 696 | 71.4% | 82.8% | -13.1% |
| AwayTG_1_5 | 444 | 62.2% | 77.8% | -20.1% |
| BTTS | 102 | 56.9% | 72.8% | -22.3% |
| HomeTG_1_5 | 404 | 54.2% | 77.5% | -30.0% |
| 1X2 | 224 | 40.6% | 81.5% | -49.7% |
| OU_2_5 | 0 | — | — | — |

## Current Backtest Results (2026-07-14, 5-model full retrain, 296 features, per-league thresholds)

### Step 3: Per-League Backtest (52 weeks, `--by-league`, using new league_thresholds.json)
Overall market results with per-league confidence floors applied:
| Market | Preds | Accuracy | Cnsv ROI | Actual ROI |
|--------|-------|----------|----------|------------|
| 1X2 | 447 | **99.3%** | -0.5% | +30.2% |
| OU_0_5 | 6,091 | **95.5%** | -11.2% | -2.9% |
| OU_1_5 | 2,278 | 86.4% | -12.9% | +5.2% |
| OU_2_5 | 3,436 | 63.0% | -27.3% | +8.7% |
| OU_3_5 | 3,290 | 78.6% | -20.0% | +3.5% |
| OU_4_5 | 2,453 | **93.4%** | -11.8% | -2.3% |
| BTTS | 2,858 | 64.5% | -25.1% | +8.9% |
| HomeTG_1_5 | 341 | **99.4%** | -5.2% | — |
| AwayTG_0_5 | 396 | **99.2%** | -5.7% | — |

### Best Leagues per Market (per-league thresholds, 52-week backtest, positive ROI):
*(Use these for actual bets — they represent the most reliable league+market combos)*

**OU_2_5 (top profitable):**
KNVB +25.1% (42p), DFB +22.5% (24p), EC +14.3% (45p), CDF +11.2% (39p), BEC +11.0% (91p), TCP +11.0% (41p), CDR +10.0% (27p), TFC +2.8% (65p), FAC +0.2% (359p)

**BTTS (top profitable):**
TCP +28.3% (32p), BEC +26.8% (26p), DFB +23.7% (14p), CDF +18.7% (38p), KNVB +17.6% (26p), CIT +9.8% (8p), TFC +7.0% (30p), CDR +1.6% (45p)

**OU_1_5 (top profitable):**
I2 +12.1% (63p), CIT +11.0% (10p), KNVB +10.1% (48p), CDR +9.9% (32p), DFB +9.0% (25p), BEC +7.1% (108p), TCP +6.1% (50p), CDF +5.6% (64p), CRO +5.2% (31p), TFC +4.4% (76p), UECL +1.2% (81p), EC +1.1% (72p), FAC +1.0% (490p)

**TotalYC_O1_5 (top profitable):**
BEC +7.7% (182p), SC1 +6.8% (185p), G1 +3.8% (236p), FAC +3.1% (812p), SP2 +2.1% (467p), SWZ +1.9% (229p), P1 +1.8% (308p), T1 +1.1% (304p), EC +0.8% (363p), SP1 +0.7% (376p), D2 +0.5% (308p), I2 +0.0% (389p)

**TotalYC_O2_5 (top profitable):**
BEC +8.5% (182p), SC1 +7.0% (185p), FAC +3.9% (807p), TCP +2.5% (115p), TFC +0.3% (155p)

**BookingPts_O20_5 (top profitable):**
BEC +9.0% (182p), SC1 +8.3% (185p), FAC +4.1% (805p), TCP +3.3% (115p)

**AwayTeam_Card (top profitable):**
BEC +7.7% (182p, 100%), SC1 +7.1% (185p, 100%), FAC +3.5% (810p, 96%), SWZ +2.3% (229p), EC +1.7% (364p), G1 +1.3% (236p), P1 +0.5% (307p), SP2 +0.5% (467p), SP1 +0.2% (376p)

**HomeTeam_Card (top profitable):**
BEC +7.2% (182p, 100%), SC1 +6.4% (185p, 100%), FAC +2.5% (810p, 95%)

**HomeTG_0_5 (top profitable):**
CIT/KNVB/BEC/TCP/DFB/NOR/EC all 95-100% acc: CIT +6.3% (10p), KNVB +4.8% (18p), BEC +4.8% (28p), TCP +4.1% (11p), DFB +3.3% (15p), NOR +1.9% (80p), EC +0.7% (58p)

**OU_0_5 (solid at 100%, small edge):**
CIT +2.4% (20p), TCP +2.1% (71p), CDF +2.1% (94p), CDR +2.0% (57p), BEC +2.0% (127p), DFB +1.9% (33p), KNVB +1.7% (59p), EC +1.4% (183p)

**OU_3_5:**
TCP +14.9% (7p), KNVB +14.2% (5p), DFB +12.8% (6p), N1 +11.9% (26p), CDR +8.6% (19p), FAC +7.8% (17p), CDF +7.4% (12p)
*(small sample sizes — treat with caution)*

### Step 1: 52-Week Backtest Results (--min-confidence 0.01)
*(11,444 matches, 2025-06-21 to 2026-06-20, with prior league_thresholds.json applied)*
| Market | Preds | Accuracy | Cnsv ROI | Actual ROI |
|--------|-------|----------|----------|------------|
| TotalYC_O4_5 | 367 | **100.0%** | -8.9% | — |
| HomeTG_1_5 | 352 | **98.0%** | -7.1% | — |
| 1X2 | 438 | **97.5%** | -1.9% | +30.2% |
| OU_0_5 | 6,869 | **95.7%** | -11.3% | -2.1% |
| OU_4_5 | 2,344 | **93.4%** | -11.6% | -0.7% |

---

## Previous Backtest Results (2026-07-01, post auto_tune #5, temperature_binary=10.0)
At 70% min confidence. Models: LightGBM balanced, 41 markets. Key: temperature_binary=10.0 fixed, ml_weight_cap_1x2=0.4.
| Market | Preds | Accuracy | Avg Conf | ROI (fair) | Source |
|--------|-------|----------|----------|------------|--------|
| OU_2_5 | 32 | **96.9%** | 73.0% | **+32.9%** | BLEND |
| OU_0_5 | 870 | **94.4%** | 93.8% | **+0.6%** | BLEND |
| TotalYC_O6_5 | 874 | **86.8%** | 71.9% | **+21.5%** | ML |
| OU_4_5 | 805 | **87.6%** | 87.3% | **+0.2%** | BLEND |
| OU_1_5 | 614 | **82.6%** | 78.6% | **+4.8%** | BLEND |
| HomeTG_0_5 | 685 | **80.7%** | 84.8% | -5.1% | BLEND |
| 1X2 | 262 | **79.0%** | 81.1% | -2.9% | BLEND |
| BookingPts_O20_5 | 566 | **78.8%** | 70.1% | **+12.5%** | ML |
| TotalCorners_O12_5 | 874 | **78.1%** | 78.9% | -1.5% | ML |
| TotalCorners_O6_5 | 844 | **79.5%** | 72.1% | **+10.2%** | ML |
| OU_3_5 | 543 | **77.7%** | 77.9% | -0.7% | BLEND |
| BTTS | 70 | **77.1%** | 73.4% | **+5.4%** | BLEND |

**8 markets with positive ROI at 70%**: OU_2_5, TotalYC_O6_5, BookingPts_O20_5, TotalCorners_O6_5, OU_1_5, BTTS, OU_0_5, OU_4_5
**1X2: 79.0% accuracy at 81.1% conf** — much better calibrated (was 60% at 81%). -2.9% ROI (threshold may push to +ROI).
**OU_3_5: -0.7% ROI** — one threshold bump from positive.
Note: OU_2_5 only 32 predictions (small sample). TotalYC_O6_5 and TotalCorners_O6_5 are new profitable markets.

## Archived Backtest (2026-06-30, pre auto_tune #5)
At 70% min confidence: 1X2 60.0% acc (-26.8% ROI), BTTS 75.0% (+3.5%), HomeTG_0_5 78.5% (-14.1%), 4 positive-ROI markets.

## User Preferences
- **Accuracy over everything** — always optimize toward accuracy
- **Include/drop features based on predictive logic** — no attachment to existing features, add anything that could help the models
- **No hardcoded values** — derive from data wherever possible (e.g., league profiles, thresholds, weights)
- **Overnight runs are fine** — can leave laptop running
- **Inform clearly** about what needs retraining vs what is purely post-processing
- **Be direct** — short answers, no fluff
- **Keep CLAUDE.md updated constantly** — update after every step so crashes don't lose progress

## Improvement Backlog (Accuracy-Focused)
Ordered roughly by expected impact:

### High Impact — Active Work
1. ~~**Rebuild features** with H2H columns~~ **DONE 2026-06-28**
2. ~~**Add ma10 + ma20 rolling windows**~~ **DONE 2026-06-28** — FORM_WINDOWS = [3,5,10,20] in config.py
3. ~~**Add EWM rolling features**~~ **DONE 2026-06-29** — spans 3 and 10, all stats incl. Fouls/Corners/Cards
4. ~~**Fix calibration with holdout cal fold**~~ **DONE 2026-06-28** — holdout cal split in models.py
5. ~~**Run auto_tune**~~ **DONE 2026-06-29** — best score 54.780, params corrected and saved
6. ~~**Rebuild features.parquet**~~ **DONE 2026-06-30** — 403 cols (252 features + 151 targets), 31,174 rows
7. ~~**Retrain models (balanced)**~~ **DONE 2026-06-30** — 11 markets, 35.4 min, LightGBM, hash 04322c2f
8. ~~**Re-learn blend weights**~~ **DONE 2026-06-30** — 6235 val rows (last 20%, from 2025-08-23), all 41 leagues fitted
9. ~~**Run backtest post-retrain**~~ **DONE 2026-06-30** — see results table. 1X2: 40.6%→60.0%, BTTS: 56.9%→75.0%, 4 markets positive ROI
10. **Re-run auto_tune** — **RUNNING 2026-06-30** (overnight). Log: `outputs/auto_tune_2026-06-30.log`
11. **FULL mode retrain** — PENDING after auto_tune. `$env:PYTHONIOENCODING='utf-8'; py run_weekly.py --speed full --mode 4 --non-interactive`

### Medium Impact — Queued
- **Add venue/stadium features** — altitude, pitch size, artificial vs grass
- **Market-specific model per output** — separate LGB model per market rather than one shared model
- **Injury impact score** — already in DB, currently unused
- **Season phase features** — week of season, title race / relegation battle indicators

### Lower Impact — Future
- **Remove low-importance features** — run SHAP analysis, drop bottom 20% by importance
- **League-specific models** — separate model per league rather than one global model
- **Kelly Criterion bet sizing** — size bets proportional to edge rather than flat stakes

## Workflow Commands
```bash
# Weekly run
py run_weekly.py

# Rebuild features (after features.py changes)
py -c "from features import build_features; build_features(force=True)"

# Retrain models (balanced = fast, ~1-2h; full = overnight)
py run_weekly.py --speed balanced --mode 4

# Re-learn blend weights (after every retrain)
py -c "from blending import learn_blend_weights; learn_blend_weights()"

# Backtest (last 4 weeks, 70% confidence floor)
py market_backtest.py --weeks 4 --min-confidence 0.70

# Autotune (overnight — delete cache first if models were retrained)
del outputs\tuning_preds_cache.parquet
py auto_tune.py

# Verify auto_tune best params (tuning_report.json can be wrong — use this instead)
py -c "
import pandas as pd; df = pd.read_csv('outputs/tuning_log.csv', encoding='latin1')
df['ts'] = pd.to_datetime(df['timestamp'])
run = df[df['ts'] >= '2026-06-29']
best = run.loc[run['score'].idxmax()]
print(best[['trial','group','params_json','accuracy','score']])
"
```
