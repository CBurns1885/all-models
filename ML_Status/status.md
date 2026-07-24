# Football ML Pipeline Status

**Last updated:** 2026-07-14 ~20:00

---

## Step 4: run_weekly.py — COMPLETE ✓

Exit code 0. API: 1057/7500 requests used. Speed: full (loaded existing 41 models, no retrain).

**Fixtures (2026-07-14, next 7 days): 66 total**
- UCL: 18 (CL qualifying round 1)
- UECL: 26 (Conference League qualifying)
- UEL: 6 (EL qualifying)
- NOR: 8 (Norwegian Eliteserien)
- SWE: 8 (Swedish Allsvenskan)
- All other domestic leagues: 0 (between seasons)

**Key outputs** (all in `outputs/2026-07-14/`):
- `weekly_bets.csv` — 66 match predictions
- `high_confidence_combined.csv` — 31 matches ≥90% on key markets
- `ou_analysis.html` — 150 O/U predictions, avg confidence 95.9%
- `elite_picks.html`, `top50_weighted_lite.html`

**O/U breakdown (≥90% confidence):**
- OU_0_5 Over: 62 predictions (97-99% conf) — all matches expected to score
- OU_4_5 Under: 50 predictions — low-scoring expected (early qualifier rounds)
- OU_3_5 Under: 17 predictions

**⚠️ Caveat:** Many European qualifier 1X2 predictions show `0.9047/0.0476/0.0476` exact fallback (unknown teams — no ML/DC training data). Treat as unreliable. NOR/SWE domestic picks are credible.

**7059 predictions logged** for week 2026-W28. Accuracy DB will auto-update when results come in.

---

## Pipeline Chain — Steps 1-3 Complete ✓

- [x] **auto_tune** ✓ (score 63.5575)
- [x] **Step 1: 52-week backtest** (`--min-conf 0.01`) ✓
- [x] **Step 2: Threshold analysis** (`--by-league`) ✓ — 379 per-league combos in league_thresholds.json
- [x] **Step 3: Per-league backtest** (`--weeks 52 --by-league`) ✓ — see results below
- [ ] **Step 4: run_weekly.py** ← RUNNING / STALLED

---

## Step 3: Per-League Backtest Results ✓
*(52 weeks, new league_thresholds.json, 11,444 test matches)*

Overall market performance with per-league floors applied:
| Market | Preds | Accuracy | Cnsv ROI |
|--------|-------|----------|----------|
| 1X2 | 447 | **99.3%** | -0.5% |
| HomeTG_1_5 | 341 | **99.4%** | -5.2% |
| AwayTG_0_5 | 396 | **99.2%** | -5.7% |
| OU_0_5 | 6,091 | **95.5%** | -11.2% |
| OU_4_5 | 2,453 | **93.4%** | -11.8% |
| OU_1_5 | 2,278 | 86.4% | -12.9% |
| BTTS | 2,858 | 64.5% | -25.1% |

### Best Leagues by Market (positive ROI):

**OU_2_5:** KNVB +25%, DFB +23%, EC +14%, CDF/BEC/TCP +11%, CDR +10%

**BTTS:** TCP +28%, BEC +27%, DFB +24%, CDF +19%, KNVB +18%

**OU_1_5:** I2 +12%, KNVB +10%, CDR +10%, DFB +9%, BEC +7%, TCP +6%, CDF +6%

**TotalYC_O1_5:** BEC +8%, SC1 +7%, G1 +4%, FAC +3%, SP2/SWZ/P1/T1/EC/SP1/D2 all +0-2%

**TotalYC_O2_5:** BEC +9%, SC1 +7%, FAC +4%, TCP +3%

**BookingPts_O20_5:** BEC +9%, SC1 +8%, FAC +4%, TCP +3%

**AwayTeam_Card:** BEC/SC1 100% acc (+7-8%), FAC +4%, SWZ/EC/G1/P1/SP2/SP1 +0-2%

**HomeTeam_Card:** BEC/SC1 100% acc (+6-7%), FAC +3%

**OU_0_5 (micro-edge):** 7 leagues at 100% acc (+1.7–2.4%): CIT/TCP/CDF/CDR/BEC/DFB/KNVB

---

## Step 2: Threshold Analysis Results ✓

**Global +ROI thresholds** (4 markets only — most need per-league approach):
| Market | Threshold | Preds | Accuracy | ROI |
|--------|-----------|-------|----------|-----|
| 1X2 | 97% | 22 | 100.0% | +2.5% |
| OU_0_5 | 99% | 63 | 100.0% | +0.9% |
| HomeTG_0_5 | 99% | 25 | 100.0% | +0.9% |
| HomeTG_1_5 | 99% | 36 | 100.0% | +0.8% |

379 per-league combos saved → `outputs/league_thresholds.json`

---

## auto_tune Complete ✓

**Final score: 63.5575** | Key changes for 296-feature model:
- temperature_binary: 10.0 → **3.0**
- time_half_life: 120 → **210**
- home_adv_1x2: 0.2/0.2/0.1 → **0.4/0.4/0.2**

---

## Model State

- **Models**: LGB + RF + ET + XGB + CatBoost, --speed full, 300 trees, 41 markets
- **Feature hash**: 6d2f183c (296 features, 447 cols)
- **Trained**: 2026-07-14 03:38
- **Best params**: `outputs/tuning_best_params.json` (_tuning_score: 63.5575)

---

## Data Coverage

| Source | Coverage | Notes |
|--------|----------|-------|
| match_stats | 28,440 / 34,725 (81.9%) | API ceiling |
| xG (rolling) | 100% populated | Bug fixed |
| Table position | 100% populated | Bug fixed |
| Prev-season standings | 100% populated | Bug fixed |

---

## Completed ✓

- [x] 5 pipeline bugs fixed (xG zeros, 31 missing features, Season collision, ROI margin)
- [x] Features rebuilt: 296 features / 447 cols
- [x] Full 5-model retrain (--speed full, 41 markets, 260 min)
- [x] auto_tune (63.5575)
- [x] 52-week backtest (Step 1)
- [x] Threshold analysis — new league_thresholds.json (379 combos)
- [x] Per-league backtest (Step 3)
