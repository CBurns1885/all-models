#!/usr/bin/env bash
set -e
cd "c:/Users/Chris/OneDrive/My Documents/NewOnedrive/OneDrive/Desktop/Chris Code/all_models"

export PYTHONIOENCODING=utf-8
export API_FOOTBALL_KEY=0f17fdba78d15a625710f7244a1cc770

echo "[CHAIN] Waiting for auto_tune.py (PID 6368) to finish..."
while py -c "import ctypes; k=ctypes.windll.kernel32; h=k.OpenProcess(1,0,6368); r=h!=0; k.CloseHandle(h); exit(0 if r else 1)" 2>/dev/null; do
  sleep 30
  echo "[CHAIN] ... auto_tune still running ($(date))"
done
echo "[CHAIN] auto_tune finished. Verifying tuning_best_params.json has final score..."
# Extra safety: ensure tuning_best_params.json has _tuning_score (written at end of auto_tune)
for i in 1 2 3 4 5; do
  py -c "import json,sys; d=json.load(open('outputs/tuning_best_params.json')); sys.exit(0 if '_tuning_score' in d else 1)" 2>/dev/null && break
  echo "[CHAIN] Waiting for _tuning_score to appear in params (attempt $i)..."
  sleep 30
done
echo "[CHAIN] Params confirmed. Starting pipeline chain."
sleep 5

echo ""
echo "=========================================="
echo "STEP 1: 52-week backtest (min conf 0.01)"
echo "=========================================="
py market_backtest.py --weeks 52 --min-confidence 0.01 2>&1 | tee outputs/backtest_52w_new.log
echo "[CHAIN] Step 1 done."

echo ""
echo "=========================================="
echo "STEP 2: Threshold analysis (by-league)"
echo "=========================================="
py threshold_analysis.py --by-league 2>&1 | tee outputs/threshold_analysis_new.log
echo "[CHAIN] Step 2 done."

echo ""
echo "=========================================="
echo "STEP 3: Per-league backtest"
echo "=========================================="
py market_backtest.py --weeks 52 --by-league 2>&1 | tee outputs/backtest_52w_league_new.log
echo "[CHAIN] Step 3 done."

echo ""
echo "=========================================="
echo "STEP 4: Full end-to-end weekly run"
echo "=========================================="
py run_weekly.py --non-interactive 2>&1 | tee outputs/weekly_run_new.log
echo "[CHAIN] Step 4 done."

echo ""
echo "[CHAIN] ALL STEPS COMPLETE."
