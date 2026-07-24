#!/usr/bin/env python3
"""
Calibration report — measures how honest the FINAL output probabilities are.

Reads outputs/tuning_preds_cache.parquet (written by market_backtest.py's
historical backtest: per-match P_/DC_/BLEND_ predictions merged with y_*
actual outcomes) and reports, per market:

  N            matches evaluated
  Brier        multiclass Brier score (lower is better)
  ECE          expected calibration error of the picked outcome (10 bins)
  AvgConf      mean predicted probability of the picked outcome
  Accuracy     how often the picked outcome actually happened
  Overconf     AvgConf - Accuracy  (positive = overconfident)

A well-calibrated market has |Overconf| and ECE under ~3-5pp. Also writes a
per-confidence-bucket reliability table so you can see WHERE it breaks
(e.g. "at 80-90% claimed confidence we're only 61% right").

IMPORTANT: this measures the probabilities in the cache. If the cache was
built from an IN-SAMPLE backtest (models trained on the test period), the
report will look artificially perfect. Build the cache honestly first:
    TRAIN_CUTOFF_DATE=<test start> py run_weekly.py --speed full --non-interactive
    py market_backtest.py --weeks N
    py tools/calibration_report.py

Usage:
  py tools/calibration_report.py                    # auto source (BLEND > P)
  py tools/calibration_report.py --source P         # force raw ML columns
  py tools/calibration_report.py --source DC        # force Dixon-Coles columns
  py tools/calibration_report.py --min-n 100 --bins 10
"""
import sys

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from market_backtest import MARKET_CONFIGS  # market -> {actual_col, pred_cols, outcomes}

DEFAULT_CACHE = ROOT / "outputs" / "tuning_preds_cache.parquet"


def _resolve_cols(df: pd.DataFrame, pred_cols, source: str):
    """Return (cols, source_label) for the requested probability source."""
    if source in ("BLEND", "DC"):
        cols = [c.replace("P_", f"{source}_") for c in pred_cols]
        if all(c in df.columns for c in cols):
            return cols, source
        return None, None
    # auto: prefer BLEND, fall back to P
    blend = [c.replace("P_", "BLEND_") for c in pred_cols]
    if all(c in df.columns for c in blend):
        return blend, "BLEND"
    if all(c in df.columns for c in pred_cols):
        return pred_cols, "P"
    return None, None


def _market_report(df: pd.DataFrame, market: str, cfg: dict, source: str, n_bins: int):
    actual_col = cfg["actual_col"]
    outcomes = cfg["outcomes"]
    cols, src = _resolve_cols(df, cfg["pred_cols"], source)
    if cols is None or actual_col not in df.columns:
        return None, None

    sub = df[cols + [actual_col]].copy()
    sub[cols] = sub[cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(subset=cols + [actual_col])
    # Rows where every probability is 0 are untrained-market placeholders
    P = sub[cols].to_numpy(dtype=float)
    live = P.sum(axis=1) > 1e-9
    sub, P = sub[live], P[live]
    if len(sub) == 0:
        return None, None

    # Normalise rows (pairs like O/U may not sum exactly to 1 after adjustments)
    P = P / P.sum(axis=1, keepdims=True)

    actual = sub[actual_col].astype(str).to_numpy()
    onehot = np.zeros_like(P)
    for j, out in enumerate(outcomes):
        onehot[:, j] = (actual == out).astype(float)
    # Drop rows whose actual value isn't a known outcome (e.g. NaN strings)
    known = onehot.sum(axis=1) > 0
    P, onehot, actual = P[known], onehot[known], actual[known]
    n = len(P)
    if n == 0:
        return None, None

    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))

    pick_idx = P.argmax(axis=1)
    conf = P[np.arange(n), pick_idx]
    correct = onehot[np.arange(n), pick_idx]

    # Top-label ECE over equal-width confidence bins
    lo = 1.0 / len(outcomes)
    edges = np.linspace(lo, 1.0, n_bins + 1)
    ece = 0.0
    buckets = []
    for b in range(n_bins):
        left, right = edges[b], edges[b + 1]
        mask = (conf >= left) & (conf < right) if b < n_bins - 1 else (conf >= left) & (conf <= right)
        nb = int(mask.sum())
        if nb == 0:
            continue
        bucket_conf = float(conf[mask].mean())
        bucket_acc = float(correct[mask].mean())
        ece += (nb / n) * abs(bucket_conf - bucket_acc)
        buckets.append({
            "Market": market, "Source": src,
            "ConfBucket": f"{left:.2f}-{right:.2f}",
            "N": nb,
            "AvgConf": round(bucket_conf, 4),
            "Accuracy": round(bucket_acc, 4),
            "Gap": round(bucket_conf - bucket_acc, 4),
        })

    summary = {
        "Market": market,
        "Source": src,
        "N": n,
        "Brier": round(brier, 4),
        "ECE": round(ece, 4),
        "AvgConf": round(float(conf.mean()), 4),
        "Accuracy": round(float(correct.mean()), 4),
        "Overconf": round(float(conf.mean() - correct.mean()), 4),
    }
    return summary, buckets


def main():
    ap = argparse.ArgumentParser(description="Per-market calibration report")
    ap.add_argument("--cache", type=str, default=str(DEFAULT_CACHE),
                    help="Path to tuning_preds_cache.parquet")
    ap.add_argument("--source", type=str, default="auto", choices=["auto", "BLEND", "P", "DC"],
                    help="Probability columns to evaluate (auto = BLEND, else P)")
    ap.add_argument("--bins", type=int, default=10, help="Confidence bins for ECE/reliability")
    ap.add_argument("--min-n", type=int, default=50, help="Skip markets with fewer matches")
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        print(f"[ERROR] {cache} not found.")
        print("        Build it first: py market_backtest.py --weeks N")
        print("        (with models trained under TRAIN_CUTOFF_DATE for an honest report)")
        sys.exit(1)

    df = pd.read_parquet(cache)
    print(f"Loaded {len(df):,} matches from {cache}")
    if "Date" in df.columns:
        d = pd.to_datetime(df["Date"], errors="coerce")
        print(f"Period: {d.min()} .. {d.max()}")
    print()

    summaries, all_buckets = [], []
    for market, cfg in MARKET_CONFIGS.items():
        try:
            summary, buckets = _market_report(df, market, cfg, args.source, args.bins)
        except Exception as e:
            print(f"  [WARN] {market}: {e}")
            continue
        if summary is None or summary["N"] < args.min_n:
            continue
        summaries.append(summary)
        all_buckets.extend(buckets)

    if not summaries:
        print("[ERROR] No markets could be evaluated — check the cache columns.")
        sys.exit(1)

    res = pd.DataFrame(summaries).sort_values("ECE", ascending=False)
    print(f"{'Market':<22} {'Src':<6} {'N':>6}  {'Brier':>7}  {'ECE':>7}  {'AvgConf':>8}  {'Acc':>7}  {'Overconf':>9}")
    print("-" * 84)
    for _, r in res.iterrows():
        flag = "  <-- MISCALIBRATED" if abs(r["Overconf"]) > 0.05 or r["ECE"] > 0.05 else ""
        print(f"{r['Market']:<22} {r['Source']:<6} {r['N']:>6}  {r['Brier']:>7.4f}  {r['ECE']:>7.4f}  "
              f"{r['AvgConf']:>8.1%}  {r['Accuracy']:>7.1%}  {r['Overconf']:>+9.1%}{flag}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    res.to_csv(out_dir / "calibration_report.csv", index=False)
    pd.DataFrame(all_buckets).to_csv(out_dir / "calibration_buckets.csv", index=False)
    print(f"\n[OK] Wrote {out_dir / 'calibration_report.csv'}")
    print(f"[OK] Wrote {out_dir / 'calibration_buckets.csv'} (per-confidence-bucket reliability)")

    worst = res.iloc[0]
    print(f"\nWorst-calibrated market: {worst['Market']} "
          f"(ECE {worst['ECE']:.1%}, overconfidence {worst['Overconf']:+.1%})")
    print("Reminder: this is only meaningful if the cache came from an out-of-sample backtest")
    print("(models trained with TRAIN_CUTOFF_DATE before the test period).")


if __name__ == "__main__":
    main()
