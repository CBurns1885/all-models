#!/usr/bin/env python3
"""
Overnight Automated Parameter Tuning Script
============================================
Greedy sequential search across 6 parameter groups.
Checkpoint/resume, signal handling, CSV+JSON output.

Usage:
    python auto_tune.py              # full run (~43 trials)
    python auto_tune.py --resume     # resume from last checkpoint
    python auto_tune.py --dry-run    # print plan without running
"""

import argparse
import csv
import json
import signal
import sys
import time
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

LOG_CSV = OUTPUTS / "tuning_log.csv"
REPORT_JSON = OUTPUTS / "tuning_report.json"
BEST_JSON = OUTPUTS / "tuning_best_params.json"
CHECKPOINT = OUTPUTS / "tuning_checkpoint.json"

# ---------------------------------------------------------------------------
# Parameter search space (greedy sequential: one group at a time)
# ---------------------------------------------------------------------------
PARAM_GROUPS = [
    # Group 1: Calibration scalar (7 trials)
    {
        "name": "calibration",
        "params": {
            "calibration_scalar": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        },
    },
    # Group 2: Style boost magnitude (5 trials)
    {
        "name": "style_boost",
        "params": {
            "style_boost_magnitude": [0.03, 0.05, 0.08, 0.10, 0.12],
        },
    },
    # Group 3: Home advantage multipliers (8 trials)
    {
        "name": "home_advantage",
        "params": {
            "home_adv_1x2_h": [0.15, 0.2, 0.3, 0.4],
            "home_adv_1x2_a": [0.15, 0.2, 0.3, 0.4],
        },
        "constraint": lambda p: p["home_adv_1x2_h"] == p["home_adv_1x2_a"],
        "extra_static": {"home_adv_1x2_d": None},  # derived: half of H
    },
    # Group 4: Poisson blend weights (6 trials)
    {
        "name": "poisson_blend",
        "params": {
            "_poisson_scale": [0.7, 0.85, 1.0, 1.15, 1.3, 1.5],
        },
        "transform": lambda p: {
            "poisson_blend_weights": {
                "0_5": round(0.3 * p["_poisson_scale"], 3),
                "1_5": round(0.4 * p["_poisson_scale"], 3),
                "2_5": round(0.5 * p["_poisson_scale"], 3),
                "3_5": round(0.4 * p["_poisson_scale"], 3),
                "4_5": round(0.3 * p["_poisson_scale"], 3),
            },
            "btts_poisson_weight": round(0.35 * p["_poisson_scale"], 3),
        },
    },
    # Group 5: Time half-life (7 trials)
    {
        "name": "time_halflife",
        "params": {
            "time_half_life": [90, 120, 150, 180, 210, 270, 365],
        },
    },
    # Group 6: ML weight cap (5 trials)
    {
        "name": "ml_weight_cap",
        "params": {
            "ml_weight_cap": [0.70, 0.75, 0.80, 0.85, 0.90],
        },
    },
]


def _expand_combos(group: dict):
    """Yield dicts of param combos for a group, respecting constraints."""
    names = list(group["params"].keys())
    values = list(group["params"].values())

    for combo in product(*values):
        p = dict(zip(names, combo))

        # Apply constraint filter
        constraint = group.get("constraint")
        if constraint and not constraint(p):
            continue

        # Apply transform (replaces raw keys with real override keys)
        transform = group.get("transform")
        if transform:
            p = transform(p)

        # Derive extras
        extra = group.get("extra_static", {})
        for k, v in extra.items():
            if v is None and "home_adv_1x2_h" in p:
                p[k] = round(p["home_adv_1x2_h"] / 2, 3)
            elif v is not None:
                p[k] = v

        yield p


def _trial_count():
    total = 0
    for g in PARAM_GROUPS:
        total += sum(1 for _ in _expand_combos(g))
    return total


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------
def run_backtest(overrides: dict) -> dict:
    """Run a single backtest with the given TUNING_OVERRIDES.

    Returns dict with keys: accuracy, brier, roi, duration_s
    """
    # Inject overrides into predict module at runtime
    import predict
    predict.TUNING_OVERRIDES.clear()
    predict.TUNING_OVERRIDES.update(overrides)

    from backtest import BacktestEngine

    start = time.time()
    try:
        engine = BacktestEngine()
        results = engine.run()

        # Extract summary metrics
        if results is None or results.empty:
            return {"accuracy": 0.0, "brier": 1.0, "roi": -100.0, "duration_s": time.time() - start}

        accuracy = results["Accuracy_%"].mean() if "Accuracy_%" in results.columns else 0.0
        brier = results["Brier_Score"].mean() if "Brier_Score" in results.columns else 1.0
        roi = results["ROI_%"].mean() if "ROI_%" in results.columns else -100.0

        return {
            "accuracy": round(float(accuracy), 2),
            "brier": round(float(brier), 4),
            "roi": round(float(roi), 2),
            "duration_s": round(time.time() - start, 1),
        }
    except Exception as e:
        print(f"  [ERROR] Backtest failed: {e}")
        return {"accuracy": 0.0, "brier": 1.0, "roi": -100.0, "duration_s": round(time.time() - start, 1)}


def score_result(r: dict) -> float:
    """Single scalar score: higher is better.

    Weighted: 50% accuracy, 30% brier (inverted), 20% ROI.
    """
    return r["accuracy"] * 0.50 + (1 - r["brier"]) * 100 * 0.30 + max(r["roi"], -50) * 0.20


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(state: dict):
    CHECKPOINT.write_text(json.dumps(state, indent=2))


def load_checkpoint() -> dict | None:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return None


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------
def init_csv():
    if not LOG_CSV.exists():
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial", "group", "params_json",
                "accuracy", "brier", "roi", "score", "duration_s", "timestamp",
            ])


def append_csv(trial: int, group: str, params: dict, result: dict, score: float):
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            trial, group, json.dumps(params),
            result["accuracy"], result["brier"], result["roi"],
            round(score, 3), result["duration_s"],
            datetime.now().isoformat(timespec="seconds"),
        ])


# ---------------------------------------------------------------------------
# Signal handling for graceful shutdown
# ---------------------------------------------------------------------------
_shutdown = False


def _handle_signal(sig, frame):
    global _shutdown
    print(f"\n[SIGNAL] Received {signal.Signals(sig).name}. Will stop after current trial.")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Overnight automated parameter tuning")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args()

    total_trials = _trial_count()
    print(f"=== AUTO-TUNE: {total_trials} trials across {len(PARAM_GROUPS)} groups ===")

    if args.dry_run:
        for g in PARAM_GROUPS:
            combos = list(_expand_combos(g))
            print(f"  {g['name']}: {len(combos)} trials")
            for c in combos:
                print(f"    {c}")
        return

    # Load checkpoint if resuming
    best_params: dict = {}
    best_score: float = -999.0
    start_group = 0
    trial_counter = 0

    if args.resume:
        ckpt = load_checkpoint()
        if ckpt:
            best_params = ckpt.get("best_params", {})
            best_score = ckpt.get("best_score", -999.0)
            start_group = ckpt.get("next_group", 0)
            trial_counter = ckpt.get("trial_counter", 0)
            print(f"[RESUME] From group {start_group}, trial {trial_counter}, best score {best_score:.3f}")
        else:
            print("[RESUME] No checkpoint found, starting fresh.")

    init_csv()
    overall_start = time.time()

    for gi, group in enumerate(PARAM_GROUPS):
        if gi < start_group:
            continue

        group_name = group["name"]
        combos = list(_expand_combos(group))
        print(f"\n--- Group {gi+1}/{len(PARAM_GROUPS)}: {group_name} ({len(combos)} trials) ---")

        group_best_score = -999.0
        group_best_params = {}

        for ci, combo in enumerate(combos):
            if _shutdown:
                print("[SHUTDOWN] Saving checkpoint and exiting.")
                save_checkpoint({
                    "best_params": best_params,
                    "best_score": best_score,
                    "next_group": gi,
                    "trial_counter": trial_counter,
                })
                _write_outputs(best_params, best_score, overall_start)
                sys.exit(0)

            trial_counter += 1
            # Merge locked-in best params with current group's combo
            overrides = {**best_params, **combo}

            print(f"  Trial {trial_counter}/{total_trials}: {combo}")
            result = run_backtest(overrides)
            sc = score_result(result)
            print(f"    -> acc={result['accuracy']:.1f}% brier={result['brier']:.4f} roi={result['roi']:.1f}% score={sc:.3f} ({result['duration_s']:.0f}s)")

            append_csv(trial_counter, group_name, combo, result, sc)

            if sc > group_best_score:
                group_best_score = sc
                group_best_params = combo

        # Lock in this group's best
        best_params.update(group_best_params)
        if group_best_score > best_score:
            best_score = group_best_score
        print(f"  -> Best for {group_name}: {group_best_params} (score={group_best_score:.3f})")

        # Checkpoint after each group
        save_checkpoint({
            "best_params": best_params,
            "best_score": best_score,
            "next_group": gi + 1,
            "trial_counter": trial_counter,
        })

    _write_outputs(best_params, best_score, overall_start)
    # Clean up checkpoint on successful completion
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

    print(f"\n=== TUNING COMPLETE === Best score: {best_score:.3f}")
    print(f"Best params saved to: {BEST_JSON}")


def _write_outputs(best_params: dict, best_score: float, start_time: float):
    """Write final JSON outputs."""
    elapsed = time.time() - start_time

    BEST_JSON.write_text(json.dumps(best_params, indent=2))

    report = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "best_score": round(best_score, 3),
        "best_params": best_params,
        "log_file": str(LOG_CSV),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
