#!/usr/bin/env python3
"""
Overnight Automated Parameter Tuning Script
============================================
Two-phase tuning:
  Phase 1 - Greedy sequential search across 6 parameter groups (~43 trials)
  Phase 2 - Joint refinement around best values, testing interactions (~80-120 trials)
Checkpoint/resume, signal handling, CSV+JSON output.

Usage:
    python auto_tune.py              # full run (both phases)
    python auto_tune.py --resume     # resume from last checkpoint
    python auto_tune.py --dry-run    # print plan without running
    python auto_tune.py --phase1     # greedy phase only
    python auto_tune.py --phase2     # joint refinement only (uses existing best)
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
# ---------------------------------------------------------------------------
# Joint refinement: build neighbouring combos around greedy-best values
# ---------------------------------------------------------------------------
def _neighbours(value, candidates):
    """Return the value and its immediate neighbours from a sorted candidate list."""
    if value not in candidates:
        return [value]
    idx = candidates.index(value)
    lo = max(0, idx - 1)
    hi = min(len(candidates), idx + 2)
    return candidates[lo:hi]


def _build_joint_refinement(best_params: dict) -> list[tuple[str, dict]]:
    """Build joint-refinement combos: pairwise and 3-way crosses around the greedy best.

    Tests parameter interactions the greedy pass cannot detect.
    """
    combos = []

    # --- Pair 1: calibration_scalar × style_boost_magnitude ---
    cal_vals = _neighbours(
        best_params.get("calibration_scalar", 0.2),
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    )
    style_vals = _neighbours(
        best_params.get("style_boost_magnitude", 0.05),
        [0.03, 0.05, 0.08, 0.10, 0.12],
    )
    for cal, sty in product(cal_vals, style_vals):
        combo = {"calibration_scalar": cal, "style_boost_magnitude": sty}
        combos.append(("cal×style", combo))

    # --- Pair 2: home_adv × poisson_scale ---
    home_vals = _neighbours(
        best_params.get("home_adv_1x2_h", 0.2),
        [0.15, 0.2, 0.3, 0.4],
    )
    poisson_candidates = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    # Find closest poisson scale from stored blend weights
    best_btts = best_params.get("btts_poisson_weight", 0.35)
    best_scale = round(best_btts / 0.35, 2) if best_btts else 1.0
    closest_scale = min(poisson_candidates, key=lambda x: abs(x - best_scale))
    poisson_vals = _neighbours(closest_scale, poisson_candidates)
    for hv, ps in product(home_vals, poisson_vals):
        combo = {
            "home_adv_1x2_h": hv,
            "home_adv_1x2_a": hv,
            "home_adv_1x2_d": round(hv / 2, 3),
            "poisson_blend_weights": {
                "0_5": round(0.3 * ps, 3),
                "1_5": round(0.4 * ps, 3),
                "2_5": round(0.5 * ps, 3),
                "3_5": round(0.4 * ps, 3),
                "4_5": round(0.3 * ps, 3),
            },
            "btts_poisson_weight": round(0.35 * ps, 3),
        }
        combos.append(("home×poisson", combo))

    # --- Pair 3: time_half_life × ml_weight_cap ---
    time_vals = _neighbours(
        best_params.get("time_half_life", 180),
        [90, 120, 150, 180, 210, 270, 365],
    )
    ml_vals = _neighbours(
        best_params.get("ml_weight_cap", 0.80),
        [0.70, 0.75, 0.80, 0.85, 0.90],
    )
    for tv, mv in product(time_vals, ml_vals):
        combo = {"time_half_life": tv, "ml_weight_cap": mv}
        combos.append(("time×ml", combo))

    # --- 3-way cross: calibration × time_half_life × ml_weight_cap ---
    for cal, tv, mv in product(cal_vals, time_vals, ml_vals):
        combo = {"calibration_scalar": cal, "time_half_life": tv, "ml_weight_cap": mv}
        combos.append(("cal×time×ml", combo))

    # Deduplicate (same param dict can appear in multiple pairs)
    seen = set()
    unique = []
    for name, combo in combos:
        key = json.dumps(combo, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append((name, combo))

    return unique


def main():
    parser = argparse.ArgumentParser(description="Overnight automated parameter tuning")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--phase1", action="store_true", help="Run greedy phase only")
    parser.add_argument("--phase2", action="store_true", help="Run joint refinement only (uses existing best)")
    args = parser.parse_args()

    run_phase1 = not args.phase2  # run phase 1 unless --phase2 only
    run_phase2 = not args.phase1  # run phase 2 unless --phase1 only

    greedy_trials = _trial_count()

    if args.dry_run:
        print(f"=== PHASE 1: Greedy search — {greedy_trials} trials across {len(PARAM_GROUPS)} groups ===")
        for g in PARAM_GROUPS:
            combos = list(_expand_combos(g))
            print(f"  {g['name']}: {len(combos)} trials")
            for c in combos:
                print(f"    {c}")
        print(f"\n=== PHASE 2: Joint refinement — ~80-120 trials (depends on Phase 1 results) ===")
        print("  Pairs tested: calibration×style_boost, home_adv×poisson, time_halflife×ml_cap")
        print("  Plus a final 3-way cross of the top movers")
        return

    # Load checkpoint if resuming
    best_params: dict = {}
    best_score: float = -999.0
    start_group = 0
    trial_counter = 0
    phase = 1

    if args.resume:
        ckpt = load_checkpoint()
        if ckpt:
            best_params = ckpt.get("best_params", {})
            best_score = ckpt.get("best_score", -999.0)
            start_group = ckpt.get("next_group", 0)
            trial_counter = ckpt.get("trial_counter", 0)
            phase = ckpt.get("phase", 1)
            print(f"[RESUME] Phase {phase}, group {start_group}, trial {trial_counter}, best score {best_score:.3f}")
            if phase == 2:
                run_phase1 = False
        else:
            print("[RESUME] No checkpoint found, starting fresh.")

    if args.phase2 and not args.resume:
        # Load best params from previous run
        if BEST_JSON.exists():
            best_params = json.loads(BEST_JSON.read_text())
            print(f"[PHASE2] Loaded best params from {BEST_JSON}")
        else:
            print("[PHASE2] No best params found. Run phase 1 first or provide tuning_best_params.json")
            sys.exit(1)

    init_csv()
    overall_start = time.time()

    # ===================================================================
    # PHASE 1: Greedy sequential search
    # ===================================================================
    if run_phase1:
        print(f"\n{'='*60}")
        print(f"  PHASE 1: Greedy Sequential Search ({greedy_trials} trials)")
        print(f"{'='*60}")

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
                        "phase": 1,
                    })
                    _write_outputs(best_params, best_score, overall_start)
                    sys.exit(0)

                trial_counter += 1
                overrides = {**best_params, **combo}

                print(f"  Trial {trial_counter}: {combo}")
                result = run_backtest(overrides)
                sc = score_result(result)
                print(f"    -> acc={result['accuracy']:.1f}% brier={result['brier']:.4f} roi={result['roi']:.1f}% score={sc:.3f} ({result['duration_s']:.0f}s)")

                append_csv(trial_counter, group_name, combo, result, sc)

                if sc > group_best_score:
                    group_best_score = sc
                    group_best_params = combo

            best_params.update(group_best_params)
            if group_best_score > best_score:
                best_score = group_best_score
            print(f"  -> Best for {group_name}: {group_best_params} (score={group_best_score:.3f})")

            save_checkpoint({
                "best_params": best_params,
                "best_score": best_score,
                "next_group": gi + 1,
                "trial_counter": trial_counter,
                "phase": 1,
            })

        print(f"\n  Phase 1 complete. Greedy best score: {best_score:.3f}")

    # ===================================================================
    # PHASE 2: Joint refinement
    # ===================================================================
    if run_phase2:
        print(f"\n{'='*60}")
        print(f"  PHASE 2: Joint Refinement (parameter interactions)")
        print(f"{'='*60}")

        save_checkpoint({
            "best_params": best_params,
            "best_score": best_score,
            "next_group": 0,
            "trial_counter": trial_counter,
            "phase": 2,
        })

        phase2_best = deepcopy(best_params)
        phase2_score = best_score

        joint_combos = _build_joint_refinement(best_params)
        total_joint = len(joint_combos)
        print(f"  {total_joint} joint refinement trials planned\n")

        for ji, (joint_name, combo) in enumerate(joint_combos):
            if _shutdown:
                print("[SHUTDOWN] Saving checkpoint and exiting.")
                save_checkpoint({
                    "best_params": phase2_best,
                    "best_score": phase2_score,
                    "next_group": ji,
                    "trial_counter": trial_counter,
                    "phase": 2,
                })
                _write_outputs(phase2_best, phase2_score, overall_start)
                sys.exit(0)

            trial_counter += 1
            overrides = {**phase2_best, **combo}

            print(f"  Joint {ji+1}/{total_joint} [{joint_name}]: {combo}")
            result = run_backtest(overrides)
            sc = score_result(result)
            print(f"    -> acc={result['accuracy']:.1f}% brier={result['brier']:.4f} roi={result['roi']:.1f}% score={sc:.3f} ({result['duration_s']:.0f}s)")

            append_csv(trial_counter, f"joint_{joint_name}", combo, result, sc)

            if sc > phase2_score:
                phase2_score = sc
                phase2_best.update(combo)
                print(f"    *** NEW BEST: {sc:.3f} ***")

        best_params = phase2_best
        best_score = phase2_score
        print(f"\n  Phase 2 complete. Joint-refined best score: {best_score:.3f}")

    _write_outputs(best_params, best_score, overall_start)
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
