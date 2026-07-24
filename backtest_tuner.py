"""
backtest_tuner.py — Standalone Walk-Forward Backtest with Tweakable Feature Weights
======================================================================================
Edit the CONFIG block below, then run:   py backtest_tuner.py

No full pipeline needed. Loads directly from features.parquet.
Results saved to RESULTS_DIR (CSV) and printed to console.

HOW FEATURE WEIGHTS WORK
-------------------------
Each feature group has a weight multiplier applied before training.
  1.0 = normal (baseline)
  0.0 = exclude that feature group entirely
  2.0 = double the signal strength of that group
  0.5 = halve it (reduce influence)

This lets you cheaply test: "does xG matter more than rolling form?"
without retraining the whole pipeline.

WALK-FORWARD METHOD
--------------------
- Slide a window forward in time, month by month
- Train on the last TRAIN_MONTHS of data
- Test on the following TEST_MONTHS of matches
- Evaluate predicted probabilities vs actual outcomes
- Repeat until end of dataset
- Report mean accuracy and Brier score per market

Run a few configs overnight and compare the RESULTS CSV to find best weights.
"""

import os
import sys
import json
import warnings
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ============================================================
#  CONFIG — edit this section, leave everything else alone
# ============================================================

# --- PATHS ---
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
FEATURES_FILE = DATA_DIR / "features.parquet"
RESULTS_DIR = Path(__file__).parent / "backtest_results"

# --- FEATURE GROUP WEIGHTS ---
# 1.0 = normal | 0.0 = exclude | 2.0 = double | 0.5 = halve
FEATURE_WEIGHTS = {
    "xg":           1.0,   # Home_xG, Away_xG
    "elo":          1.0,   # Elo_Home/Away/Diff + Elo_Mom_*
    "rolling_ma3":  1.0,   # *_ma3 features
    "rolling_ma5":  1.0,   # *_ma5 features
    "rolling_ma10": 1.0,   # *_ma10 features
    "rolling_ma20": 0.5,   # *_ma20 features (fewer matches, noisier)
    "ewm":          1.0,   # *_ewm features
    "btts_rate":    1.0,   # Home/Away_BTTS_rate*
    "clean_sheet":  1.0,   # Home/Away_CleanSheet_rate*
    "fts_rate":     1.0,   # Home/Away_FTS_rate* (failed to score)
    "h2h":          1.0,   # H2H_* features
    "match_stats":  1.0,   # Shots, Possession, Corners, Cards, etc.
    "odds":         1.0,   # B365_Impl_H/D/A, B365_Overround
    "contextual":   0.5,   # DayOfWeek, IsWeekend, Month, SeasonProgress, RestDiff
}

# --- MODEL ---
# Options: "lightgbm", "xgboost", "random_forest", "extra_trees", "logistic", "ensemble"
# "ensemble" trains LGB + RF and averages probabilities
MODEL = "lightgbm"
N_ESTIMATORS = 200       # Trees (ignored for logistic)
MAX_DEPTH = 6            # Tree depth (-1 = unlimited for LGB)
LEARNING_RATE = 0.05     # LGB/XGB only
MIN_CHILD_SAMPLES = 20   # LGB: min samples per leaf (regularisation)
SUBSAMPLE = 0.8          # Row subsampling per tree
COLSAMPLE = 0.8          # Feature subsampling per tree

# --- WALK-FORWARD BACKTEST ---
TRAIN_MONTHS = 18        # Months of history to train on each fold
TEST_MONTHS = 1          # Months per test window
MIN_TRAIN_MATCHES = 1500 # Don't test until at least this many training rows
MAX_FOLDS = 99           # Safety cap — set lower to quick-test

# --- MARKETS TO EVALUATE ---
# Each entry: (column_name, task_type)
#   task_type = "binary" or "multiclass"
MARKETS = [
    ("y_1X2",    "multiclass"),  # H/D/A
    ("y_BTTS",   "binary"),      # 0/1
    ("y_OU_2_5", "binary"),      # 0/1
    ("y_OU_1_5", "binary"),
    ("y_OU_3_5", "binary"),
    ("y_HomeTG_1_5", "binary"),
    ("y_AwayTG_1_5", "binary"),
]

# --- OUTPUT ---
SAVE_RESULTS = True       # Save per-fold detail CSV + summary CSV
VERBOSE = True            # Print fold-by-fold progress
RUN_TAG = ""              # Optional tag appended to output filename (e.g. "xg_boost")

# --- GRID SEARCH (optional) ---
# Set GRID_SEARCH = True to run multiple configs and rank them.
# Each dict in GRID overrides values in FEATURE_WEIGHTS for that run.
# Slower — each entry is a full backtest. Good for overnight runs.
GRID_SEARCH = False
GRID = [
    {"xg": 0.0, "rolling_ma3": 1.0},   # No xG
    {"xg": 2.0, "rolling_ma3": 1.0},   # Double xG
    {"xg": 1.0, "rolling_ma3": 2.0},   # Double ma3 rolling
    {"xg": 1.0, "rolling_ma3": 0.0, "rolling_ma5": 0.0, "rolling_ma10": 2.0},  # Long window only
    {"elo": 2.0},                        # Double Elo
    {"elo": 0.0},                        # No Elo
    {"odds": 0.0},                       # No odds features
    {"h2h": 0.0},                        # No H2H
    {"match_stats": 0.0},                # No match stats
]

# ============================================================
#  END CONFIG
# ============================================================


# ---------- Feature group definitions ----------

FEATURE_GROUPS = {
    "xg":          ["Home_xG", "Away_xG"],
    "elo":         ["Elo_Home", "Elo_Away", "Elo_Diff",
                    "Elo_Mom_Home", "Elo_Mom_Away", "Elo_Mom_Diff"],
    "rolling_ma3": None,   # resolved dynamically
    "rolling_ma5": None,
    "rolling_ma10": None,
    "rolling_ma20": None,
    "ewm":         None,
    "btts_rate":   None,
    "clean_sheet": None,
    "fts_rate":    None,
    "h2h":         None,
    "match_stats": ["Home_Shots", "Away_Shots", "Home_ShotsT", "Away_ShotsT",
                    "Home_ShotsInBox", "Away_ShotsInBox", "Home_Possession",
                    "Away_Possession", "Home_Corners", "Away_Corners",
                    "Home_CardsY", "Away_CardsY", "Home_CardsR", "Away_CardsR",
                    "Home_PassAcc", "Away_PassAcc", "Home_BigChances",
                    "Away_BigChances"],
    "odds":        ["B365_Impl_H", "B365_Impl_D", "B365_Impl_A", "B365_Overround",
                    "PSCH", "PSCD", "PSCA"],
    "contextual":  ["DayOfWeek", "IsWeekend", "Month", "SeasonProgress",
                    "RestDiff", "Home_RestDays", "Away_RestDays"],
}


def resolve_groups(all_cols):
    """Fill in dynamically detected groups from column list."""
    g = dict(FEATURE_GROUPS)
    g["rolling_ma3"]  = [c for c in all_cols if "_ma3"  in c and not c.startswith("y_")]
    g["rolling_ma5"]  = [c for c in all_cols if "_ma5"  in c and not c.startswith("y_")]
    g["rolling_ma10"] = [c for c in all_cols if "_ma10" in c and not c.startswith("y_")]
    g["rolling_ma20"] = [c for c in all_cols if "_ma20" in c and not c.startswith("y_")]
    g["ewm"]          = [c for c in all_cols if "_ewm"  in c and not c.startswith("y_")]
    g["btts_rate"]    = [c for c in all_cols if "BTTS_rate" in c]
    g["clean_sheet"]  = [c for c in all_cols if "CleanSheet_rate" in c]
    g["fts_rate"]     = [c for c in all_cols if "FTS_rate" in c]
    g["h2h"]          = [c for c in all_cols if c.startswith("H2H_")]
    return g


def apply_weights(X: pd.DataFrame, groups: dict, weights: dict) -> pd.DataFrame:
    """Scale feature groups by their weights. 0.0 = drop columns."""
    X = X.copy()
    # Track which columns have been assigned
    assigned = set()
    for group, cols in groups.items():
        if not cols:
            continue
        w = weights.get(group, 1.0)
        existing = [c for c in cols if c in X.columns]
        if not existing:
            continue
        assigned.update(existing)
        if w == 0.0:
            X = X.drop(columns=existing)
        elif w != 1.0:
            X[existing] = X[existing] * w
    return X


def get_model(task: str, weights_override: dict = None):
    """Return an untrained sklearn-compatible model."""
    w = weights_override or {}
    n_est = w.get("n_estimators", N_ESTIMATORS)
    lr    = w.get("learning_rate", LEARNING_RATE)
    depth = w.get("max_depth", MAX_DEPTH)
    sub   = w.get("subsample", SUBSAMPLE)
    col   = w.get("colsample", COLSAMPLE)
    mcs   = w.get("min_child_samples", MIN_CHILD_SAMPLES)

    multiclass = task == "multiclass"
    obj_lgb = "multiclass" if multiclass else "binary"
    obj_xgb = "multi:softprob" if multiclass else "binary:logistic"

    if MODEL == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=n_est, learning_rate=lr, max_depth=depth,
            subsample=sub, colsample_bytree=col,
            min_child_samples=mcs, objective=obj_lgb,
            num_class=3 if multiclass else 1,
            verbose=-1, n_jobs=-1, random_state=42
        )
    elif MODEL == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=n_est, learning_rate=lr, max_depth=depth,
            subsample=sub, colsample_bytree=col,
            objective=obj_xgb, eval_metric="logloss",
            use_label_encoder=False, verbosity=0,
            n_jobs=-1, random_state=42
        )
    elif MODEL == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=n_est, max_depth=None if depth < 0 else depth,
            n_jobs=-1, random_state=42
        )
    elif MODEL == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            n_estimators=n_est, max_depth=None if depth < 0 else depth,
            n_jobs=-1, random_state=42
        )
    elif MODEL == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    elif MODEL == "ensemble":
        return None  # handled separately
    else:
        raise ValueError(f"Unknown MODEL: {MODEL}")


def train_predict(X_train, y_train, X_test, task, feature_weights):
    """Train model and return predicted probabilities for X_test."""
    # Apply feature weights
    groups = resolve_groups(list(X_train.columns))
    X_tr = apply_weights(X_train, groups, feature_weights)
    X_te = apply_weights(X_test,  groups, feature_weights)

    # Align columns after potential drops
    shared_cols = [c for c in X_tr.columns if c in X_te.columns]
    X_tr = X_tr[shared_cols].fillna(0)
    X_te = X_te[shared_cols].fillna(0)

    if X_tr.empty or len(X_tr) < 100:
        return None

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    if MODEL == "ensemble":
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import RandomForestClassifier
        multiclass = task == "multiclass"
        lgb = LGBMClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH, subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE, min_child_samples=MIN_CHILD_SAMPLES,
            objective="multiclass" if multiclass else "binary",
            num_class=3 if multiclass else 1,
            verbose=-1, n_jobs=-1, random_state=42
        )
        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=42
        )
        lgb.fit(X_tr, y_enc)
        rf.fit(X_tr, y_enc)
        probs_lgb = lgb.predict_proba(X_te)
        probs_rf  = rf.predict_proba(X_te)
        probs = (probs_lgb + probs_rf) / 2
    else:
        clf = get_model(task)
        clf.fit(X_tr, y_enc)
        probs = clf.predict_proba(X_te)

    return probs, le


def evaluate(probs, le, y_test, task):
    """Return accuracy, Brier/log-loss, and optional AUC."""
    y_enc = le.transform(y_test)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_enc, preds)

    if task == "binary":
        brier = brier_score_loss(y_enc, probs[:, 1])
        try:
            auc = roc_auc_score(y_enc, probs[:, 1])
        except Exception:
            auc = float("nan")
        ll = log_loss(y_enc, probs)
    else:
        # Multiclass: mean Brier
        n_classes = probs.shape[1]
        brier = 0.0
        for c in range(n_classes):
            brier += brier_score_loss((y_enc == c).astype(int), probs[:, c])
        brier /= n_classes
        auc = float("nan")
        ll = log_loss(y_enc, probs)

    return {"accuracy": acc, "brier": brier, "log_loss": ll, "auc": auc, "n": len(y_enc)}


def run_backtest(feature_weights: dict, tag: str = "") -> pd.DataFrame:
    """
    Run full walk-forward backtest with given feature weights.
    Returns DataFrame of fold-level results.
    """
    if VERBOSE:
        print(f"\n{'='*60}")
        print(f"  BACKTEST: {tag or 'default'}")
        print(f"  Model: {MODEL} | Train window: {TRAIN_MONTHS}m | Test: {TEST_MONTHS}m")
        print(f"  Markets: {[m for m,_ in MARKETS]}")
        print(f"{'='*60}")

    df = pd.read_parquet(FEATURES_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Non-feature / non-target columns to exclude
    META_COLS = {"fixture_id", "League", "Date", "Season", "HomeTeam", "AwayTeam",
                 "FTHG", "FTAG", "FTR", "HTHG", "HTAG",
                 "B365H", "B365D", "B365A"}
    target_cols = {m for m, _ in MARKETS}
    y_cols = {c for c in df.columns if c.startswith("y_")}
    drop_cols = META_COLS | (y_cols - target_cols)

    feature_cols = [c for c in df.columns if c not in drop_cols and not c.startswith("y_")]

    start_date = df["Date"].min()
    end_date   = df["Date"].max()

    # Build sliding windows
    from pandas.tseries.offsets import DateOffset
    windows = []
    train_start = start_date
    fold_n = 0
    while True:
        train_end = train_start + DateOffset(months=TRAIN_MONTHS)
        test_start = train_end
        test_end   = test_start + DateOffset(months=TEST_MONTHS)

        if test_end > end_date + DateOffset(days=1):
            break
        if fold_n >= MAX_FOLDS:
            break

        windows.append((train_start, train_end, test_start, test_end))
        train_start = train_start + DateOffset(months=TEST_MONTHS)
        fold_n += 1

    if not windows:
        print("ERROR: Not enough data to build any backtest folds.")
        print(f"  Data range: {start_date.date()} to {end_date.date()}")
        print(f"  Need at least {TRAIN_MONTHS + TEST_MONTHS} months.")
        return pd.DataFrame()

    all_results = []

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        train_mask = (df["Date"] >= tr_start) & (df["Date"] < tr_end)
        test_mask  = (df["Date"] >= te_start) & (df["Date"] < te_end)

        df_train = df[train_mask]
        df_test  = df[test_mask]

        if len(df_train) < MIN_TRAIN_MATCHES or len(df_test) < 30:
            if VERBOSE:
                print(f"  Fold {fold_idx+1}: skip (train={len(df_train)}, test={len(df_test)})")
            continue

        X_train = df_train[feature_cols]
        X_test  = df_test[feature_cols]

        if VERBOSE:
            print(f"\n  Fold {fold_idx+1}: train {tr_start.date()}–{tr_end.date()} "
                  f"({len(df_train)} rows) | test {te_start.date()}–{te_end.date()} "
                  f"({len(df_test)} rows)")

        for market, task in MARKETS:
            if market not in df.columns:
                continue

            y_train = df_train[market].dropna()
            y_test  = df_test[market].dropna()

            if len(y_test) < 20 or len(y_train.unique()) < 2:
                continue

            X_tr = X_train.loc[y_train.index]
            X_te = X_test.loc[y_test.index]

            result = train_predict(X_tr, y_train, X_te, task, feature_weights)
            if result is None:
                continue

            probs, le = result
            metrics = evaluate(probs, le, y_test, task)

            row = {
                "tag":        tag or "default",
                "fold":       fold_idx + 1,
                "train_from": tr_start.date(),
                "train_to":   tr_end.date(),
                "test_from":  te_start.date(),
                "test_to":    te_end.date(),
                "market":     market,
                "task":       task,
                **metrics,
            }
            all_results.append(row)

            if VERBOSE:
                auc_str = f" | AUC {metrics['auc']:.4f}" if not np.isnan(metrics["auc"]) else ""
                print(f"    {market:<18} acc={metrics['accuracy']:.4f}  "
                      f"brier={metrics['brier']:.4f}  ll={metrics['log_loss']:.4f}"
                      f"{auc_str}  n={metrics['n']}")

    return pd.DataFrame(all_results)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold results to per-market summary."""
    if results.empty:
        return results
    summary = (
        results.groupby(["tag", "market"])
        .agg(
            folds=("fold", "count"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_brier=("brier", "mean"),
            mean_logloss=("log_loss", "mean"),
            mean_auc=("auc", lambda x: x[~np.isnan(x)].mean() if any(~np.isnan(x)) else float("nan")),
            total_matches=("n", "sum"),
        )
        .reset_index()
        .sort_values(["market", "mean_accuracy"], ascending=[True, False])
    )
    return summary


def print_summary(summary: pd.DataFrame):
    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Tag':<16} {'Market':<18} {'Acc':>6} {'±':>5} {'Brier':>7} {'LL':>7} {'AUC':>6} {'N':>6}")
    print(f"  {'-'*69}")
    for _, row in summary.iterrows():
        auc = f"{row.mean_auc:.4f}" if not np.isnan(row.mean_auc) else "  —   "
        print(f"  {str(row.tag):<16} {row.market:<18} "
              f"{row.mean_accuracy:.4f} {row.std_accuracy:.3f} "
              f"{row.mean_brier:.4f} {row.mean_logloss:.4f} {auc:>6} "
              f"{int(row.total_matches):>6}")
    print()


def save_results(results: pd.DataFrame, summary: pd.DataFrame, tag: str):
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"_{tag}" if tag else ""
    detail_path  = RESULTS_DIR / f"detail{slug}_{ts}.csv"
    summary_path = RESULTS_DIR / f"summary{slug}_{ts}.csv"
    results.to_csv(detail_path,  index=False)
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {detail_path}")
    print(f"  Saved: {summary_path}")


# ---------- GRID SEARCH ----------

def run_grid():
    """Run multiple configs from GRID and compare summaries."""
    all_summaries = []

    for i, overrides in enumerate(GRID):
        # Build weights by merging base FEATURE_WEIGHTS with overrides
        w = dict(FEATURE_WEIGHTS)
        w.update(overrides)
        tag = "grid_" + "_".join(f"{k}{v}" for k, v in overrides.items())
        print(f"\n[{i+1}/{len(GRID)}] {tag}")
        results = run_backtest(feature_weights=w, tag=tag)
        if results.empty:
            continue
        summary = summarise(results)
        all_summaries.append(summary)
        if SAVE_RESULTS:
            save_results(results, summary, tag)

    if all_summaries:
        combined = pd.concat(all_summaries)
        print_summary(combined)
        if SAVE_RESULTS:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = RESULTS_DIR / f"grid_comparison_{ts}.csv"
            combined.to_csv(path, index=False)
            print(f"\n  Grid comparison saved: {path}")


# ---------- MAIN ----------

if __name__ == "__main__":
    # Validate data file
    if not FEATURES_FILE.exists():
        print(f"ERROR: features.parquet not found at:\n  {FEATURES_FILE}")
        print("Run the main pipeline once to build it:")
        print("  py run_weekly.py --speed fast")
        sys.exit(1)

    tag = RUN_TAG

    if GRID_SEARCH:
        print("Grid search mode — running all configs in GRID...")
        run_grid()
    else:
        results = run_backtest(feature_weights=FEATURE_WEIGHTS, tag=tag)
        if results.empty:
            print("No results produced.")
            sys.exit(1)
        summary = summarise(results)
        print_summary(summary)
        if SAVE_RESULTS:
            save_results(results, summary, tag)
