#!/usr/bin/env python3
"""
Data sanity check — read-only snapshot of the historical training data.

Run on session start (via .claude/hooks/session-start.sh) so any Claude Code
session immediately sees what data it is actually working against, instead of
silently running the model on a stale, partial, or missing parquet.

Never fails the session: always exits 0, never mutates anything.

Manual use:
    python data_sanity_check.py
    python data_sanity_check.py --top 20      # show more leagues
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Read-only data snapshot")
    parser.add_argument('--top', type=int, default=12,
                        help='How many leagues to list by match count (default: 12)')
    args = parser.parse_args()

    print("=" * 64)
    print(" DATA SANITY CHECK")
    print("=" * 64)

    # Resolve the real parquet path from config (single source of truth)
    try:
        import config
        hist_path = Path(config.HISTORICAL_PARQUET)
        feat_path = Path(config.FEATURES_PARQUET)
    except Exception as e:
        print(f" [WARN] Could not import config to resolve data paths: {e}")
        print("=" * 64)
        return 0

    try:
        import pandas as pd
    except ImportError:
        print(" [WARN] pandas not installed — cannot inspect data.")
        print("        Run: pip install -r requirements.txt")
        print("=" * 64)
        return 0

    # --- Historical matches parquet ---
    print(f" Historical parquet: {hist_path}")
    if not hist_path.exists():
        print(" [MISSING] No historical_matches.parquet found.")
        print("           Pull data first:  python data_ingest.py")
        print(f"           (config expects it at {hist_path})")
        print("=" * 64)
        return 0

    try:
        df = pd.read_parquet(hist_path)
    except Exception as e:
        print(f" [ERROR] Failed to read parquet: {e}")
        print("=" * 64)
        return 0

    size_mb = hist_path.stat().st_size / (1024 * 1024)
    print(f" Rows: {len(df):,}   Columns: {len(df.columns)}   Size: {size_mb:.1f} MB")

    # Date range
    if 'Date' in df.columns:
        try:
            dates = pd.to_datetime(df['Date'], errors='coerce').dropna()
            if len(dates):
                span_days = (dates.max() - dates.min()).days
                print(f" Date range: {dates.min().date()} -> {dates.max().date()}  ({span_days/365:.1f} yrs)")
                # Freshness flag — warn if newest match is old
                from datetime import datetime
                age_days = (datetime.now() - dates.max()).days
                if age_days > 30:
                    print(f" [STALE] Newest match is {age_days} days old — run data_ingest.py to refresh.")
        except Exception:
            pass

    # Season span
    if 'Season' in df.columns:
        seasons = sorted(str(s) for s in df['Season'].dropna().unique())
        if seasons:
            print(f" Seasons ({len(seasons)}): {', '.join(seasons)}")

    # League breakdown
    if 'League' in df.columns:
        counts = df['League'].value_counts()
        print(f" Leagues: {len(counts)}")
        print(f"\n Top {min(args.top, len(counts))} leagues by match count:")
        for lg, n in counts.head(args.top).items():
            print(f"   {str(lg):<10} {n:>7,}")
        if len(counts) > args.top:
            print(f"   ... and {len(counts) - args.top} more")

    # Completeness flags for key columns the pipeline needs
    print("\n Key column coverage:")
    for col in ['FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']:
        if col in df.columns:
            pct = 100.0 * df[col].notna().mean()
            flag = '' if pct >= 90 else '  [LOW]'
            print(f"   {col:<8} {pct:5.1f}% present{flag}")
        else:
            print(f"   {col:<8} MISSING COLUMN")

    # --- Features parquet ---
    print()
    if feat_path.exists():
        try:
            fdf = pd.read_parquet(feat_path, columns=None)
            n_targets = sum(1 for c in fdf.columns if str(c).startswith('y_'))
            print(f" Features parquet: present — {len(fdf):,} rows, {len(fdf.columns)} cols, {n_targets} y_* targets")
        except Exception as e:
            print(f" Features parquet: present but unreadable ({e})")
    else:
        print(" Features parquet: NOT built yet — run features.build_features() / the pipeline.")

    print("=" * 64)
    return 0


if __name__ == '__main__':
    sys.exit(main())
