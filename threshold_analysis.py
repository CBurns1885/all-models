"""
Per-market confidence threshold analysis.

For each market, sweeps confidence thresholds (50%-99%) and finds the threshold
that maximises accuracy subject to:
  - Fair ROI >= 0  (model edge over fair-odds market)
  - n_predictions >= MIN_PREDS  (enough bets to be meaningful)

Fair ROI formula per bet:
  correct: return = 1/confidence - 1   (decimal fair odds minus stake)
  wrong:   return = -1
  roi = mean(returns across all bets)

Run after a low-threshold 4-week backtest to build the full prediction cache:
  py market_backtest.py --weeks 4 --min-confidence 0.01
  py threshold_analysis.py

Output: outputs/threshold_analysis.csv + printed table of suggested thresholds.
"""

import sys, pathlib, argparse
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

from market_backtest import MARKET_CONFIGS

MIN_PREDS   = 20          # minimum predictions per 4-week period to be useful
SWEEP       = [t / 100 for t in range(50, 100)]   # 0.50 … 0.99


def load_cache(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            "Run first: py market_backtest.py --weeks 4 --min-confidence 0.01"
        )
    return pd.read_parquet(path)


def fair_roi(correct_series: pd.Series, confidence_series: pd.Series) -> float:
    """Fair-odds ROI: assumes we bet at 1/confidence decimal odds."""
    returns = np.where(correct_series, 1.0 / confidence_series - 1.0, -1.0)
    return float(returns.mean())


def analyse_market(df: pd.DataFrame, market_name: str, config: dict) -> dict:
    actual_col = config['actual_col']
    pred_cols  = config['pred_cols']
    outcomes   = config['outcomes']

    # Prefer BLEND cols over P_ cols
    blend_cols = [c.replace('P_', 'BLEND_') for c in pred_cols if c.startswith('P_')]
    if blend_cols and all(c in df.columns for c in blend_cols):
        use_cols = blend_cols
        src = 'BLEND'
    elif all(c in df.columns for c in pred_cols):
        use_cols = pred_cols
        src = 'ML'
    else:
        return {'market': market_name, 'error': 'missing pred cols'}

    if actual_col not in df.columns:
        return {'market': market_name, 'error': f'missing actual col {actual_col}'}

    col_to_outcome = dict(zip(use_cols, outcomes))
    df2 = df.copy()
    df2['MaxProb'] = df2[use_cols].max(axis=1)
    df2['PredOutcome'] = df2[use_cols].idxmax(axis=1).map(col_to_outcome)
    df2['Correct'] = df2['PredOutcome'] == df2[actual_col].astype(str)

    rows = []
    for t in SWEEP:
        sub = df2[df2['MaxProb'] >= t]
        n = len(sub)
        if n < MIN_PREDS:
            rows.append({'threshold': t, 'n_preds': n, 'accuracy': np.nan, 'roi': np.nan})
        else:
            acc = sub['Correct'].mean()
            roi = fair_roi(sub['Correct'], sub['MaxProb'])
            rows.append({'threshold': t, 'n_preds': n, 'accuracy': acc, 'roi': roi})

    sweep_df = pd.DataFrame(rows)

    # --- Optimal: max accuracy where ROI >= 0 and n >= MIN_PREDS
    positive_roi = sweep_df[(sweep_df['roi'] >= 0.0) & (sweep_df['n_preds'] >= MIN_PREDS)]
    if len(positive_roi):
        best_row  = positive_roi.loc[positive_roi['accuracy'].idxmax()]
        opt_thr   = best_row['threshold']
        opt_acc   = best_row['accuracy']
        opt_roi   = best_row['roi']
        opt_preds = int(best_row['n_preds'])
    else:
        opt_thr = opt_acc = opt_roi = opt_preds = None

    # --- Current baseline at 70%
    base = sweep_df[sweep_df['threshold'] == 0.70]
    if len(base):
        base_row  = base.iloc[0]
        base_acc  = base_row['accuracy']
        base_roi  = base_row['roi']
        base_n    = int(base_row['n_preds'])
    else:
        base_acc = base_roi = base_n = None

    # --- Max achievable accuracy (ignoring ROI)
    valid = sweep_df.dropna(subset=['accuracy'])
    max_acc   = valid['accuracy'].max() if len(valid) else None
    max_acc_t = valid.loc[valid['accuracy'].idxmax(), 'threshold'] if len(valid) else None

    return {
        'market':     market_name,
        'source':     src,
        # Current 70% baseline
        'base_n':     base_n,
        'base_acc':   round(base_acc * 100, 1)  if base_acc  is not None else None,
        'base_roi':   round(base_roi * 100, 1)  if base_roi  is not None else None,
        # Best threshold for +ROI + volume
        'opt_threshold': round(opt_thr * 100, 0) if opt_thr is not None else None,
        'opt_acc':    round(opt_acc * 100, 1)   if opt_acc   is not None else None,
        'opt_roi':    round(opt_roi * 100, 1)   if opt_roi   is not None else None,
        'opt_preds':  opt_preds,
        # Absolute max accuracy (no ROI constraint)
        'max_acc':    round(max_acc * 100, 1)   if max_acc   is not None else None,
        'max_acc_thr':round(max_acc_t * 100, 0) if max_acc_t is not None else None,
        # Full sweep for plotting
        'sweep':      sweep_df,
    }


def print_table(results: list):
    print(f"\n{'Market':<22} {'Src':<6} {'--- Baseline 70% ---':^22} {'--- Optimal (+ROI) ---':^30} {'MaxAcc':>7}")
    print(f"{'':22} {'':6} {'Preds':>6} {'Acc':>7} {'ROI':>7}  {'Thr':>5} {'Preds':>6} {'Acc':>7} {'ROI':>7}  {'%':>7}")
    print('-' * 100)
    for r in results:
        if 'error' in r:
            print(f"  {r['market']:<20} ERROR: {r['error']}")
            continue
        bn   = r['base_n']    or 0
        ba   = f"{r['base_acc']:.1f}%"  if r['base_acc']  is not None else '—'
        br   = f"{r['base_roi']:+.1f}%" if r['base_roi']  is not None else '—'
        if r['opt_threshold'] is not None:
            ot   = f"{r['opt_threshold']:.0f}%"
            oa   = f"{r['opt_acc']:.1f}%"
            oro  = f"{r['opt_roi']:+.1f}%"
            op   = r['opt_preds']
        else:
            ot = oa = oro = '—'; op = 0
        mx   = f"{r['max_acc']:.1f}%" if r['max_acc'] is not None else '—'
        print(f"  {r['market']:<20} {r['source']:<6} {bn:>6} {ba:>7} {br:>7}  {ot:>5} {op:>6} {oa:>7} {oro:>7}  {mx:>7}")


def analyse_by_league(df: pd.DataFrame, min_preds_league: int = 5) -> dict:
    """
    Run threshold analysis per league per market.
    Returns dict: league -> market -> optimal_threshold (as fraction 0-1, or None).
    Also saves outputs/league_thresholds.json.
    """
    import json

    if 'League' not in df.columns:
        print("[WARN] No League column — skipping per-league analysis")
        return {}

    leagues = sorted(df['League'].unique())
    print(f"\n=== Per-League Threshold Analysis ({len(leagues)} leagues) ===")
    print(f"  Min preds per league-market: {min_preds_league}")

    league_thresholds = {}   # league -> market -> float threshold
    league_rows = []         # for CSV

    old_min = MIN_PREDS

    for league in leagues:
        df_lg = df[df['League'] == league]
        league_thresholds[league] = {}

        for market_name, config in MARKET_CONFIGS.items():
            # temporarily lower MIN_PREDS for per-league runs
            result = analyse_market.__wrapped__(df_lg, market_name, config, min_preds_override=min_preds_league) \
                if hasattr(analyse_market, '__wrapped__') else _analyse_market_inner(df_lg, market_name, config, min_preds_league)

            opt = result.get('opt_threshold')
            if opt is not None:
                league_thresholds[league][market_name] = round(opt / 100, 2)
            league_rows.append({
                'League': league,
                'Market': market_name,
                'Preds_70pct': result.get('base_n', 0),
                'Acc_70pct': result.get('base_acc'),
                'ROI_70pct': result.get('base_roi'),
                'Opt_Threshold': opt,
                'Opt_Preds': result.get('opt_preds'),
                'Opt_Acc': result.get('opt_acc'),
                'Opt_ROI': result.get('opt_roi'),
            })

    # Save JSON
    json_path = ROOT / 'outputs' / 'league_thresholds.json'
    json_path.write_text(json.dumps(league_thresholds, indent=2))
    print(f"[OK] Saved league thresholds -> {json_path}")
    print(f"     {sum(len(v) for v in league_thresholds.values())} league-market combos with +ROI thresholds")

    # Save CSV
    lg_df = pd.DataFrame(league_rows)
    lg_path = ROOT / 'outputs' / 'league_threshold_analysis.csv'
    lg_df.to_csv(lg_path, index=False)
    print(f"[OK] Saved league threshold CSV -> {lg_path}")

    # Print summary: markets with per-league thresholds
    print("\n--- Markets with per-league +ROI thresholds ---")
    for market_name in MARKET_CONFIGS:
        leagues_with_thresh = [lg for lg, md in league_thresholds.items() if market_name in md]
        if leagues_with_thresh:
            print(f"  {market_name}: {len(leagues_with_thresh)} leagues — {', '.join(leagues_with_thresh[:8])}"
                  + ("..." if len(leagues_with_thresh) > 8 else ""))

    return league_thresholds


def _analyse_market_inner(df: pd.DataFrame, market_name: str, config: dict, min_preds_override: int) -> dict:
    """Same as analyse_market but with a custom MIN_PREDS override."""
    actual_col = config['actual_col']
    pred_cols  = config['pred_cols']
    outcomes   = config['outcomes']
    blend_cols = [c.replace('P_', 'BLEND_') for c in pred_cols if c.startswith('P_')]
    if blend_cols and all(c in df.columns for c in blend_cols):
        use_cols = blend_cols
        src = 'BLEND'
    elif all(c in df.columns for c in pred_cols):
        use_cols = pred_cols
        src = 'ML'
    else:
        return {'market': market_name, 'error': 'missing pred cols'}
    if actual_col not in df.columns:
        return {'market': market_name, 'error': f'missing actual col {actual_col}'}

    col_to_outcome = dict(zip(use_cols, outcomes))
    df2 = df.copy()
    df2['MaxProb'] = df2[use_cols].max(axis=1)
    df2['PredOutcome'] = df2[use_cols].idxmax(axis=1).map(col_to_outcome)
    df2['Correct'] = df2['PredOutcome'] == df2[actual_col].astype(str)

    rows = []
    for t in SWEEP:
        sub = df2[df2['MaxProb'] >= t]
        n = len(sub)
        if n < min_preds_override:
            rows.append({'threshold': t, 'n_preds': n, 'accuracy': np.nan, 'roi': np.nan})
        else:
            rows.append({'threshold': t, 'n_preds': n,
                         'accuracy': sub['Correct'].mean(),
                         'roi': fair_roi(sub['Correct'], sub['MaxProb'])})

    sweep_df = pd.DataFrame(rows)
    positive_roi = sweep_df[(sweep_df['roi'] >= 0.0) & (sweep_df['n_preds'] >= min_preds_override)]
    if len(positive_roi):
        best = positive_roi.loc[positive_roi['accuracy'].idxmax()]
        opt_thr, opt_acc, opt_roi, opt_preds = best['threshold'], best['accuracy'], best['roi'], int(best['n_preds'])
    else:
        opt_thr = opt_acc = opt_roi = opt_preds = None

    base = sweep_df[sweep_df['threshold'] == 0.70]
    base_row = base.iloc[0] if len(base) else None

    valid = sweep_df.dropna(subset=['accuracy'])
    max_acc = valid['accuracy'].max() if len(valid) else None

    return {
        'market': market_name, 'source': src,
        'base_n': int(base_row['n_preds']) if base_row is not None else None,
        'base_acc': round(base_row['accuracy'] * 100, 1) if base_row is not None and not np.isnan(base_row['accuracy']) else None,
        'base_roi': round(base_row['roi'] * 100, 1) if base_row is not None and not np.isnan(base_row['roi']) else None,
        'opt_threshold': round(opt_thr * 100, 0) if opt_thr is not None else None,
        'opt_acc': round(opt_acc * 100, 1) if opt_acc is not None else None,
        'opt_roi': round(opt_roi * 100, 1) if opt_roi is not None else None,
        'opt_preds': opt_preds,
        'max_acc': round(max_acc * 100, 1) if max_acc is not None else None,
        'sweep': sweep_df,
    }


def main():
    global MIN_PREDS
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default='outputs/tuning_preds_cache.parquet')
    parser.add_argument('--min-preds', type=int, default=MIN_PREDS)
    parser.add_argument('--by-league', action='store_true',
                        help='Also run per-league threshold analysis -> league_thresholds.json')
    parser.add_argument('--min-preds-league', type=int, default=5,
                        help='Min predictions per league-market (default: 5)')
    args = parser.parse_args()
    MIN_PREDS = args.min_preds

    cache_path = ROOT / args.cache
    print(f"Loading: {cache_path}")
    df = load_cache(cache_path)
    print(f"Loaded {len(df)} match rows\n")

    results = []
    for market_name, config in MARKET_CONFIGS.items():
        results.append(analyse_market(df, market_name, config))

    # Sort: markets with opt_threshold first (best acc), then the rest
    def sort_key(r):
        return (r.get('opt_threshold') is not None, r.get('opt_acc') or 0)
    results.sort(key=sort_key, reverse=True)

    print_table(results)

    # Save CSV
    out_rows = [{k: v for k, v in r.items() if k != 'sweep'} for r in results]
    out_df   = pd.DataFrame(out_rows)
    out_path = ROOT / 'outputs' / 'threshold_analysis.csv'
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # Print suggested config
    print(f"\n--- Suggested per-market thresholds (max acc, ROI >= 0, min preds >= {MIN_PREDS}) ---")
    for r in results:
        if 'error' in r:
            continue
        if r['opt_threshold'] is not None:
            print(f"  '{r['market']}': {r['opt_threshold']:.0f}%  "
                  f"({r['opt_preds']} preds, {r['opt_acc']}% acc, {r['opt_roi']:+.1f}% ROI)")
        else:
            print(f"  '{r['market']}': no +ROI threshold found  (max achievable acc: {r['max_acc']}%)")

    if args.by_league:
        analyse_by_league(df, min_preds_league=args.min_preds_league)


if __name__ == '__main__':
    main()
