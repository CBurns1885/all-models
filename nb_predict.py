# nb_predict.py
"""
Prices corners/cards fixtures with the negative-binomial count model
(models_counts.py), mirroring dc_predict.py's role for 1X2/OU/goals markets.

Unlike DC (which only needs team names), the NB regressors need the full
engineered feature vector for each fixture, so this takes an already-built
future frame (predict.py's df_future) rather than a raw fixtures CSV.
"""
from __future__ import annotations

from typing import List

import pandas as pd

from config import FEATURES_PARQUET, log_header
from models_counts import nb_probs_for_rows, nb_supported


def build_nb_for_frame(df_future: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    """Compute NB_<target>_N / NB_<target>_Y columns for every nb_supported
    target in `targets`, aligned to df_future's row order/index.

    The NB count regressors are fit only on matches strictly before the
    earliest fixture in df_future — same anti-leakage cutoff pattern as
    dc_predict.build_dc_for_fixtures. For live predictions this filters
    nothing; for historical backtests it is essential.
    """
    log_header("NB: fitting count-distribution parameters")

    nb_targets = [t for t in targets if nb_supported(t)]
    out = pd.DataFrame(index=df_future.index)
    if not nb_targets:
        return out

    base = pd.read_parquet(FEATURES_PARQUET)
    base = base.dropna(subset=["Date"]).copy()
    base["Date"] = pd.to_datetime(base["Date"])

    cutoff = pd.to_datetime(df_future["Date"]).min()
    if pd.notna(cutoff):
        n_before = len(base)
        base = base[base["Date"] < cutoff]
        if len(base) < n_before:
            print(f"  [CUTOFF] NB training restricted to {len(base):,}/{n_before:,} "
                  f"matches before {str(cutoff)[:10]}")

    for t in nb_targets:
        probs = nb_probs_for_rows(base, df_future, t)  # (n, 2) = [N/Under, Y/Over]
        sfx = t.replace("y_", "")
        out[f"NB_{sfx}_N"] = probs[:, 0]
        out[f"NB_{sfx}_Y"] = probs[:, 1]

    print(f"  [OK] Priced {len(nb_targets)} NB targets for {len(df_future)} fixtures")
    return out
