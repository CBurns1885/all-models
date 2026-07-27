# market_anchor.py
"""
Market anchoring — turn "predict football" into "find mispricings".

The strongest single predictor available is the betting market itself. Once a
fixture has real odds in the fixture_odds table, this module:

  1. De-margins the bookmaker odds into implied probabilities
     (proportional/multiplicative overround removal per market).
  2. Writes MKT_* implied-probability columns and EDGE_* columns
     (model probability minus market probability) into the predictions frame.
  3. Optionally SHRINKS the model's BLEND_* probabilities toward the market:
         p_final = (1-w) * p_model + w * p_market
     with w = TUNING_OVERRIDES['market_anchor_weight'] (default 0.0 = off,
     i.e. purely diagnostic until enough odds accumulate to tune w honestly).

EDGE_* columns are the basis for value betting: a persistent, calibrated
positive edge is the only thing worth staking on. Coverage is currently low
(odds are only fetched live pre-match), so this starts as instrumentation
and becomes decisive as the odds table grows week by week.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# Market -> (model column stems, ODDS_* column names) — order-aligned.
# Model stems are tried with BLEND_ first, then P_.
ANCHOR_MARKETS: Dict[str, List[tuple]] = {
    "1X2": [("1X2_H", "ODDS_1X2_H"), ("1X2_D", "ODDS_1X2_D"), ("1X2_A", "ODDS_1X2_A")],
    "BTTS": [("BTTS_Y", "ODDS_BTTS_Y"), ("BTTS_N", "ODDS_BTTS_N")],
    **{f"OU_{l}": [(f"OU_{l}_O", f"ODDS_OU_{l}_O"), (f"OU_{l}_U", f"ODDS_OU_{l}_U")]
       for l in ("0_5", "1_5", "2_5", "3_5", "4_5", "5_5")},
}


def _implied_probs(odds: np.ndarray) -> np.ndarray:
    """De-margined implied probabilities for one market's odds vector.
    Proportional (multiplicative) overround removal; NaN-safe."""
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 / odds
    total = np.nansum(raw)
    if not np.isfinite(total) or total <= 0 or np.isnan(raw).any():
        return np.full_like(odds, np.nan)
    return raw / total


def apply_market_anchor(df: pd.DataFrame, anchor_weight: float = None) -> pd.DataFrame:
    """Add MKT_*/EDGE_* columns and optionally shrink BLEND_* toward market.

    Requires ODDS_* columns (merge via odds_utils.merge_odds_into_df first).
    Rows without odds are left untouched. Returns the modified frame.
    """
    odds_cols = [c for c in df.columns if c.startswith("ODDS_")]
    if not odds_cols:
        return df

    if anchor_weight is None:
        try:
            from predict import TUNING_OVERRIDES
            anchor_weight = float(TUNING_OVERRIDES.get("market_anchor_weight", 0.0))
        except Exception:
            anchor_weight = 0.0
    anchor_weight = float(np.clip(anchor_weight, 0.0, 1.0))

    df = df.copy()
    n_anchored = 0

    for market, pairs in ANCHOR_MARKETS.items():
        stems = [p[0] for p in pairs]
        ocols = [p[1] for p in pairs]
        if not all(c in df.columns for c in ocols):
            continue
        # Prefer blended model probabilities, fall back to raw ML
        mcols = None
        for prefix in ("BLEND_", "P_"):
            cand = [f"{prefix}{s}" for s in stems]
            if all(c in df.columns for c in cand):
                mcols = cand
                break
        if mcols is None:
            continue

        for i in df.index:
            odds = df.loc[i, ocols].to_numpy(dtype=float)
            if np.isnan(odds).any() or (odds <= 1.0).any():
                continue
            imp = _implied_probs(odds)
            if np.isnan(imp).any():
                continue
            model_p = df.loc[i, mcols].to_numpy(dtype=float)
            if np.isnan(model_p).any() or model_p.sum() <= 0:
                continue
            model_p = model_p / model_p.sum()

            for j, stem in enumerate(stems):
                df.loc[i, f"MKT_{stem}"] = imp[j]
                df.loc[i, f"EDGE_{stem}"] = model_p[j] - imp[j]

            if anchor_weight > 0:
                blended = (1 - anchor_weight) * model_p + anchor_weight * imp
                df.loc[i, mcols] = blended / blended.sum()
            n_anchored += 1

    if n_anchored:
        mode = f"shrink w={anchor_weight}" if anchor_weight > 0 else "diagnostic only"
        print(f"[ANCHOR] Market anchor applied to {n_anchored} market-rows ({mode})")
    return df
