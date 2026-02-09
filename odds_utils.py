"""
odds_utils.py - Utility functions for loading, merging, and looking up bookmaker odds.

Provides:
  - load_odds_for_fixtures(): Bulk load odds from DB, pivot to wide ODDS_* columns
  - merge_odds_into_df(): Merge odds into any DataFrame with fixture_id column
  - get_odds_for_prediction(): Look up odds for a specific bet prediction
"""

import sqlite3
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import API_FOOTBALL_DB


# ===========================================================================
# PREDICTION MARKET → ODDS COLUMN MAPPING
# Maps (market_name, predicted_outcome) → ODDS_* column name
# ===========================================================================

PRED_TO_ODDS_COL: Dict[tuple, str] = {
    # 1X2 Match Winner
    ('1X2', 'H'): 'ODDS_1X2_H',
    ('1X2', 'D'): 'ODDS_1X2_D',
    ('1X2', 'A'): 'ODDS_1X2_A',
    # Both Teams To Score
    ('BTTS', 'Y'): 'ODDS_BTTS_Y',
    ('BTTS', 'N'): 'ODDS_BTTS_N',
    # Over/Under lines
    ('OU_0_5', 'O'): 'ODDS_OU_0_5_O',
    ('OU_0_5', 'U'): 'ODDS_OU_0_5_U',
    ('OU_1_5', 'O'): 'ODDS_OU_1_5_O',
    ('OU_1_5', 'U'): 'ODDS_OU_1_5_U',
    ('OU_2_5', 'O'): 'ODDS_OU_2_5_O',
    ('OU_2_5', 'U'): 'ODDS_OU_2_5_U',
    ('OU_3_5', 'O'): 'ODDS_OU_3_5_O',
    ('OU_3_5', 'U'): 'ODDS_OU_3_5_U',
    ('OU_4_5', 'O'): 'ODDS_OU_4_5_O',
    ('OU_4_5', 'U'): 'ODDS_OU_4_5_U',
    ('OU_5_5', 'O'): 'ODDS_OU_5_5_O',
    ('OU_5_5', 'U'): 'ODDS_OU_5_5_U',
    # Double Chance (only "yes" side has odds)
    ('DC_1X', 'Y'): 'ODDS_DC_1X',
    ('DC_12', 'Y'): 'ODDS_DC_12',
    ('DC_X2', 'Y'): 'ODDS_DC_X2',
}


def load_odds_for_fixtures(fixture_ids: List[int]) -> pd.DataFrame:
    """
    Load bookmaker odds from fixture_odds table and pivot to wide format.

    Args:
        fixture_ids: List of API-Football fixture IDs

    Returns:
        DataFrame with columns: fixture_id, ODDS_1X2_H, ODDS_OU_2_5_O, ...
        One row per fixture_id.
    """
    if not fixture_ids:
        return pd.DataFrame(columns=['fixture_id'])

    conn = sqlite3.connect(API_FOOTBALL_DB)

    # Single query for all fixture_ids (chunked if > 500 to avoid SQLite limits)
    all_rows = []
    chunk_size = 500
    for i in range(0, len(fixture_ids), chunk_size):
        chunk = fixture_ids[i:i + chunk_size]
        placeholders = ','.join(['?'] * len(chunk))
        query = f"""
            SELECT fixture_id, market, selection, odd
            FROM fixture_odds
            WHERE fixture_id IN ({placeholders})
        """
        chunk_df = pd.read_sql_query(query, conn, params=chunk)
        all_rows.append(chunk_df)

    conn.close()

    if not all_rows:
        return pd.DataFrame(columns=['fixture_id'])

    df = pd.concat(all_rows, ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=['fixture_id'])

    # Reconstruct ODDS_* column names from market + selection
    df['odds_col'] = 'ODDS_' + df['market'] + '_' + df['selection']

    # Pivot: one row per fixture_id, one column per odds_col
    pivoted = df.pivot_table(
        index='fixture_id',
        columns='odds_col',
        values='odd',
        aggfunc='first'
    ).reset_index()

    pivoted.columns.name = None
    return pivoted


def merge_odds_into_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge bookmaker odds into a DataFrame that has a fixture_id column.

    If fixture_id is missing or no odds found, returns df unchanged.

    Args:
        df: DataFrame with fixture_id column

    Returns:
        DataFrame with ODDS_* columns merged in
    """
    if 'fixture_id' not in df.columns:
        return df

    fixture_ids = df['fixture_id'].dropna().astype(int).tolist()
    if not fixture_ids:
        return df

    odds_df = load_odds_for_fixtures(fixture_ids)
    if odds_df.empty:
        return df

    merged = df.merge(odds_df, on='fixture_id', how='left')

    odds_cols = [c for c in odds_df.columns if c.startswith('ODDS_')]
    if odds_cols:
        with_odds = merged[odds_cols[0]].notna().sum()
        print(f"[ODDS] Merged: {with_odds}/{len(df)} fixtures have bookmaker odds ({len(odds_cols)} markets)")

    return merged


def get_odds_for_prediction(row, market_name: str, predicted_outcome: str) -> Optional[float]:
    """
    Look up the bookmaker odds for a specific prediction.

    Args:
        row: Series/dict-like row from a DataFrame with ODDS_* columns
        market_name: Market identifier (e.g. '1X2', 'BTTS', 'OU_2_5')
        predicted_outcome: Predicted outcome (e.g. 'H', 'Y', 'O')

    Returns:
        Decimal odds (float) or None if not available
    """
    odds_col = PRED_TO_ODDS_COL.get((market_name, predicted_outcome))
    if odds_col is None:
        return None

    try:
        val = row[odds_col] if odds_col in row.index else None
    except (KeyError, AttributeError):
        val = row.get(odds_col) if isinstance(row, dict) else None

    # NaN check: NaN != NaN
    if val is not None and val == val:
        return float(val)
    return None


def get_odds_coverage(df: pd.DataFrame) -> float:
    """Return fraction of rows that have at least one ODDS_* column with data."""
    odds_cols = [c for c in df.columns if c.startswith('ODDS_')]
    if not odds_cols:
        return 0.0
    has_any = df[odds_cols].notna().any(axis=1).sum()
    return has_any / len(df) if len(df) > 0 else 0.0
