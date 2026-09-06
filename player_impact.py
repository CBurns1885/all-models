# player_impact.py
"""
Who actually scores this team's goals, and how much of that is missing?

`player_fixture_stats` holds per-player, per-fixture goals / assists / shots /
minutes / rating. Until now nothing read it — ~250k rows and several days of
API quota sat unused. This module turns it into the signal that matters for
pricing a fixture: *how much of a team's recent attacking output belongs to
players who will not be on the pitch*.

Core idea
---------
Weight each player by their share of the team's recent goal involvement:

    involvement = goals + assist_weight * assists          (assist_weight=0.5)
    share_p     = involvement_p / sum(involvement over squad)

Then for a fixture with a known set of unavailable players (injuries, or the
complement of a confirmed lineup):

    missing_share = sum(share_p for p in unavailable)

`missing_share` is a far better injury signal than a raw headcount: a squad
player and a 20-goal striker both count 1 in a headcount, but contribute
~0.00 and ~0.35 here. It is bounded [0, 1] and directly interpretable as
"this fraction of the team's goal threat is absent".

Availability is only known for UPCOMING fixtures (the injuries endpoint is
per-fixture and historical injury data was never stored — see CLAUDE.md), so
this is applied as a prediction-time adjustment and a picks-page signal, not
yet as a trained feature. Once `injuries` accumulates a season of history it
can graduate into features.py.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ASSIST_WEIGHT = 0.5
DEFAULT_WINDOW_MATCHES = 10
MIN_TEAM_INVOLVEMENT = 3.0   # below this the shares are too noisy to trust


def load_player_match_stats(db_path=None) -> pd.DataFrame:
    """Per-player, per-fixture attacking output joined to fixture date/team.

    Returns columns: fixture_id, Date, team_id, player_id, player_name,
    position, minutes_played, goals, assists, shots_total.
    Empty DataFrame if the table is missing or empty.
    """
    import sqlite3
    from config import API_FOOTBALL_DB
    path = db_path or API_FOOTBALL_DB
    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='player_fixture_stats'")
        if not cur.fetchone():
            conn.close()
            return pd.DataFrame()
        df = pd.read_sql_query("""
            SELECT p.fixture_id, f.date AS Date, p.team_id, p.player_id,
                   p.player_name, p.position, p.minutes_played,
                   p.goals, p.assists, p.shots_total
            FROM player_fixture_stats p
            JOIN fixtures f ON f.fixture_id = p.fixture_id
            WHERE f.status = 'FT'
        """, conn)
        conn.close()
    except Exception as e:
        print(f"[PLAYER] Could not load player stats: {e}")
        return pd.DataFrame()

    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ("goals", "assists", "shots_total", "minutes_played"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.dropna(subset=["Date"])


def team_attack_shares(
    stats: pd.DataFrame,
    team_id: int,
    before_date,
    window_matches: int = DEFAULT_WINDOW_MATCHES,
    assist_weight: float = ASSIST_WEIGHT,
) -> pd.DataFrame:
    """Goal-involvement share per player over the team's last N matches
    before `before_date`.

    Returns a frame sorted by share desc with columns:
    player_id, player_name, goals, assists, involvement, share, matches.
    Empty if there is not enough signal to be meaningful.
    """
    if stats.empty:
        return pd.DataFrame()

    before_date = pd.Timestamp(before_date)
    tdf = stats[(stats["team_id"] == team_id) & (stats["Date"] < before_date)]
    if tdf.empty:
        return pd.DataFrame()

    recent_fixtures = (tdf[["fixture_id", "Date"]].drop_duplicates()
                       .sort_values("Date", ascending=False)
                       .head(window_matches)["fixture_id"])
    tdf = tdf[tdf["fixture_id"].isin(set(recent_fixtures))]
    if tdf.empty:
        return pd.DataFrame()

    agg = (tdf.groupby(["player_id", "player_name"], dropna=False)
           .agg(goals=("goals", "sum"),
                assists=("assists", "sum"),
                minutes=("minutes_played", "sum"),
                matches=("fixture_id", "nunique"))
           .reset_index())
    agg["involvement"] = agg["goals"] + assist_weight * agg["assists"]

    total = float(agg["involvement"].sum())
    if total < MIN_TEAM_INVOLVEMENT:
        return pd.DataFrame()

    agg["share"] = agg["involvement"] / total
    return agg.sort_values("share", ascending=False).reset_index(drop=True)


def missing_attack_share(
    shares: pd.DataFrame,
    out_player_ids: Optional[Iterable] = None,
    out_player_names: Optional[Iterable] = None,
) -> float:
    """Fraction of recent goal involvement belonging to unavailable players.

    Matches on player_id when available (ids are stable, names are not) and
    falls back to a normalised name match.
    """
    if shares is None or shares.empty:
        return 0.0

    mask = pd.Series(False, index=shares.index)
    if out_player_ids:
        ids = {int(i) for i in out_player_ids if pd.notna(i)}
        if ids:
            mask |= shares["player_id"].isin(ids)
    if out_player_names:
        wanted = {_norm_name(n) for n in out_player_names if n}
        if wanted:
            mask |= shares["player_name"].map(_norm_name).isin(wanted)
    return float(shares.loc[mask, "share"].sum())


def top_scorers(shares: pd.DataFrame, n: int = 3) -> List[dict]:
    """Top n contributors, for display."""
    if shares is None or shares.empty:
        return []
    return [
        {"player_id": r.player_id, "name": r.player_name,
         "goals": int(r.goals), "assists": int(r.assists),
         "share": round(float(r.share), 4)}
        for r in shares.head(n).itertuples()
    ]


# Letters that carry no combining mark and so survive NFKD unchanged.
# Without these, 'M. Odegaard' from the injuries feed never matches
# 'Martin Ødegaard' in the player stats — and the Nordic leagues (NOR, SWE,
# DEN) are full of them.
_CHAR_FOLD = str.maketrans({
    "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "å": "a", "Å": "a",
    "ß": "ss", "đ": "d", "Đ": "d", "ð": "d", "Ð": "d",
    "ł": "l", "Ł": "l", "þ": "th", "Þ": "th", "ı": "i", "œ": "oe", "Œ": "oe",
})


def _norm_name(name) -> str:
    """Normalise for matching: casefold, fold accents, strip punctuation.

    API-Football is inconsistent between endpoints ('B. Saka' in injuries,
    'Bukayo Saka' in player stats), so callers also key on the surname alone.
    """
    import unicodedata
    s = str(name).translate(_CHAR_FOLD)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    return " ".join(s.lower().split())


def surname_key(name) -> str:
    parts = _norm_name(name).split()
    return parts[-1] if parts else ""


def match_missing_by_name(shares: pd.DataFrame, out_names: Iterable) -> float:
    """Name-only matching that also tries surnames — for the common case
    where the injuries feed abbreviates first names."""
    if shares is None or shares.empty:
        return 0.0
    out_names = [n for n in out_names if n]
    if not out_names:
        return 0.0

    full = {_norm_name(n) for n in out_names}
    surnames = {surname_key(n) for n in out_names if surname_key(n)}

    norm = shares["player_name"].map(_norm_name)
    sur = shares["player_name"].map(surname_key)
    mask = norm.isin(full) | sur.isin(surnames)
    return float(shares.loc[mask, "share"].sum())


def availability_report(
    stats: pd.DataFrame,
    team_id: int,
    before_date,
    out_player_ids: Optional[Iterable] = None,
    out_player_names: Optional[Iterable] = None,
    window_matches: int = DEFAULT_WINDOW_MATCHES,
) -> dict:
    """Everything the pipeline needs about one team's attacking availability.

    missing_share  : 0-1 fraction of recent goal involvement unavailable
    top_out        : the unavailable players that carry that share
    top_scorers    : the team's leading contributors (available or not)
    key_player_out : True when a single absentee carries >=20% of output
    """
    shares = team_attack_shares(stats, team_id, before_date, window_matches)
    if shares.empty:
        return {"missing_share": 0.0, "top_out": [], "top_scorers": [],
                "key_player_out": False, "has_data": False}

    by_id = missing_attack_share(shares, out_player_ids=out_player_ids)
    by_name = match_missing_by_name(shares, out_player_names or [])
    missing = max(by_id, by_name)

    out_mask = pd.Series(False, index=shares.index)
    if out_player_ids:
        ids = {int(i) for i in out_player_ids if pd.notna(i)}
        out_mask |= shares["player_id"].isin(ids)
    if out_player_names:
        full = {_norm_name(n) for n in out_player_names if n}
        surs = {surname_key(n) for n in out_player_names if surname_key(n)}
        out_mask |= (shares["player_name"].map(_norm_name).isin(full)
                     | shares["player_name"].map(surname_key).isin(surs))

    out_rows = shares[out_mask].sort_values("share", ascending=False)
    return {
        "missing_share": round(float(min(missing, 1.0)), 4),
        "top_out": top_scorers(out_rows, 3),
        "top_scorers": top_scorers(shares, 3),
        "key_player_out": bool((out_rows["share"] >= 0.20).any()) if not out_rows.empty else False,
        "has_data": True,
    }
