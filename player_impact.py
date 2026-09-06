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


def resolve_team_ids(db_path=None) -> Dict[str, int]:
    """team name -> team_id, from the fixtures table.

    Fixture files that come from a manual download or the CSV fallback have
    no team ids, so availability would otherwise be unresolvable for them.
    Uses the most recent appearance of each name.
    """
    import sqlite3
    from config import API_FOOTBALL_DB
    try:
        conn = sqlite3.connect(db_path or API_FOOTBALL_DB)
        df = pd.read_sql_query("""
            SELECT home_team AS name, home_team_id AS team_id, date FROM fixtures
            WHERE home_team_id IS NOT NULL
            UNION ALL
            SELECT away_team AS name, away_team_id AS team_id, date FROM fixtures
            WHERE away_team_id IS NOT NULL
        """, conn)
        conn.close()
    except Exception:
        return {}
    if df.empty:
        return {}
    df = df.sort_values("date").drop_duplicates(subset=["name"], keep="last")
    return {_norm_name(n): int(t) for n, t in zip(df["name"], df["team_id"])}


# API-Football reports availability as a `player.type`. "Missing Fixture" is a
# confirmed absence; "Questionable"/"Doubtful" players often start. Treating
# both as definitely-out overstates the adjustment, so doubts are half-weighted.
_AVAILABILITY_WEIGHT = {
    "missing fixture": 1.0,
    "out": 1.0,
    "suspended": 1.0,
    "questionable": 0.5,
    "doubtful": 0.5,
}
DEFAULT_AVAILABILITY_WEIGHT = 1.0


def injury_weight(player_type) -> float:
    """How certain is this absence? 1.0 = definitely out, 0.5 = doubtful."""
    if not player_type:
        return DEFAULT_AVAILABILITY_WEIGHT
    return _AVAILABILITY_WEIGHT.get(str(player_type).strip().lower(),
                                    DEFAULT_AVAILABILITY_WEIGHT)


def load_confirmed_lineup(fixture_id: int, team_id: int, db_path=None) -> Optional[set]:
    """player_ids in the confirmed starting XI, or None if not available.

    A confirmed lineup is strictly stronger evidence than an injury list:
    injuries are advisory (doubtful players start, unlisted players are
    rested/dropped/suspended), whereas the XI is who is actually on the pitch.
    Usually published ~1h before kickoff, so it is absent for a weekly run and
    present for a late refresh — callers fall back to injuries when None.
    """
    import sqlite3
    from config import API_FOOTBALL_DB
    try:
        conn = sqlite3.connect(db_path or API_FOOTBALL_DB)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lineups'")
        if not cur.fetchone():
            conn.close()
            return None
        rows = cur.execute(
            "SELECT player_id FROM lineups WHERE fixture_id=? AND team_id=? AND is_starter=1",
            (int(fixture_id), int(team_id))).fetchall()
        conn.close()
    except Exception:
        return None
    ids = {int(r[0]) for r in rows if r[0] is not None}
    return ids or None


def missing_from_lineup(shares: pd.DataFrame, starter_ids: set) -> float:
    """Share of recent goal involvement belonging to players NOT starting."""
    if shares is None or shares.empty or not starter_ids:
        return 0.0
    absent = ~shares["player_id"].isin(starter_ids)
    return float(shares.loc[absent, "share"].sum())


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
    out_records: Optional[List[dict]] = None,
    starter_ids: Optional[set] = None,
    window_matches: int = DEFAULT_WINDOW_MATCHES,
    key_threshold: float = 0.20,
) -> dict:
    """Everything the pipeline needs about one team's attacking availability.

    `out_records` is the natural shape from the injuries feed —
    [{player_id, player_name, player_type}, ...] — and lets doubtful players
    be half-weighted. `out_player_ids`/`out_player_names` remain for callers
    that only have the flat lists.

    missing_share  : 0-1 fraction of recent goal involvement unavailable,
                     weighted by how certain each absence is
    top_out        : the unavailable players carrying that share
    top_scorers    : the team's leading contributors (available or not)
    key_player_out : True when one absentee carries >= key_threshold of output
    """
    shares = team_attack_shares(stats, team_id, before_date, window_matches)
    if shares.empty:
        return {"missing_share": 0.0, "top_out": [], "top_scorers": [],
                "key_player_out": False, "has_data": False, "source": "none"}

    # A confirmed starting XI supersedes the injury list entirely: it already
    # accounts for rotation, suspension and late fitness calls that injuries
    # never capture, and it needs no name matching.
    if starter_ids:
        absent = ~shares["player_id"].isin(starter_ids)
        out_rows = shares[absent].sort_values("share", ascending=False)
        missing = float(out_rows["share"].sum())
        return {
            "missing_share": round(min(missing, 1.0), 4),
            "top_out": top_scorers(out_rows, 3),
            "top_scorers": top_scorers(shares, 3),
            "key_player_out": bool((out_rows["share"] >= key_threshold).any())
                              if not out_rows.empty else False,
            "has_data": True,
            "source": "lineup",
        }

    # Normalise every input shape into records carrying an availability weight
    records: List[dict] = list(out_records or [])
    if out_player_ids:
        known = {r.get("player_id") for r in records}
        records += [{"player_id": i} for i in out_player_ids
                    if pd.notna(i) and i not in known]
    if out_player_names:
        known_n = {_norm_name(r.get("player_name", "")) for r in records}
        records += [{"player_name": n} for n in out_player_names
                    if n and _norm_name(n) not in known_n]

    if not records:
        return {"missing_share": 0.0, "top_out": [],
                "top_scorers": top_scorers(shares, 3),
                "key_player_out": False, "has_data": True, "source": "none"}

    norm = shares["player_name"].map(_norm_name)
    sur = shares["player_name"].map(surname_key)

    # weight per squad row = strongest weight among the absences matching it
    weights = pd.Series(0.0, index=shares.index)
    for rec in records:
        w = injury_weight(rec.get("player_type"))
        m = pd.Series(False, index=shares.index)
        pid = rec.get("player_id")
        if pid is not None and pd.notna(pid):
            try:
                m |= shares["player_id"] == int(pid)
            except (TypeError, ValueError):
                pass
        nm = rec.get("player_name")
        if nm:
            m |= norm == _norm_name(nm)
            sk = surname_key(nm)
            if sk:
                m |= sur == sk
        weights = weights.where(~m, weights.combine(pd.Series(w, index=shares.index), max))

    effective = shares["share"] * weights
    missing = float(effective.sum())

    out_rows = (shares.assign(share=effective)[weights > 0]
                .sort_values("share", ascending=False))
    return {
        "missing_share": round(float(min(missing, 1.0)), 4),
        "top_out": top_scorers(out_rows, 3),
        "top_scorers": top_scorers(shares, 3),
        "key_player_out": bool((out_rows["share"] >= key_threshold).any())
                          if not out_rows.empty else False,
        "has_data": True,
        "source": "injuries",
    }
