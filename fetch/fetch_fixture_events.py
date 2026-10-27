"""
fetch_fixture_events.py — Populate fixture_events from /fixtures/events endpoint.

One API call per fixture returns all in-match events: goals, cards, substitutions,
VAR decisions. Enables timing features (early/late goals, card timing, comeback
tendency) and player-level event features in features.py.

Fetches fixtures that have match_stats (confirmed finished + useful for training).
Ordered by league tier then date DESC — top leagues / recent matches first.
Resumable: skips fixture_ids already in fixture_events or confirmed empty.

Usage:
    py fetch/fetch_fixture_events.py                    # fill everything missing
    py fetch/fetch_fixture_events.py --season 2025      # only 2025/26 season
    py fetch/fetch_fixture_events.py --limit 1000       # stop after N fixtures
    py fetch/fetch_fixture_events.py --dry-run          # count without fetching
"""

import sys
import time
import sqlite3
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass

from config import API_FOOTBALL_DB
from api_client import _make_request, _get_league_type, RATE_LIMIT_DELAY

API_QUOTA_BUFFER = 50
_TIER_RANK = {"elite": 0, "high": 1, "medium": 2}


def _check_quota():
    data = _make_request("status", {})
    if data and "response" in data:
        r = data["response"]
        used = r.get("requests", {}).get("current", 0)
        limit = r.get("requests", {}).get("limit_day", 7500)
        remaining = limit - used
        return used, limit, remaining
    return 0, 7500, 7500


def _ensure_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fixture_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER NOT NULL,
            elapsed INTEGER,
            extra INTEGER,
            team_id INTEGER,
            team_name TEXT,
            player_id INTEGER,
            player_name TEXT,
            assist_id INTEGER,
            assist_name TEXT,
            type TEXT,
            detail TEXT,
            comments TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_fixture_events_fixture
        ON fixture_events(fixture_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_fixture_events_type
        ON fixture_events(fixture_id, type)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fixture_events_unavailable (
            fixture_id INTEGER PRIMARY KEY,
            checked_at TEXT
        )
    """)
    conn.commit()


_QUOTA_EXHAUSTED = object()  # sentinel distinct from None and []


def _fetch_events(fixture_id):
    """
    Returns list of event dicts, [] if confirmed no events, None on transient
    failure, _QUOTA_EXHAUSTED if the daily limit has been reached.
    """
    data = _make_request("fixtures/events", {"fixture": fixture_id})
    if not data or "response" not in data:
        return None

    # Detect quota-exceeded errors — these are NOT permanent no-data responses
    errors = data.get("errors", {})
    if errors:
        err_str = str(errors).lower()
        if "limit" in err_str or "quota" in err_str or "upgrade" in err_str:
            return _QUOTA_EXHAUSTED

    rows = []
    for ev in data["response"]:
        t = ev.get("time", {})
        team = ev.get("team", {})
        player = ev.get("player", {})
        assist = ev.get("assist", {})
        rows.append({
            "fixture_id":   fixture_id,
            "elapsed":      t.get("elapsed"),
            "extra":        t.get("extra"),
            "team_id":      team.get("id"),
            "team_name":    team.get("name"),
            "player_id":    player.get("id"),
            "player_name":  player.get("name"),
            "assist_id":    assist.get("id"),
            "assist_name":  assist.get("name"),
            "type":         ev.get("type"),
            "detail":       ev.get("detail"),
            "comments":     ev.get("comments"),
        })
    return rows


def _insert_rows(conn, rows):
    conn.executemany("""
        INSERT INTO fixture_events
        (fixture_id, elapsed, extra, team_id, team_name,
         player_id, player_name, assist_id, assist_name,
         type, detail, comments)
        VALUES
        (:fixture_id, :elapsed, :extra, :team_id, :team_name,
         :player_id, :player_name, :assist_id, :assist_name,
         :type, :detail, :comments)
    """, rows)
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Fetch fixture events per fixture")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("FIXTURE EVENTS FETCH — /fixtures/events")
    print("=" * 60)

    conn = sqlite3.connect(API_FOOTBALL_DB, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_tables(conn)

    season_clause = ""
    params = []
    if args.season is not None:
        season_clause = "AND f.season = ?"
        params.append(args.season)

    query = f"""
        SELECT DISTINCT f.fixture_id, f.date, f.league_code, f.season, f.league_id
        FROM fixtures f
        JOIN match_stats ms ON f.fixture_id = ms.fixture_id
        WHERE f.status = 'FT'
          AND f.fixture_id NOT IN (
              SELECT DISTINCT fixture_id FROM fixture_events
          )
          AND f.fixture_id NOT IN (
              SELECT fixture_id FROM fixture_events_unavailable
          )
          {season_clause}
    """
    rows = conn.execute(query, params).fetchall()

    rows.sort(key=lambda r: r[1], reverse=True)  # newest first, work backwards

    total_missing = len(rows)
    if args.limit:
        rows = rows[:args.limit]

    print(f"[INFO] Fixtures missing events: {total_missing}")
    if args.season:
        print(f"[INFO] Season filter: {args.season}")
    if args.limit:
        print(f"[INFO] Limit: {args.limit} this run")
    print()

    if args.dry_run:
        print("[DRY-RUN] Exiting without fetching.")
        conn.close()
        return

    if not rows:
        print("[OK] Nothing to fetch — fixture_events is up to date.")
        conn.close()
        return

    used, limit, remaining = _check_quota()
    print(f"[API] Quota: {used}/{limit} used, {remaining} remaining today")
    print(f"[INFO] This run needs up to: {len(rows)} calls")
    if remaining <= API_QUOTA_BUFFER:
        print(f"[WARN] Only {remaining} requests left — below buffer. Run again tomorrow.")
        conn.close()
        return

    available = remaining - API_QUOTA_BUFFER
    if len(rows) > available:
        print(f"[WARN] Capping at {available} fixtures (quota limit minus buffer)")
        rows = rows[:available]

    print(f"[INFO] Will fetch {len(rows)} fixtures now.")
    print()

    fetched = 0
    skipped = 0
    failed = 0
    total_events = 0

    for i, (fixture_id, date, league, season, league_id) in enumerate(rows):
        event_rows = _fetch_events(fixture_id)

        if event_rows is _QUOTA_EXHAUSTED:
            print(f"\n[WARN] Daily quota exhausted at fixture {i+1}/{len(rows)} — stopping early.")
            break
        elif event_rows is None:
            failed += 1
        elif event_rows:
            _insert_rows(conn, event_rows)
            total_events += len(event_rows)
            fetched += 1
        else:
            skipped += 1
            conn.execute(
                "INSERT OR IGNORE INTO fixture_events_unavailable (fixture_id, checked_at) "
                "VALUES (?, datetime('now'))",
                (fixture_id,),
            )
            conn.commit()

        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            pct = round((i + 1) / len(rows) * 100, 1)
            print(f"  [{i+1}/{len(rows)}] {pct}%  fetched={fetched}  events={total_events}  "
                  f"no_data={skipped}  failed={failed}  last={league} {date}")

        time.sleep(RATE_LIMIT_DELAY)

    conn.close()

    print()
    print("=" * 60)
    print(f"[DONE] Fixtures fetched: {fetched}  Events inserted: {total_events}  No-data: {skipped}")
    if fetched < total_missing:
        print(f"[INFO] {total_missing - fetched} fixtures still missing — run again tomorrow")
    else:
        print("[OK] fixture_events fully populated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
