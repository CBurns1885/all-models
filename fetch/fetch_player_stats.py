"""
fetch_player_stats.py — Populate player_fixture_stats from /fixtures/players endpoint.

Fetches one API call per fixture (returns all players for both teams).
Resumable: skips fixture_ids already in player_fixture_stats.
Ordered by date DESC so most recent seasons are filled first.

Priority filter: only fetches fixtures that also have match_stats
(i.e. confirmed finished and useful for training).

Usage:
    py fetch_player_stats.py                    # fill everything missing
    py fetch_player_stats.py --season 2025      # only 2025/26 season
    py fetch_player_stats.py --limit 1000       # stop after N fixtures
    py fetch_player_stats.py --dry-run          # count without fetching
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
from api_client import _make_request, _get_headers, RATE_LIMIT_DELAY

API_QUOTA_BUFFER = 50  # stop when this many requests remain


def _check_quota():
    data = _make_request("status", {})
    if data and "response" in data:
        r = data["response"]
        used = r.get("requests", {}).get("current", 0)
        limit = r.get("requests", {}).get("limit_day", 7500)
        remaining = limit - used
        return used, limit, remaining
    return 0, 7500, 7500


def _fetch_fixture_players(fixture_id):
    """Call /fixtures/players?fixture={id}. Returns list of player-stat dicts."""
    data = _make_request("fixtures/players", {"fixture": fixture_id})
    if not data or "response" not in data:
        return []

    rows = []
    for team_block in data["response"]:
        team_id = team_block.get("team", {}).get("id")
        for entry in team_block.get("players", []):
            player = entry.get("player", {})
            stats_list = entry.get("statistics", [{}])
            s = stats_list[0] if stats_list else {}

            games    = s.get("games", {})
            goals    = s.get("goals", {})
            shots    = s.get("shots", {})
            passes   = s.get("passes", {})
            tackles  = s.get("tackles", {})
            duels    = s.get("duels", {})
            dribbles = s.get("dribbles", {})
            fouls    = s.get("fouls", {})
            cards    = s.get("cards", {})

            def _int(v):
                try:
                    return int(v) if v is not None else None
                except (TypeError, ValueError):
                    return None

            def _float(v):
                try:
                    return float(v) if v is not None else None
                except (TypeError, ValueError):
                    return None

            rows.append({
                "fixture_id":        fixture_id,
                "team_id":           team_id,
                "player_id":         player.get("id"),
                "player_name":       player.get("name"),
                "position":          games.get("position"),
                "minutes_played":    _int(games.get("minutes")),
                "rating":            _float(games.get("rating")),
                "goals":             _int(goals.get("total")),
                "assists":           _int(goals.get("assists")),
                "shots_total":       _int(shots.get("total")),
                "shots_on_target":   _int(shots.get("on")),
                "passes_total":      _int(passes.get("total")),
                "passes_accurate":   _int(passes.get("accuracy")),
                "passes_key":        _int(passes.get("key")),
                "tackles":           _int(tackles.get("total")),
                "interceptions":     _int(tackles.get("interceptions")),
                "duels_total":       _int(duels.get("total")),
                "duels_won":         _int(duels.get("won")),
                "dribbles_attempts": _int(dribbles.get("attempts")),
                "dribbles_success":  _int(dribbles.get("success")),
                "yellow_cards":      _int(cards.get("yellow")),
                "red_cards":         _int(cards.get("red")),
                "fouls_committed":   _int(fouls.get("committed")),
                "fouls_drawn":       _int(fouls.get("drawn")),
            })
    return rows


def _insert_rows(conn, rows):
    conn.executemany("""
        INSERT OR IGNORE INTO player_fixture_stats
        (fixture_id, team_id, player_id, player_name, position, minutes_played,
         rating, goals, assists, shots_total, shots_on_target,
         passes_total, passes_accurate, passes_key,
         tackles, interceptions, duels_total, duels_won,
         dribbles_attempts, dribbles_success,
         yellow_cards, red_cards, fouls_committed, fouls_drawn)
        VALUES
        (:fixture_id, :team_id, :player_id, :player_name, :position, :minutes_played,
         :rating, :goals, :assists, :shots_total, :shots_on_target,
         :passes_total, :passes_accurate, :passes_key,
         :tackles, :interceptions, :duels_total, :duels_won,
         :dribbles_attempts, :dribbles_success,
         :yellow_cards, :red_cards, :fouls_committed, :fouls_drawn)
    """, rows)
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Fetch player stats per fixture")
    parser.add_argument("--season", type=int, default=None,
                        help="Limit to one season (e.g. 2025 = 2025/26)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max fixtures to fetch this run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count missing fixtures without fetching")
    args = parser.parse_args()

    print("=" * 60)
    print("PLAYER STATS FETCH — /fixtures/players")
    print("=" * 60)

    conn = sqlite3.connect(API_FOOTBALL_DB, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")

    # Find fixtures that have match_stats but no player_fixture_stats
    season_clause = ""
    params = []
    if args.season is not None:
        season_clause = "AND f.season = ?"
        params.append(args.season)

    query = f"""
        SELECT DISTINCT f.fixture_id, f.date, f.league_code, f.season
        FROM fixtures f
        JOIN match_stats ms ON f.fixture_id = ms.fixture_id
        WHERE f.status = 'FT'
          AND f.fixture_id NOT IN (
              SELECT DISTINCT fixture_id FROM player_fixture_stats
          )
          {season_clause}
        ORDER BY f.date DESC
    """
    rows = conn.execute(query, params).fetchall()

    total_missing = len(rows)
    if args.limit:
        rows = rows[:args.limit]

    print(f"[INFO] Fixtures missing player stats: {total_missing}")
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
        print("[OK] Nothing to fetch — player_fixture_stats is up to date.")
        conn.close()
        return

    # Check quota
    used, limit, remaining = _check_quota()
    print(f"[API] Quota: {used}/{limit} used, {remaining} remaining today")
    print(f"[INFO] This run needs: {len(rows)} calls")
    if remaining <= API_QUOTA_BUFFER:
        print(f"[WARN] Only {remaining} requests left — below buffer ({API_QUOTA_BUFFER}). Run again tomorrow.")
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
    total_players = 0

    for i, (fixture_id, date, league, season) in enumerate(rows):
        player_rows = _fetch_fixture_players(fixture_id)

        if player_rows:
            _insert_rows(conn, player_rows)
            total_players += len(player_rows)
            fetched += 1
        else:
            skipped += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            pct = round((i + 1) / len(rows) * 100, 1)
            print(f"  [{i+1}/{len(rows)}] {pct}%  fetched={fetched}  players={total_players}  "
                  f"no_data={skipped}  last={league} {date}")

        time.sleep(RATE_LIMIT_DELAY)

    conn.close()

    print()
    print("=" * 60)
    print(f"[DONE] Fixtures fetched: {fetched}  Players inserted: {total_players}  No-data: {skipped}")
    if fetched < total_missing:
        remaining_after = total_missing - fetched
        print(f"[INFO] {remaining_after} fixtures still missing — run again tomorrow")
    else:
        print("[OK] player_fixture_stats fully populated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
