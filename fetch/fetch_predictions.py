"""
fetch_predictions.py — Fetch API-Football's own predictions per fixture.

Stores their 1X2 percentages, OU2.5 call, and comparison stats (form, attack,
defence, Poisson, H2H). Used two ways:
  1. Benchmark: compare their accuracy vs ours on historical fixtures
  2. Feature: add APIF_* columns to training data (disagreement signal)

Backfill mode: fills historical FT fixtures with match_stats (same gate as
player_stats / fixture_events). ~29K calls, ~4 days at 7,500/day.

Usage:
    py fetch/fetch_predictions.py                    # fill all missing
    py fetch/fetch_predictions.py --season 2025      # one season only
    py fetch/fetch_predictions.py --limit 1000       # stop after N
    py fetch/fetch_predictions.py --dry-run          # count without fetching
    py fetch/fetch_predictions.py --upcoming         # fetch for NS fixtures only
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
from api_client import _make_request, RATE_LIMIT_DELAY

API_QUOTA_BUFFER = 50


def _ensure_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fixture_predictions (
            fixture_id INTEGER PRIMARY KEY,
            apif_home_pct REAL,
            apif_draw_pct REAL,
            apif_away_pct REAL,
            apif_winner TEXT,
            apif_win_or_draw INTEGER,
            apif_under_over TEXT,
            apif_advice TEXT,
            cmp_form_home REAL,
            cmp_form_away REAL,
            cmp_att_home REAL,
            cmp_att_away REAL,
            cmp_def_home REAL,
            cmp_def_away REAL,
            cmp_poisson_home REAL,
            cmp_poisson_away REAL,
            cmp_h2h_home REAL,
            cmp_h2h_away REAL,
            cmp_goals_home REAL,
            cmp_goals_away REAL,
            cmp_total_home REAL,
            cmp_total_away REAL,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fixture_predictions_unavailable (
            fixture_id INTEGER PRIMARY KEY,
            checked_at TEXT
        )
    """)
    conn.commit()


def _pct(val):
    """Parse '45%' → 45.0, None → None."""
    if val is None:
        return None
    try:
        return float(str(val).replace("%", "").strip())
    except (ValueError, TypeError):
        return None


_QUOTA_EXHAUSTED = object()


def _fetch_prediction(fixture_id):
    """
    Returns parsed dict, {} if confirmed no data, None on transient failure,
    _QUOTA_EXHAUSTED if the daily limit has been reached.
    """
    data = _make_request("predictions", {"fixture": fixture_id})
    if not data or "response" not in data:
        return None

    errors = data.get("errors", {})
    if errors:
        err_str = str(errors).lower()
        if "limit" in err_str or "quota" in err_str or "upgrade" in err_str:
            return _QUOTA_EXHAUSTED

    if not data["response"]:
        return {}

    resp = data["response"][0]
    pred = resp.get("predictions", {})
    cmp = resp.get("comparison", {})

    def _cmp(key, side):
        return _pct(cmp.get(key, {}).get(side))

    return {
        "fixture_id":       fixture_id,
        "apif_home_pct":    _pct(pred.get("percent", {}).get("home")),
        "apif_draw_pct":    _pct(pred.get("percent", {}).get("draw")),
        "apif_away_pct":    _pct(pred.get("percent", {}).get("away")),
        "apif_winner":      (pred.get("winner") or {}).get("name"),
        "apif_win_or_draw": 1 if pred.get("win_or_draw") else 0,
        "apif_under_over":  pred.get("under_over"),
        "apif_advice":      pred.get("advice"),
        "cmp_form_home":    _cmp("form", "home"),
        "cmp_form_away":    _cmp("form", "away"),
        "cmp_att_home":     _cmp("att", "home"),
        "cmp_att_away":     _cmp("att", "away"),
        "cmp_def_home":     _cmp("def", "home"),
        "cmp_def_away":     _cmp("def", "away"),
        "cmp_poisson_home": _cmp("poisson_distribution", "home"),
        "cmp_poisson_away": _cmp("poisson_distribution", "away"),
        "cmp_h2h_home":     _cmp("h2h", "home"),
        "cmp_h2h_away":     _cmp("h2h", "away"),
        "cmp_goals_home":   _cmp("goals", "home"),
        "cmp_goals_away":   _cmp("goals", "away"),
        "cmp_total_home":   _cmp("total", "home"),
        "cmp_total_away":   _cmp("total", "away"),
    }


def _insert_row(conn, row):
    conn.execute("""
        INSERT OR REPLACE INTO fixture_predictions
        (fixture_id, apif_home_pct, apif_draw_pct, apif_away_pct,
         apif_winner, apif_win_or_draw, apif_under_over, apif_advice,
         cmp_form_home, cmp_form_away, cmp_att_home, cmp_att_away,
         cmp_def_home, cmp_def_away, cmp_poisson_home, cmp_poisson_away,
         cmp_h2h_home, cmp_h2h_away, cmp_goals_home, cmp_goals_away,
         cmp_total_home, cmp_total_away, fetched_at)
        VALUES
        (:fixture_id, :apif_home_pct, :apif_draw_pct, :apif_away_pct,
         :apif_winner, :apif_win_or_draw, :apif_under_over, :apif_advice,
         :cmp_form_home, :cmp_form_away, :cmp_att_home, :cmp_att_away,
         :cmp_def_home, :cmp_def_away, :cmp_poisson_home, :cmp_poisson_away,
         :cmp_h2h_home, :cmp_h2h_away, :cmp_goals_home, :cmp_goals_away,
         :cmp_total_home, :cmp_total_away, datetime('now'))
    """, row)
    conn.commit()


def _build_fixture_list(conn, season=None, upcoming=False, from_csv=None):
    params = []
    if upcoming or from_csv:
        # Resolve CSV path
        if from_csv:
            csv_path = Path(from_csv)
        else:
            # Find most recent upcoming_fixtures.csv in outputs/
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from config import OUTPUT_DIR
            candidates = sorted(OUTPUT_DIR.glob("*/upcoming_fixtures.csv"), reverse=True)
            csv_path = candidates[0] if candidates else None

        if csv_path and csv_path.exists():
            import pandas as pd
            df_csv = pd.read_csv(csv_path)
            if "fixture_id" in df_csv.columns:
                ids = df_csv["fixture_id"].dropna().astype(int).tolist()
                placeholders = ",".join("?" * len(ids))
                # Return as list of tuples matching expected shape
                rows = conn.execute(
                    f"""SELECT DISTINCT f.fixture_id, f.date, f.league_code,
                               COALESCE(f.season, 2025), COALESCE(f.league_id, 0)
                        FROM fixtures f
                        WHERE f.fixture_id IN ({placeholders})
                          AND f.fixture_id NOT IN (SELECT fixture_id FROM fixture_predictions)""",
                    ids
                ).fetchall()
                if not rows:
                    # Fixture IDs from CSV may not be in DB — build minimal tuples
                    already = set(r[0] for r in conn.execute(
                        "SELECT fixture_id FROM fixture_predictions"
                    ).fetchall())
                    rows = [
                        (int(r["fixture_id"]), r.get("Date", ""), r.get("League", ""), 2025, 0)
                        for _, r in df_csv.iterrows()
                        if pd.notna(r.get("fixture_id")) and int(r["fixture_id"]) not in already
                    ]
                return rows
        # Fallback: NS fixtures from DB
        query = """
            SELECT DISTINCT fixture_id, date, league_code,
                   COALESCE(season, 2025), COALESCE(league_id, 0)
            FROM fixtures
            WHERE status = 'NS'
              AND fixture_id NOT IN (SELECT fixture_id FROM fixture_predictions)
            ORDER BY date ASC
        """
        return conn.execute(query).fetchall()
    else:
        season_clause = ""
        if season is not None:
            season_clause = "AND f.season = ?"
            params.append(season)
        query = f"""
            SELECT DISTINCT f.fixture_id, f.date, f.league_code, f.season, f.league_id
            FROM fixtures f
            JOIN match_stats ms ON f.fixture_id = ms.fixture_id
            WHERE f.status = 'FT'
              AND f.fixture_id NOT IN (
                  SELECT fixture_id FROM fixture_predictions
              )
              AND f.fixture_id NOT IN (
                  SELECT fixture_id FROM fixture_predictions_unavailable
              )
              {season_clause}
            ORDER BY f.date DESC
        """
    return conn.execute(query, params).fetchall()


def _check_quota():
    data = _make_request("status", {})
    if data and "response" in data:
        r = data["response"]
        used = r.get("requests", {}).get("current", 0)
        limit = r.get("requests", {}).get("limit_day", 7500)
        return used, limit, limit - used
    return 0, 7500, 7500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upcoming", action="store_true",
                        help="Fetch predictions for upcoming fixtures (reads from latest upcoming_fixtures.csv)")
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Path to upcoming_fixtures.csv to pull fixture_ids from")
    args = parser.parse_args()

    print("=" * 60)
    print("PREDICTIONS FETCH — /predictions")
    print("=" * 60)

    conn = sqlite3.connect(API_FOOTBALL_DB, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_tables(conn)

    rows = _build_fixture_list(conn, season=args.season, upcoming=args.upcoming, from_csv=args.from_csv)
    total_missing = len(rows)
    if args.limit:
        rows = rows[:args.limit]

    mode = "upcoming NS" if args.upcoming else "historical FT"
    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Fixtures missing predictions: {total_missing}")
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
        print("[OK] Nothing to fetch — fixture_predictions is up to date.")
        conn.close()
        return

    used, limit, remaining = _check_quota()
    print(f"[API] Quota: {used}/{limit} used, {remaining} remaining today")
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

    for i, (fixture_id, date, league, season, league_id) in enumerate(rows):
        result = _fetch_prediction(fixture_id)

        if result is _QUOTA_EXHAUSTED:
            print(f"\n[WARN] Daily quota exhausted at fixture {i+1}/{len(rows)} — stopping early.")
            break
        elif result is None:
            failed += 1
        elif result:
            _insert_row(conn, result)
            fetched += 1
        else:
            skipped += 1
            conn.execute(
                "INSERT OR IGNORE INTO fixture_predictions_unavailable "
                "(fixture_id, checked_at) VALUES (?, datetime('now'))",
                (fixture_id,),
            )
            conn.commit()

        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            pct = round((i + 1) / len(rows) * 100, 1)
            print(f"  [{i+1}/{len(rows)}] {pct}%  fetched={fetched}  "
                  f"no_data={skipped}  failed={failed}  last={league} {date}")

        time.sleep(RATE_LIMIT_DELAY)

    conn.close()

    print()
    print("=" * 60)
    print(f"[DONE] Predictions fetched: {fetched}  No-data: {skipped}  Failed: {failed}")
    remaining_after = total_missing - fetched
    if remaining_after > 0:
        print(f"[INFO] {remaining_after} fixtures still missing — run again tomorrow")
    else:
        print("[OK] fixture_predictions fully populated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
