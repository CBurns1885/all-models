"""
Fetch all FT fixtures + match_stats for season 2025 (2025/26).
Resumes safely — skips fixtures that already have stats.
"""
import sys
import time
import sqlite3
from pathlib import Path

# ensure we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env before any project imports so API_FOOTBALL_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass

from config import LEAGUE_CODES, API_FOOTBALL_DB, API_LEAGUE_MAP
from api_client import (
    test_api_connection,
    fetch_fixtures_for_league,
    fetch_fixture_statistics,
    _init_database,
    _get_headers,
    _make_request,
    _get_league_type,
    RATE_LIMIT_DELAY,
)

_TIER_RANK = {"elite": 0, "high": 1, "medium": 2}

SEASON = 2025


def fetch_all_fixtures():
    """Fetch FT fixtures for all leagues for SEASON."""
    print(f"\n=== PHASE 1: Fetch FT fixtures (season {SEASON}) ===")
    _init_database()
    total = 0
    for i, lc in enumerate(LEAGUE_CODES):
        if lc not in API_LEAGUE_MAP:
            continue
        count = fetch_fixtures_for_league(lc, SEASON, status='FT')
        total += count
        if count > 0:
            print(f"  [{i+1}/{len(LEAGUE_CODES)}] {lc}: {count} matches")
        else:
            print(f"  [{i+1}/{len(LEAGUE_CODES)}] {lc}: 0 (no data or not started)")
    print(f"\n[OK] Total FT fixtures fetched/updated: {total}")
    return total


def fetch_missing_stats():
    """Fetch match_stats for all FT fixtures that don't have stats yet.

    Fixtures where the API confirms no stats exist (empty response, not a
    request failure) are recorded in match_stats_unavailable so they aren't
    re-fetched (and don't burn quota) on every future run.
    """
    print("\n=== PHASE 2: Fetch match_stats for fixtures without stats ===")

    conn = sqlite3.connect(API_FOOTBALL_DB, timeout=60); conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS match_stats_unavailable (
            fixture_id INTEGER PRIMARY KEY,
            checked_at TEXT
        )
    """)
    conn.commit()
    rows = conn.execute("""
        SELECT f.fixture_id, f.date, f.league_code, f.league_id
        FROM fixtures f
        LEFT JOIN match_stats ms ON f.fixture_id = ms.fixture_id
        LEFT JOIN match_stats_unavailable msu ON f.fixture_id = msu.fixture_id
        WHERE f.status = 'FT' AND ms.fixture_id IS NULL AND msu.fixture_id IS NULL
    """).fetchall()

    # Prioritize elite/high-tier leagues first, then most recent within each tier —
    # top-league recent form matters most for the model and is most likely to have
    # API stats coverage; minor/older fixtures often have none at all.
    rows.sort(key=lambda r: r[1], reverse=True)  # newest first
    rows.sort(key=lambda r: _TIER_RANK.get(_get_league_type(r[3]), 2))  # stable: tier wins ties

    total_needed = len(rows)
    print(f"  Fixtures needing stats: {total_needed}")
    if total_needed == 0:
        conn.close()
        print("[OK] All fixtures already have match_stats (or confirmed unavailable).")
        return 0

    # Estimate: ~0.5s/call
    est_min = total_needed * (RATE_LIMIT_DELAY + 0.05) / 60
    print(f"  Estimated time: ~{est_min:.0f} minutes")

    success = 0
    empty = 0
    failed = 0
    for i, (fixture_id, date, league_code, league_id) in enumerate(rows):
        result = fetch_fixture_statistics(fixture_id)
        if result > 0:
            success += 1
        elif result == 0:
            empty += 1
            conn.execute(
                "INSERT OR IGNORE INTO match_stats_unavailable (fixture_id, checked_at) VALUES (?, datetime('now'))",
                (fixture_id,),
            )
            conn.commit()
        else:
            failed += 1

        if (i + 1) % 100 == 0 or i == total_needed - 1:
            pct = (i + 1) / total_needed * 100
            remaining = total_needed - (i + 1)
            eta = remaining * (RATE_LIMIT_DELAY + 0.05) / 60
            print(f"  [{i+1}/{total_needed}] {pct:.1f}% | ok={success} empty={empty} fail={failed} | ETA ~{eta:.0f}m")

    conn.close()
    print(f"\n[OK] Stats fetch complete: {success} OK, {empty} confirmed no-data, {failed} request failures")
    return success


def print_db_summary():
    conn = sqlite3.connect(API_FOOTBALL_DB, timeout=60); conn.execute("PRAGMA journal_mode=WAL")
    ft = conn.execute("SELECT COUNT(*) FROM fixtures WHERE status='FT'").fetchone()[0]
    ft_with_stats = conn.execute("""
        SELECT COUNT(DISTINCT f.fixture_id) FROM fixtures f
        JOIN match_stats ms ON f.fixture_id = ms.fixture_id
        WHERE f.status='FT'
    """).fetchone()[0]
    max_date = conn.execute("SELECT MAX(date) FROM fixtures WHERE status='FT'").fetchone()[0]
    conn.close()
    print(f"\n=== DB Summary ===")
    print(f"  FT fixtures: {ft}")
    print(f"  FT with match_stats: {ft_with_stats}")
    print(f"  Latest FT date: {max_date}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"SEASON DATA FETCH — season {SEASON}")
    print("=" * 60)

    print("\nTesting API connection...")
    if not test_api_connection():
        print("[ERROR] API connection failed. Check key.")
        sys.exit(1)

    print_db_summary()

    fetch_all_fixtures()
    fetch_missing_stats()

    print_db_summary()
    print("\nDone. Next steps:")
    print("  1. py -c \"from features import build_features; build_features(force=True)\"")
    print("  2. py run_weekly.py --speed full --mode 4 --non-interactive")
    print("  3. py -c \"from blending import learn_blend_weights; learn_blend_weights()\"")
    print("  4. del outputs\\tuning_preds_cache.parquet && py auto_tune.py")
