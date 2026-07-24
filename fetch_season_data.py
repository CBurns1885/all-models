"""
Fetch all FT fixtures + match_stats for season 2025 (2025/26).
Resumes safely — skips fixtures that already have stats.
"""
import sys
import time
import sqlite3
from pathlib import Path

# ensure we can import project modules
sys.path.insert(0, str(Path(__file__).parent))

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
    RATE_LIMIT_DELAY,
)

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
    """Fetch match_stats for all FT fixtures that don't have stats yet."""
    print("\n=== PHASE 2: Fetch match_stats for fixtures without stats ===")

    conn = sqlite3.connect(API_FOOTBALL_DB, timeout=60); conn.execute("PRAGMA journal_mode=WAL")
    rows = conn.execute("""
        SELECT f.fixture_id, f.date, f.league_code
        FROM fixtures f
        LEFT JOIN match_stats ms ON f.fixture_id = ms.fixture_id
        WHERE f.status = 'FT' AND ms.fixture_id IS NULL
        ORDER BY f.date ASC
    """).fetchall()
    conn.close()

    total_needed = len(rows)
    print(f"  Fixtures needing stats: {total_needed}")
    if total_needed == 0:
        print("[OK] All fixtures already have match_stats.")
        return 0

    # Estimate: ~0.5s/call
    est_min = total_needed * (RATE_LIMIT_DELAY + 0.05) / 60
    print(f"  Estimated time: ~{est_min:.0f} minutes")

    success = 0
    failed = 0
    for i, (fixture_id, date, league_code) in enumerate(rows):
        ok = fetch_fixture_statistics(fixture_id)
        if ok:
            success += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0 or i == total_needed - 1:
            pct = (i + 1) / total_needed * 100
            remaining = total_needed - (i + 1)
            eta = remaining * (RATE_LIMIT_DELAY + 0.05) / 60
            print(f"  [{i+1}/{total_needed}] {pct:.1f}% | ok={success} fail={failed} | ETA ~{eta:.0f}m")

    print(f"\n[OK] Stats fetch complete: {success} OK, {failed} empty/failed")
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
