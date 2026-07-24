"""
Backfill bookmaker odds for all historical FT fixtures.
Fetches from API-Football (Bet365/Pinnacle priority) and stores in fixture_odds table.

API cost: ~1 call per fixture. Run over multiple days if needed (7500/day limit).
Resumable: skips fixtures that already have odds in the DB.
"""
import time
import sys
from api_client import fetch_odds_for_fixture, _connect, _make_request

def check_quota() -> int:
    data = _make_request("status", {})
    if data and "response" in data:
        r = data["response"]
        used = r.get("requests", {}).get("current", 0)
        limit = r.get("requests", {}).get("limit_day", 7500)
        remaining = limit - used
        print(f"[API] Quota: {used}/{limit} used, {remaining} remaining today")
        return remaining
    return 9999

def get_fixtures_needing_odds(limit: int = None):
    """Return fixture_ids that don't yet have odds stored."""
    conn = _connect()
    cursor = conn.cursor()
    query = """
        SELECT f.fixture_id, f.date, f.league_code
        FROM fixtures f
        LEFT JOIN (
            SELECT DISTINCT fixture_id FROM fixture_odds
        ) fo ON f.fixture_id = fo.fixture_id
        WHERE f.status = 'FT'
        AND fo.fixture_id IS NULL
        ORDER BY f.date DESC
    """
    if limit:
        query += f" LIMIT {limit}"
    rows = cursor.fetchall() if False else conn.execute(query).fetchall()
    conn.close()
    return rows

def main():
    print("=" * 60)
    print("HISTORICAL ODDS BACKFILL")
    print("=" * 60)

    remaining = check_quota()
    # Leave 200 calls buffer for other operations
    safe_limit = max(0, remaining - 200)

    fixtures = get_fixtures_needing_odds(limit=safe_limit)
    total = len(fixtures)

    if total == 0:
        print("[OK] All fixtures already have odds — nothing to fetch")
        return

    print(f"[INFO] Fixtures needing odds: {total} (fetching up to {safe_limit} today)")

    ok = 0
    empty = 0
    fail = 0

    for i, (fixture_id, date, league) in enumerate(fixtures, 1):
        try:
            odds = fetch_odds_for_fixture(fixture_id)
            if odds:
                ok += 1
            else:
                empty += 1
        except Exception as e:
            fail += 1
            print(f"  [ERR] fixture {fixture_id}: {e}")

        if i % 100 == 0:
            pct = i / total * 100
            remaining_fixtures = total - i
            print(f"  [{i}/{total}] {pct:.1f}% | got={ok} empty={empty} fail={fail} | ~{remaining_fixtures} left")

        time.sleep(0.15)  # ~6 calls/sec

    print(f"\n[DONE] Fetched odds: {ok} fixtures with odds, {empty} no odds available, {fail} errors")
    print(f"       Run again tomorrow to fetch more (7500/day limit)")

if __name__ == "__main__":
    main()
