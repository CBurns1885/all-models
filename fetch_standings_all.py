"""
Fetch end-of-season standings for all leagues, seasons 2022-2025.
Stores in the standings table of football_api.db.
~120-160 API calls total (well within daily limit).
"""
import sys
import time
from config import API_LEAGUE_MAP
from api_client import fetch_standings, _make_request

# Cup competitions don't have standings — skip to avoid wasted API calls
CUP_CODES = {"EC", "FAC", "DFB", "CDR", "CIT", "CDF", "KNVB", "BEC", "TCP", "SFC", "TFC", "UCL", "UEL", "UECL"}

LEAGUE_ONLY_CODES = [code for code in API_LEAGUE_MAP if code not in CUP_CODES]
SEASONS = [2022, 2023, 2024, 2025]

def check_api_quota():
    data = _make_request("status", {})
    if data and "response" in data:
        r = data["response"]
        used = r.get("requests", {}).get("current", 0)
        limit = r.get("requests", {}).get("limit_day", 7500)
        remaining = limit - used
        print(f"[API] Quota: {used}/{limit} used, {remaining} remaining today")
        return remaining
    return 9999

def main():
    print("=" * 60)
    print("STANDINGS FETCH — all leagues, seasons 2022-2025")
    print("=" * 60)

    remaining = check_api_quota()
    total_needed = len(LEAGUE_ONLY_CODES) * len(SEASONS)
    print(f"[INFO] Leagues with standings: {len(LEAGUE_ONLY_CODES)}")
    print(f"[INFO] Seasons: {SEASONS}")
    print(f"[INFO] Max API calls needed: {total_needed}")

    if remaining < total_needed:
        print(f"[WARN] Only {remaining} API calls remaining — may not complete today")

    total_fetched = 0
    total_calls = 0

    for season in SEASONS:
        print(f"\n--- Season {season} ---")
        for code in sorted(LEAGUE_ONLY_CODES):
            n = fetch_standings(code, season)
            total_calls += 1
            if n > 0:
                total_fetched += n
                print(f"  {code} {season}: {n} teams")
            time.sleep(0.15)  # ~6-7 calls/sec, well within rate limit

    print(f"\n[DONE] Fetched {total_fetched} team-season standings rows in {total_calls} API calls")

if __name__ == "__main__":
    main()
