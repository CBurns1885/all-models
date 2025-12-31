# download_football_data.py
from __future__ import annotations
from pathlib import Path
import requests
from typing import Iterable, Tuple
from config import DATA_DIR, log_header
from progress_utils import Timer, heartbeat

# League codes supported by football-data.co.uk (extend as needed)
DEFAULT_LEAGUES = [
    "E0", "E1", "E2", "E3", "EC",      # England
    "D1", "D2",                         # Germany
    "SP1", "SP2",                       # Spain
    "I1", "I2",                         # Italy
    "F1", "F2",                         # France
    "N1",                               # Netherlands
    "B1",                               # Belgium
    "P1",                               # Portugal
    "G1",                               # Greece
    "SC0", "SC1", "SC2", "SC3",        # Scotland
    "T1",                               # Turkey
]

def _season_code(year_start: int) -> str:
    # e.g. 2025 -> "2223"
    ys = year_start % 100
    ye = (year_start + 1) % 100
    return f"{ys:02d}{ye:02d}"

def _url_for(league: str, season_code: str) -> str:
    # standard mmz4281 path
    return f"https://www.football-data.co.uk/mmz4281/{season_code}/{league}.csv"

def download(leagues: Iterable[str], seasons: Iterable[int]) -> None:
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with Timer("Downloading Football-Data CSVs"):
        for lg in leagues:
            for y in seasons:
                sc = _season_code(y)
                url = _url_for(lg, sc)
                out = raw_dir / f"{lg}_{sc}.csv"
                try:
                    heartbeat(f"GET {url}")
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200 and (len(r.content) > 150):  # basic sanity check
                        out.write_bytes(r.content)
                        print(f"  ↳ saved {out}")
                    else:
                        print(f"  ↳ skip {lg} {sc}: not available ({r.status_code})")
                except Exception as e:
                    print(f"  ↳ error {lg} {sc}: {e}")

if __name__ == "__main__":
    import argparse, datetime as dt
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="*", default=DEFAULT_LEAGUES)
    ap.add_argument("--start_season", type=int, default=2025, help="Season start year, e.g., 2017 for 2017-18")
    ap.add_argument("--end_season", type=int, default=dt.datetime.now().year, help="Inclusive start-year of season, e.g., 2025 for 2025-26")
    args = ap.parse_args()
    seasons = list(range(args.start_season, args.end_season + 1))
    log_header(f"Download leagues={args.leagues} seasons={seasons}")
    download(args.leagues, seasons)
