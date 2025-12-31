# data_ingest.py - Fixed version
from __future__ import annotations
import io
import time
from typing import List, Dict, Optional
import pandas as pd
import requests
from pathlib import Path
import numpy as np

from config import (
    RAW_DIR, PROCESSED_DIR, FOOTBALL_DATA_CSV_BASE, SEASONS, LEAGUE_CODES,
    HISTORICAL_PARQUET, FOOTBALL_DATA_ORG_TOKEN, FOOTBALL_DATA_ORG_BASE, log_header
)



def _safe_request(url: str, headers: Optional[Dict[str, str]] = None, max_retries: int = 3, timeout: int = 30) -> requests.Response:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")

def _download_historic_csv(season: str, league: str) -> pd.DataFrame:
    url = FOOTBALL_DATA_CSV_BASE.format(season=season, league=league)
    resp = _safe_request(url)
    raw = resp.content
    # Handle encoding issues
    df = pd.read_csv(io.BytesIO(raw), encoding='latin1')
    df["Season"] = season
    df["League"] = league
    return df

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required columns exist
    for col in ["B365H","B365D","B365A","PSCH","PSCD","PSCA"]:
        if col not in df.columns:
            df[col] = np.nan
    
    # Fix date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    
    # Convert odds columns to numeric
    odds_cols = ["B365H","B365D","B365A","PSCH","PSCD","PSCA"]
    for col in odds_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Keep only required columns
    keep_cols = ["League","Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
                "B365H","B365D","B365A","PSCH","PSCD","PSCA","Season"]
    available_cols = [c for c in keep_cols if c in df.columns]
    return df[available_cols]

def build_historical_results(seasons: List[str] = SEASONS, leagues: List[str] = LEAGUE_CODES, force: bool = False) -> Path:
    """
    Build historical results - now uses API-Football database (shared with dc_laptop)
    Falls back to CSV download if API-Football DB not available
    """
    out_path = HISTORICAL_PARQUET
    if out_path.exists() and not force:
        log_header(f"Historical parquet already exists at {out_path}. Skipping download.")
        return out_path

    # Try API-Football database first (NEW - shared with dc_laptop)
    try:
        from api_football_adapter import check_api_football_db, build_historical_from_api

        if check_api_football_db():
            log_header("Using API-Football database (shared with dc_laptop)")
            # Convert string seasons (e.g., "2324") to start year integers (e.g., 2023)
            # Season "2324" means 2023-2024 season, so we take first 2 digits and add 2000
            season_ints = []
            for s in seasons:
                if len(s) == 4:  # Format like "2324"
                    year_start = 2000 + int(s[:2])  # "23" -> 2023
                    season_ints.append(year_start)
                else:
                    season_ints.append(int(s))  # Already a year
            return build_historical_from_api(seasons=season_ints, leagues=leagues, force=force)
        else:
            log_header("API-Football database not found, falling back to CSV download...")
    except ImportError:
        log_header("API-Football adapter not available, using CSV download...")
    except Exception as e:
        log_header(f"API-Football adapter failed ({e}), falling back to CSV download...")

    # Fallback to old CSV download method
    log_header("Downloading from football-data.co.uk (legacy method)...")
    frames = []
    for s in seasons:
        for lg in leagues:
            try:
                print(f"Fetching {lg} {s} ...")
                df = _download_historic_csv(s, lg)
                df = _standardize_columns(df)
                df = df.dropna(subset=["HomeTeam","AwayTeam","Date"])
                if len(df) > 0:  # Only add if we got data
                    frames.append(df)
            except Exception as e:
                print(f"  [WARN] {lg} {s}: {e}")
                continue

    if not frames:
        raise RuntimeError("No historical data could be downloaded. Check network or league/season lists.")

    hist = pd.concat(frames, ignore_index=True)
    
    # Create additional columns safely
    hist["HomeGoals"] = pd.to_numeric(hist["FTHG"], errors='coerce').astype("Int64")
    hist["AwayGoals"] = pd.to_numeric(hist["FTAG"], errors='coerce').astype("Int64")
    
    # Create OU25 column
    total_goals = hist["HomeGoals"].fillna(0) + hist["AwayGoals"].fillna(0)
    hist["OU25"] = np.where(total_goals > 2, "O", "U").astype(str)
    
    # Sort and save
    hist = hist.sort_values(["Date","League"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(out_path, index=False)
    
    log_header(f"Wrote historical results to {out_path} with {len(hist):,} rows.")
    return out_path

def fetch_fixtures(date_from: str, date_to: str, competitions: Optional[List[str]] = None) -> pd.DataFrame:
    if not FOOTBALL_DATA_ORG_TOKEN:
        raise RuntimeError("FOOTBALL_DATA_ORG_TOKEN is not set. Provide token or supply fixtures CSV.")
    
    params = []
    if competitions:
        params.append(f"competitions={','.join(competitions)}")
    params.append(f"dateFrom={date_from}")
    params.append(f"dateTo={date_to}")
    
    url = f"{FOOTBALL_DATA_ORG_BASE}/matches?{'&'.join(params)}"
    headers = {"X-Auth-Token": FOOTBALL_DATA_ORG_TOKEN}
    resp = _safe_request(url, headers=headers)
    js = resp.json()
    
    matches = js.get("matches", [])
    rows = []
    for m in matches:
        rows.append({
            "utcDate": m.get("utcDate"),
            "competition": m.get("competition", {}).get("code") or m.get("competition", {}).get("name"),
            "HomeTeam": m.get("homeTeam", {}).get("name"),
            "AwayTeam": m.get("awayTeam", {}).get("name"),
            "status": m.get("status"),
            "id": m.get("id"),
        })
    
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    
    df["Date"] = pd.to_datetime(df["utcDate"]).dt.tz_convert("Europe/London").dt.date
    return df[["Date","competition","HomeTeam","AwayTeam","status","id"]]

if __name__ == "__main__":
    build_historical_results()