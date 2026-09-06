"""
Betfair price snapshot — READ ONLY.

Fetches the current best back/lay price for each pick in best_bets.csv and
writes them to betfair_odds.csv alongside it, so tools/best_bets.py can size
stakes (see staking.py) against real exchange prices.

This module deliberately contains NO order placement code and imports nothing
that can place an order. It defaults to the DELAYED application key, which
cannot place bets at all. Live execution (betfair/betfair_placer.py,
betfair/betfair_ltd.py) remains switched off pending a separate decision — see
CLAUDE.md; nothing here changes that.

Usage:
  py betfair/betfair_odds.py                 # delayed prices for latest best_bets.csv
  py betfair/betfair_odds.py --live-key      # live (undelayed) prices, still read-only
  py betfair/betfair_odds.py --bets path.csv --out odds.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# betfair/ -> repo root
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_kickoff(date_str: str, time_val=None):
    """Kickoff datetime from a date plus optional HH:MM (UTC)."""
    try:
        base = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    hour, minute = 15, 0
    if time_val is not None and str(time_val) not in ("", "nan", "NaT", "None"):
        try:
            parts = str(time_val).strip().split(":")
            hour, minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            pass
    return base.replace(hour=hour, minute=minute, tzinfo=timezone.utc)


def find_latest_bets_file() -> Path | None:
    dated = sorted(d for d in OUTPUTS.iterdir()
                   if d.is_dir() and d.name[:4].isdigit())
    for d in reversed(dated):
        f = d / "best_bets.csv"
        if f.exists():
            return f
    return None


def fetch_odds(bets_file: Path, out_file: Path, live_key: bool = False,
               horizon_hours: int = 96) -> pd.DataFrame:
    """Fetch best back/lay prices for each (fixture, market, selection) pick."""
    from betfair_auth import get_client
    from betfair_markets import find_market, find_runner_selection
    import betfairlightweight.filters as filters

    bets = pd.read_csv(bets_file)
    print(f"Loaded {len(bets)} picks from {bets_file}")

    client = get_client(live=live_key)
    key_kind = "LIVE (read-only)" if live_key else "DELAYED"
    print(f"Connected with {key_kind} key — this script never places orders\n")

    now = datetime.now(timezone.utc)
    horizon = now + timedelta(hours=horizon_hours)

    # One market lookup per (fixture, market) even if several picks share it
    market_cache: dict[tuple, tuple] = {}
    rows = []
    skipped = 0

    for _, bet in bets.iterrows():
        market_name = str(bet.get("Market", ""))
        outcome = str(bet.get("Bet", ""))
        home = str(bet.get("Home", ""))
        away = str(bet.get("Away", ""))
        kickoff = _parse_kickoff(bet.get("Date"), bet.get("Time"))

        if kickoff is None or kickoff < now or kickoff > horizon:
            skipped += 1
            continue

        cache_key = (home, away, market_name, kickoff.date())
        if cache_key not in market_cache:
            try:
                market_cache[cache_key] = find_market(client, home, away, kickoff, market_name)
            except Exception as e:
                logger.warning("find_market failed for %s v %s (%s): %s",
                               home, away, market_name, e)
                market_cache[cache_key] = (None, None)
        market_id, runner_map = market_cache[cache_key]

        if market_id is None or not runner_map:
            skipped += 1
            continue

        selection_id = find_runner_selection(runner_map, outcome)
        if selection_id is None:
            logger.debug("No runner for %s in %s: %s", outcome, market_name, list(runner_map))
            skipped += 1
            continue

        back_price = lay_price = None
        try:
            books = client.betting.list_market_book(
                market_ids=[market_id],
                price_projection=filters.price_projection(price_data=["EX_BEST_OFFERS"]),
            )
            for runner in (books[0].runners if books else []):
                if runner.selection_id == selection_id:
                    if runner.ex:
                        if runner.ex.available_to_back:
                            back_price = runner.ex.available_to_back[0].price
                        if runner.ex.available_to_lay:
                            lay_price = runner.ex.available_to_lay[0].price
                    break
        except Exception as e:
            logger.warning("list_market_book failed for %s: %s", market_id, e)

        if back_price is None:
            skipped += 1
            continue

        rows.append({
            "Date": str(bet.get("Date", ""))[:10],
            "Home": home,
            "Away": away,
            "Market": market_name,
            "Bet": outcome,
            "BackPrice": back_price,
            "LayPrice": lay_price,
            "MarketId": market_id,
            "SelectionId": selection_id,
            "FetchedAt": now.isoformat(timespec="seconds"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[WARN] No prices retrieved ({skipped} picks skipped — out of window, "
              f"market not found, or no price available)")
        return df

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"[OK] Wrote {len(df)} prices -> {out_file}  ({skipped} skipped)")
    print(f"\nRerun best_bets to attach stakes:  py tools/best_bets.py --bank <amount>")
    return df


def main():
    ap = argparse.ArgumentParser(description="Fetch Betfair prices for current picks (read-only)")
    ap.add_argument("--bets", type=str, default=None,
                    help="best_bets.csv to price (default: latest dated outputs folder)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output CSV (default: betfair_odds.csv beside the bets file)")
    ap.add_argument("--live-key", action="store_true",
                    help="Use the live app key for undelayed prices (still read-only)")
    ap.add_argument("--horizon-hours", type=int, default=96,
                    help="Only price fixtures kicking off within this many hours")
    args = ap.parse_args()

    bets_file = Path(args.bets) if args.bets else find_latest_bets_file()
    if bets_file is None or not bets_file.exists():
        print("[ERROR] No best_bets.csv found — run tools/best_bets.py first")
        sys.exit(1)

    out_file = Path(args.out) if args.out else bets_file.parent / "betfair_odds.csv"
    fetch_odds(bets_file, out_file, live_key=args.live_key,
               horizon_hours=args.horizon_hours)


if __name__ == "__main__":
    main()
