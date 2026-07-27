"""
Lay the Draw (LTD) — model-enhanced automated football trading strategy.

Entry logic:
  1. Load 1X2 predictions from best_bets.csv
  2. For each match: check Betfair draw price (target range 3.0-4.0)
  3. Compare our model's draw probability vs Betfair implied draw probability
  4. Lay the draw only when we have genuine edge (model says draw less likely than market)

In-play management:
  - Poll match_odds market every 30s
  - Goal detected when draw price rises > GOAL_SPIKE_PCT (e.g. 25%)
  - On goal: execute hedge back bet to lock profit on all outcomes
  - Stop-loss: if 0-0 at STOP_LOSS_MINUTE, close at small loss
  - Full-time: position auto-settled by Betfair

Run:
  py betfair_ltd.py --dry-run --bank 500
  py betfair_ltd.py --live --bank 500 --max-stake 30
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# This file lives in betfair/ — outputs/ is at the repo root, one level up.
ROOT    = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
LTD_LOG = OUTPUTS / "ltd_trades.csv"

# Allow sibling imports (betfair_auth, betfair_markets) regardless of CWD
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- Strategy parameters ---
DRAW_ODDS_MIN       = 2.8    # min Betfair draw price to enter
DRAW_ODDS_MAX       = 4.2    # max Betfair draw price to enter
MIN_MODEL_EDGE      = 0.05   # our draw prob must be this much lower than Betfair implied
GOAL_SPIKE_PCT      = 0.20   # draw price rise >= 20% → treat as goal → hedge
STOP_LOSS_MINUTE    = 30     # close position if still open at this minute (0-0)
POLL_INTERVAL_SECS  = 30     # how often to check price in-play
COMMISSION          = 0.05


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

@dataclass
class LTDPosition:
    match_key:       str           # "Home v Away"
    market_id:       str
    draw_selection:  int           # runner selection_id for The Draw
    lay_price:       float         # price we laid at
    lay_stake:       float         # stake (= potential winnings if draw doesn't happen)
    liability:       float         # max loss if draw does happen = stake * (price - 1)
    kickoff:         datetime
    our_draw_prob:   float
    bf_draw_implied: float
    status:          str = "open"  # open | hedged | stopped | settled
    hedge_stake:     float = 0.0
    hedge_price:     float = 0.0
    locked_profit:   float = 0.0
    dry_run:         bool = True
    bet_id:          Optional[str] = None
    hedge_bet_id:    Optional[str] = None
    notes:           str = ""

    @property
    def edge(self) -> float:
        return self.bf_draw_implied - self.our_draw_prob

    def calc_hedge(self, current_draw_price: float) -> float:
        """
        Back stake needed to lock equal profit on both outcomes.
        hedge_stake = (lay_stake * lay_price) / current_draw_price
        """
        return round((self.lay_stake * self.lay_price) / current_draw_price, 2)

    def locked_profit_at(self, hedge_stake: float, hedge_price: float) -> float:
        """
        Profit locked in regardless of outcome after hedge placed.
        If no draw: win lay_stake, lose hedge_stake
        If draw:    win hedge_stake*(price-1)*(1-comm) - liability
        Returns the minimum of the two (worst-case locked profit).
        """
        no_draw  = (self.lay_stake - hedge_stake) * (1 - COMMISSION)
        draw_win = (hedge_stake * (hedge_price - 1) * (1 - COMMISSION)) - self.liability
        return round(min(no_draw, draw_win), 2)

    def minutes_elapsed(self) -> float:
        now = datetime.now(timezone.utc)
        return (now - self.kickoff).total_seconds() / 60


def _parse_kickoff(date_str: str, time_val=None) -> Optional[datetime]:
    """Parse a kickoff datetime from a date string plus optional HH:MM time.

    Falls back to 15:00 UTC only when no time is available; in-play timing
    (stop-loss minutes, goal-spike detection) depends on this being right,
    so fixture files should always carry a real Time column.
    """
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


# ---------------------------------------------------------------------------
# Entry scan — find qualifying LTD opportunities
# ---------------------------------------------------------------------------

def scan_entries(client, bank: float, max_stake: float, kelly_fraction: float = 0.25) -> list[LTDPosition]:
    """
    Read best_bets.csv 1X2 predictions and find Betfair draw prices.
    Return list of qualifying LTD positions (not yet placed).
    """
    from betfair_markets import find_market, get_best_back_price
    import betfairlightweight.filters as filters

    dated_dirs = sorted(d for d in OUTPUTS.iterdir() if d.is_dir() and d.name[:4].isdigit())
    bets_file = None
    for d in reversed(dated_dirs):
        f = d / "best_bets.csv"
        if f.exists():
            bets_file = f
            break

    if bets_file is None:
        logger.error("No best_bets.csv found")
        return []

    bets = pd.read_csv(bets_file)
    # We need ALL 1X2 predictions to extract draw probabilities
    # best_bets only keeps top-scored bets — we also need the weekly_bets_full for draw probs
    preds_file = None
    for d in reversed(dated_dirs):
        f = d / "weekly_bets_full.csv"
        if f.exists():
            preds_file = f
            break

    positions = []

    if preds_file is None:
        logger.error("No weekly_bets_full.csv found")
        return []

    preds = pd.read_csv(preds_file)
    draw_col = next((c for c in ["BLEND_1X2_D", "P_1X2_D"] if c in preds.columns), None)
    if draw_col is None:
        logger.error("No draw probability column found in weekly_bets_full.csv")
        return []

    now = datetime.now(timezone.utc)

    for _, row in preds.iterrows():
        home = str(row.get("HomeTeam", ""))
        away = str(row.get("AwayTeam", ""))
        date_str = str(row.get("Date", ""))[:10]
        our_draw_prob = float(row.get(draw_col, 0))

        kickoff = _parse_kickoff(date_str, row.get("Time"))
        if kickoff is None:
            continue

        # Only upcoming matches (within next 48h)
        if kickoff < now or kickoff > now + timedelta(hours=48):
            continue

        # Find match odds market on Betfair
        market_id, runner_map = find_market(client, home, away, kickoff, "1X2")
        if market_id is None:
            continue

        draw_sel_id = runner_map.get("The Draw")
        if draw_sel_id is None:
            for k, v in runner_map.items():
                if "draw" in k.lower():
                    draw_sel_id = v
                    break
        if draw_sel_id is None:
            continue

        # Get current lay price for draw
        try:
            books = client.betting.list_market_book(
                market_ids=[market_id],
                price_projection=filters.price_projection(price_data=["EX_BEST_OFFERS"]),
            )
        except Exception as e:
            logger.warning("list_market_book error: %s", e)
            continue

        if not books:
            continue

        draw_lay_price = None
        for runner in books[0].runners:
            if runner.selection_id == draw_sel_id and runner.ex:
                lays = runner.ex.available_to_lay
                if lays:
                    draw_lay_price = lays[0].price
                break

        if draw_lay_price is None:
            continue

        # Apply entry filters
        if not (DRAW_ODDS_MIN <= draw_lay_price <= DRAW_ODDS_MAX):
            logger.debug("Draw price %.2f outside range for %s v %s", draw_lay_price, home, away)
            continue

        bf_draw_implied = 1.0 / draw_lay_price
        edge = bf_draw_implied - our_draw_prob

        if edge < MIN_MODEL_EDGE:
            logger.debug("Insufficient edge %.1f%% for %s v %s (bf=%.0f%% model=%.0f%%)",
                         edge*100, home, away, bf_draw_implied*100, our_draw_prob*100)
            continue

        # Kelly stake on the lay.
        # For a lay bet the amount RISKED is the liability = stake * (price - 1),
        # and winning pays `stake` (i.e. net odds b = 1/(price-1) per unit risked).
        # Kelly therefore gives the fraction of bank to put at risk as LIABILITY;
        # sizing the stake directly would overbet risk by (price-1)x.
        p_win = 1.0 - our_draw_prob
        b = 1.0 / (draw_lay_price - 1.0)
        kelly_f = (p_win * b - our_draw_prob) / b
        if kelly_f <= 0:
            continue
        liability = bank * kelly_f * kelly_fraction
        stake = round(min(liability / (draw_lay_price - 1.0), max_stake), 2)
        if stake < 2.0:
            continue

        liability = round(stake * (draw_lay_price - 1), 2)

        positions.append(LTDPosition(
            match_key       = f"{home} v {away}",
            market_id       = market_id,
            draw_selection  = draw_sel_id,
            lay_price       = draw_lay_price,
            lay_stake       = stake,
            liability       = liability,
            kickoff         = kickoff,
            our_draw_prob   = our_draw_prob,
            bf_draw_implied = bf_draw_implied,
        ))
        logger.info("LTD candidate: %s v %s | draw=%.2f | model=%.0f%% bf=%.0f%% edge=%.0f%% stake=£%.2f liab=£%.2f",
                    home, away, draw_lay_price, our_draw_prob*100, bf_draw_implied*100, edge*100, stake, liability)

    return positions


# ---------------------------------------------------------------------------
# Bet placement helpers
# ---------------------------------------------------------------------------

def place_lay(client, pos: LTDPosition, dry_run: bool) -> bool:
    """Place the initial lay-the-draw bet."""
    if dry_run:
        print(f"  [DRY-RUN] LAY draw  {pos.match_key:<30} @ {pos.lay_price:.2f}  "
              f"stake=£{pos.lay_stake:.2f}  liab=£{pos.liability:.2f}  "
              f"edge={pos.edge:.0%}  kickoff={pos.kickoff.strftime('%H:%M')}")
        pos.status = "open"
        return True

    import betfairlightweight.filters as filters
    try:
        resp = client.betting.place_orders(
            market_id=pos.market_id,
            instructions=[filters.place_instruction(
                selection_id=pos.draw_selection,
                side="LAY",
                order_type="LIMIT",
                limit_order=filters.limit_order(
                    size=pos.lay_stake,
                    price=pos.lay_price,
                    persistence_type="LAPSE",
                ),
            )],
        )
        report = resp.place_instruction_reports[0] if resp.place_instruction_reports else None
        if report and report.status == "SUCCESS":
            pos.bet_id = report.bet_id
            pos.status = "open"
            print(f"  [LIVE] LAY placed  {pos.match_key}  bet_id={pos.bet_id}")
            return True
        else:
            logger.error("LAY failed for %s: %s", pos.match_key, report)
            return False
    except Exception as e:
        logger.error("place_lay error for %s: %s", pos.match_key, e)
        return False


def place_hedge(client, pos: LTDPosition, current_price: float, dry_run: bool):
    """Back the draw to lock profit after goal detected."""
    hedge_stake = pos.calc_hedge(current_price)
    locked = pos.locked_profit_at(hedge_stake, current_price)

    if dry_run:
        print(f"  [DRY-RUN] HEDGE back draw  {pos.match_key:<30} @ {current_price:.2f}  "
              f"stake=£{hedge_stake:.2f}  locked=£{locked:.2f}")
        pos.hedge_stake  = hedge_stake
        pos.hedge_price  = current_price
        pos.locked_profit = locked
        pos.status       = "hedged"
        return

    import betfairlightweight.filters as filters
    try:
        resp = client.betting.place_orders(
            market_id=pos.market_id,
            instructions=[filters.place_instruction(
                selection_id=pos.draw_selection,
                side="BACK",
                order_type="LIMIT",
                limit_order=filters.limit_order(
                    size=hedge_stake,
                    price=current_price,
                    persistence_type="LAPSE",
                ),
            )],
        )
        report = resp.place_instruction_reports[0] if resp.place_instruction_reports else None
        if report and report.status == "SUCCESS":
            pos.hedge_bet_id  = report.bet_id
            pos.hedge_stake   = hedge_stake
            pos.hedge_price   = current_price
            pos.locked_profit = locked
            pos.status        = "hedged"
            print(f"  [LIVE] HEDGE placed  {pos.match_key}  locked=£{locked:.2f}")
    except Exception as e:
        logger.error("place_hedge error for %s: %s", pos.match_key, e)


def close_position(client, pos: LTDPosition, current_price: float, reason: str, dry_run: bool):
    """Close position at stop-loss (back draw at current price to exit)."""
    close_stake = pos.calc_hedge(current_price)
    if dry_run:
        print(f"  [DRY-RUN] CLOSE ({reason})  {pos.match_key:<30} @ {current_price:.2f}  "
              f"stake=£{close_stake:.2f}")
        pos.status = "stopped"
        pos.notes  = reason
        return

    import betfairlightweight.filters as filters
    try:
        resp = client.betting.place_orders(
            market_id=pos.market_id,
            instructions=[filters.place_instruction(
                selection_id=pos.draw_selection,
                side="BACK",
                order_type="LIMIT",
                limit_order=filters.limit_order(
                    size=close_stake,
                    price=current_price,
                    persistence_type="LAPSE",
                ),
            )],
        )
        pos.status = "stopped"
        pos.notes  = reason
        print(f"  [LIVE] CLOSED ({reason})  {pos.match_key}")
    except Exception as e:
        logger.error("close_position error: %s", e)


# ---------------------------------------------------------------------------
# In-play monitoring loop
# ---------------------------------------------------------------------------

def monitor_positions(client, positions: list[LTDPosition], dry_run: bool):
    """
    Poll open positions every POLL_INTERVAL_SECS.
    Detect goal via draw price spike, execute hedge.
    Apply stop-loss at STOP_LOSS_MINUTE.
    """
    import betfairlightweight.filters as filters

    print(f"\nMonitoring {len(positions)} open position(s). Poll every {POLL_INTERVAL_SECS}s.")
    print("Press Ctrl+C to stop.\n")

    while True:
        open_positions = [p for p in positions if p.status == "open"]
        if not open_positions:
            print("All positions closed or hedged.")
            break

        market_ids = list({p.market_id for p in open_positions})

        try:
            books = client.betting.list_market_book(
                market_ids=market_ids,
                price_projection=filters.price_projection(price_data=["EX_BEST_OFFERS"]),
            )
        except Exception as e:
            logger.warning("Poll error: %s", e)
            time.sleep(POLL_INTERVAL_SECS)
            continue

        book_map = {b.market_id: b for b in books}

        for pos in open_positions:
            book = book_map.get(pos.market_id)
            if book is None:
                continue

            # Get current best back price for draw
            current_price = None
            for runner in book.runners:
                if runner.selection_id == pos.draw_selection:
                    if runner.ex:
                        backs = runner.ex.available_to_back
                        if backs:
                            current_price = backs[0].price
                    break

            if current_price is None:
                continue

            mins = pos.minutes_elapsed()

            # Detect goal: draw price spiked significantly
            price_change = (current_price - pos.lay_price) / pos.lay_price
            if price_change >= GOAL_SPIKE_PCT:
                print(f"\n  GOAL DETECTED  {pos.match_key}  "
                      f"draw {pos.lay_price:.2f} -> {current_price:.2f} (+{price_change:.0%})  min={mins:.0f}")
                place_hedge(client, pos, current_price, dry_run)
                log_trade(pos, "hedged")
                continue

            # Stop-loss: still 0-0 at stop-loss minute
            if mins >= STOP_LOSS_MINUTE and pos.status == "open":
                print(f"\n  STOP-LOSS  {pos.match_key}  min={mins:.0f}  draw={current_price:.2f}")
                close_position(client, pos, current_price, f"stop_loss_{STOP_LOSS_MINUTE}min", dry_run)
                log_trade(pos, "stopped")
                continue

            # Status update every poll
            print(f"  MONITORING  {pos.match_key:<30} min={mins:4.0f}  "
                  f"draw={current_price:.2f} (entry={pos.lay_price:.2f})  status={pos.status}")

        time.sleep(POLL_INTERVAL_SECS)


# ---------------------------------------------------------------------------
# Trade logging
# ---------------------------------------------------------------------------

def log_trade(pos: LTDPosition, event: str):
    row = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "event":        event,
        "match":        pos.match_key,
        "market_id":    pos.market_id,
        "lay_price":    pos.lay_price,
        "lay_stake":    pos.lay_stake,
        "liability":    pos.liability,
        "our_draw_pct": round(pos.our_draw_prob * 100, 1),
        "bf_draw_pct":  round(pos.bf_draw_implied * 100, 1),
        "edge_pct":     round(pos.edge * 100, 1),
        "hedge_price":  pos.hedge_price,
        "hedge_stake":  pos.hedge_stake,
        "locked_profit": pos.locked_profit,
        "status":       pos.status,
        "dry_run":      pos.dry_run,
        "notes":        pos.notes,
    }
    write_header = not LTD_LOG.exists()
    with open(LTD_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(bank: float, max_stake: float, dry_run: bool, live_key: bool, kelly: float):
    from betfair_auth import get_client
    client = get_client(live=live_key)

    mode = "DRY-RUN" if dry_run else "LIVE"
    print(f"\n{'='*60}")
    print(f"LAY THE DRAW — {mode}")
    print(f"Bank: £{bank:.2f}  Max stake: £{max_stake:.2f}  Kelly: {kelly:.0%}")
    print(f"Entry filter: draw odds {DRAW_ODDS_MIN}-{DRAW_ODDS_MAX}, min model edge {MIN_MODEL_EDGE:.0%}")
    print(f"Stop-loss at {STOP_LOSS_MINUTE} min, goal spike threshold {GOAL_SPIKE_PCT:.0%}")
    print(f"{'='*60}\n")

    print("Scanning for LTD entries...")
    positions = scan_entries(client, bank, max_stake, kelly)

    if not positions:
        print("No qualifying LTD entries found.")
        return

    print(f"\nFound {len(positions)} qualifying entry/entries:\n")
    for pos in positions:
        print(f"  {pos.match_key:<35} draw={pos.lay_price:.2f}  "
              f"model={pos.our_draw_prob:.0%}  bf={pos.bf_draw_implied:.0%}  "
              f"edge={pos.edge:.0%}  stake=£{pos.lay_stake:.2f}  liab=£{pos.liability:.2f}")

    print()
    placed = []
    for pos in positions:
        pos.dry_run = dry_run
        if place_lay(client, pos, dry_run):
            log_trade(pos, "placed")
            placed.append(pos)

    if not placed:
        print("No positions opened.")
        return

    print(f"\n{len(placed)} position(s) open. Starting in-play monitor...\n")

    # Wait for kickoff of first match
    first_kickoff = min(p.kickoff for p in placed)
    wait_secs = max(0, (first_kickoff - datetime.now(timezone.utc)).total_seconds() - 60)
    if wait_secs > 0:
        print(f"Waiting {wait_secs/60:.0f} min until kickoff...")
        time.sleep(wait_secs)

    monitor_positions(client, placed, dry_run)

    print("\n--- Session Summary ---")
    for pos in placed:
        print(f"  {pos.match_key:<35} status={pos.status:<8} locked=£{pos.locked_profit:.2f}  notes={pos.notes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lay the Draw automated trader")
    parser.add_argument("--bank",       type=float, default=500.0)
    parser.add_argument("--max-stake",  type=float, default=30.0)
    parser.add_argument("--kelly",      type=float, default=0.25)
    parser.add_argument("--dry-run",    action="store_true", help="Simulate (default unless --live)")
    parser.add_argument("--live",       action="store_true")
    args = parser.parse_args()

    if args.live and args.dry_run:
        parser.error("--live and --dry-run are mutually exclusive")

    run(
        bank=args.bank,
        max_stake=args.max_stake,
        dry_run=not args.live,
        live_key=args.live,
        kelly=args.kelly,
    )
