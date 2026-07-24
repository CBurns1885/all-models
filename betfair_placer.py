"""
Betfair automated placer.

Reads best_bets.csv, checks live Betfair prices, calculates edge,
sizes stakes with Kelly criterion, and places bets (or dry-runs).

Risk controls (adapted from exon risk manager):
  - Min edge threshold (default 5% after commission)
  - Max stake per bet (configurable)
  - Max daily loss circuit breaker
  - Max total daily exposure
  - Drawdown scaling: reduces stakes as drawdown grows

Usage:
  py betfair_placer.py --dry-run              # simulate, no real bets
  py betfair_placer.py --live --bank 500      # live, £500 bank
  py betfair_placer.py --live --bank 500 --max-stake 20  # cap individual bets at £20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import io
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
BET_LOG = OUTPUTS / "betfair_bets.csv"
COMMISSION = 0.05          # Betfair standard commission rate


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------

def kelly_stake(
    probability: float,
    back_price: float,
    bank: float,
    fraction: float = 0.25,
    max_stake: float = 50.0,
) -> float:
    """
    Quarter-Kelly stake sizing.

    Kelly fraction = (prob * (price - 1) - (1 - prob)) / (price - 1)
    Negative Kelly = no bet.
    """
    b = back_price - 1.0          # net decimal odds
    q = 1.0 - probability
    kelly_f = (probability * b - q) / b
    if kelly_f <= 0:
        return 0.0
    stake = bank * kelly_f * fraction
    return round(min(stake, max_stake), 2)


# ---------------------------------------------------------------------------
# Risk / circuit breaker
# ---------------------------------------------------------------------------

class BettingRiskManager:
    """Session-level risk controls."""

    def __init__(
        self,
        bank: float,
        max_daily_loss_pct: float = 0.10,   # stop if down >10% of bank
        max_daily_exposure_pct: float = 0.30,  # max total at-risk per day
        drawdown_scale_threshold: float = 0.05, # start scaling at 5% drawdown
    ):
        self.bank = bank
        self.peak = bank
        self.realised_pnl = 0.0
        self.daily_exposure = 0.0
        self.max_daily_loss = bank * max_daily_loss_pct
        self.max_daily_exposure = bank * max_daily_exposure_pct
        self.drawdown_scale_threshold = drawdown_scale_threshold
        self.paused = False

    @property
    def current_equity(self) -> float:
        return self.bank + self.realised_pnl

    @property
    def drawdown(self) -> float:
        self.peak = max(self.peak, self.current_equity)
        return (self.current_equity - self.peak) / self.peak

    def exposure_remaining(self) -> float:
        return max(0.0, self.max_daily_exposure - self.daily_exposure)

    def check(self, proposed_stake: float) -> tuple[float, str]:
        """
        Returns (approved_stake, reason).
        approved_stake = 0 means blocked.
        """
        if self.paused:
            return 0.0, "paused"

        # Daily loss circuit breaker
        if self.realised_pnl < -self.max_daily_loss:
            self.paused = True
            return 0.0, f"daily_loss_limit: P&L={self.realised_pnl:.2f}"

        # Daily exposure cap
        exposure_left = self.exposure_remaining()
        if proposed_stake > exposure_left:
            proposed_stake = round(exposure_left, 2)
            if proposed_stake < 2.0:
                return 0.0, "daily_exposure_cap"

        # Drawdown scaling (reduce stakes as we fall)
        dd = self.drawdown
        if dd < -self.drawdown_scale_threshold:
            scale = max(0.25, 1.0 + dd / self.drawdown_scale_threshold)
            proposed_stake = round(proposed_stake * scale, 2)
            logger.warning("Drawdown scaling %.0f%% applied (dd=%.1f%%)", scale * 100, dd * 100)

        if proposed_stake < 2.0:
            return 0.0, "stake_below_minimum"

        return proposed_stake, "ok"

    def record_exposure(self, stake: float):
        self.daily_exposure += stake

    def settle(self, stake: float, back_price: float, won: bool):
        if won:
            self.realised_pnl += stake * (back_price - 1) * (1 - COMMISSION)
        else:
            self.realised_pnl -= stake


# ---------------------------------------------------------------------------
# Bet logging
# ---------------------------------------------------------------------------

def log_bet(row: dict):
    df = pd.DataFrame([row])
    if BET_LOG.exists():
        existing = pd.read_csv(BET_LOG)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(BET_LOG, index=False)


# ---------------------------------------------------------------------------
# Main placer
# ---------------------------------------------------------------------------

def run_placer(
    bank: float = 500.0,
    max_stake: float = 50.0,
    min_edge: float = 0.05,
    kelly_fraction: float = 0.25,
    dry_run: bool = True,
    live_key: bool = False,
    top_n: int = 50,
):
    from betfair_auth import get_client
    from betfair_markets import find_market, get_best_back_price, OUTCOME_TO_RUNNER

    client = get_client(live=live_key)
    risk = BettingRiskManager(bank=bank)

    # Find latest best_bets.csv
    dated_dirs = sorted(d for d in OUTPUTS.iterdir() if d.is_dir() and d.name[:4].isdigit())
    bets_file = None
    for d in reversed(dated_dirs):
        f = d / "best_bets.csv"
        if f.exists():
            bets_file = f
            break

    if bets_file is None:
        print("[ERROR] No best_bets.csv found — run best_bets.py first")
        return

    bets = pd.read_csv(bets_file).head(top_n)
    print(f"Loaded {len(bets)} bets from {bets_file}")
    print(f"Bank: £{bank:.2f} | Max stake: £{max_stake:.2f} | Min edge: {min_edge:.0%} | {'DRY RUN' if dry_run else 'LIVE'}\n")

    placed = 0
    skipped = 0

    for _, bet in bets.iterrows():
        market_name = bet["Market"]
        outcome     = bet["Bet"]       # "Home", "Draw", "Away", "Yes", "No", "Over", "Under"
        confidence  = float(str(bet["Confidence"]).strip("%")) / 100
        date_str    = str(bet["Date"])
        home        = str(bet["Home"])
        away        = str(bet["Away"])
        league      = str(bet["League"])

        # Parse kickoff datetime (assume UTC noon if no time)
        try:
            kickoff = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=12, tzinfo=timezone.utc)
        except ValueError:
            logger.warning("Could not parse date: %s", date_str)
            continue

        # Skip past matches
        if kickoff < datetime.now(timezone.utc):
            continue

        # Find Betfair market
        market_id, runner_map = find_market(client, home, away, kickoff, market_name)
        if market_id is None:
            logger.debug("Market not found: %s %s v %s", market_name, home, away)
            skipped += 1
            continue

        # Find the runner for our predicted outcome
        bf_runner_name = OUTCOME_TO_RUNNER.get(outcome)
        if bf_runner_name is None:
            logger.debug("No runner mapping for outcome: %s", outcome)
            skipped += 1
            continue

        selection_id = runner_map.get(bf_runner_name)
        if selection_id is None:
            # Try case-insensitive match
            for k, v in runner_map.items():
                if k.lower() == bf_runner_name.lower():
                    selection_id = v
                    break
        if selection_id is None:
            logger.debug("Runner '%s' not in market %s: %s", bf_runner_name, market_id, list(runner_map.keys()))
            skipped += 1
            continue

        # Get live price
        back_price = get_best_back_price(client, market_id, selection_id)
        if back_price is None or back_price < 1.01:
            logger.debug("No price for %s %s", market_name, outcome)
            skipped += 1
            continue

        # Calculate edge (our probability vs implied probability)
        implied_prob = 1.0 / back_price
        edge = confidence - implied_prob
        net_edge = edge - (implied_prob * COMMISSION)   # after commission

        if net_edge < min_edge:
            logger.debug("Insufficient edge %.1f%% for %s %s v %s @ %.2f",
                         net_edge * 100, market_name, home, away, back_price)
            skipped += 1
            continue

        # Kelly stake
        raw_stake = kelly_stake(confidence, back_price, bank, kelly_fraction, max_stake)
        approved_stake, reason = risk.check(raw_stake)

        if approved_stake == 0:
            print(f"  BLOCKED  {market_name:<22} {outcome:<6} {home} v {away}  reason={reason}")
            skipped += 1
            continue

        # Place or simulate
        if dry_run:
            print(f"  DRY-RUN  {market_name:<22} {outcome:<6} {home} v {away:<20}  "
                  f"price={back_price:.2f}  conf={confidence:.0%}  edge={net_edge:+.1%}  stake=£{approved_stake:.2f}")
        else:
            import betfairlightweight.filters as filters
            try:
                resp = client.betting.place_orders(
                    market_id=market_id,
                    instructions=[
                        filters.place_instruction(
                            selection_id=selection_id,
                            side="BACK",
                            order_type="LIMIT",
                            limit_order=filters.limit_order(
                                size=approved_stake,
                                price=back_price,
                                persistence_type="LAPSE",
                            ),
                        )
                    ],
                )
                result = resp.place_instruction_reports[0] if resp.place_instruction_reports else None
                status = result.status if result else "UNKNOWN"
                bet_id = result.bet_id if result else None
                print(f"  PLACED   {market_name:<22} {outcome:<6} {home} v {away:<20}  "
                      f"price={back_price:.2f}  stake=£{approved_stake:.2f}  status={status}  id={bet_id}")
            except Exception as e:
                print(f"  ERROR    {market_name:<22} {outcome:<6} {home} v {away}: {e}")
                skipped += 1
                continue

        risk.record_exposure(approved_stake)
        log_bet({
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "market":     market_name,
            "outcome":    outcome,
            "home":       home,
            "away":       away,
            "league":     league,
            "kickoff":    kickoff.isoformat(),
            "market_id":  market_id,
            "price":      back_price,
            "confidence": confidence,
            "edge":       round(net_edge, 4),
            "stake":      approved_stake,
            "dry_run":    dry_run,
        })
        placed += 1

    print(f"\nDone: {placed} bets placed, {skipped} skipped")
    print(f"Total exposure: £{risk.daily_exposure:.2f} / £{risk.max_daily_exposure:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Betfair automated placer")
    parser.add_argument("--bank",         type=float, default=500.0, help="Total betting bank (£)")
    parser.add_argument("--max-stake",    type=float, default=50.0,  help="Max stake per bet (£)")
    parser.add_argument("--min-edge",     type=float, default=0.05,  help="Min edge after commission (0.05 = 5%%)")
    parser.add_argument("--kelly",        type=float, default=0.25,  help="Kelly fraction (0.25 = quarter-Kelly)")
    parser.add_argument("--top",          type=int,   default=50,    help="Use top N bets from best_bets.csv")
    parser.add_argument("--dry-run",      action="store_true", default=True,  help="Simulate without placing (default)")
    parser.add_argument("--live",         action="store_true", help="Place real bets (use live key)")
    args = parser.parse_args()

    run_placer(
        bank=args.bank,
        max_stake=args.max_stake,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly,
        dry_run=not args.live,
        live_key=args.live,
        top_n=args.top,
    )
