# staking.py
"""
Kelly stake sizing for picks output.

PICKS ONLY — nothing here places a bet. It annotates each recommended pick
with the stake a given bankroll and Kelly fraction implies, so the sizing is
worked out and reviewable well before any money is involved.

Exchange Kelly
--------------
Backing at decimal odds `o` on an exchange that charges commission `c` on net
winnings:

    win :  profit = stake * (o - 1) * (1 - c)
    lose:  profit = -stake

so the net odds received per unit staked are  b = (o - 1) * (1 - c)  and the
Kelly-optimal fraction of the bankroll is

    f* = (p * b - (1 - p)) / b

Half Kelly (`fraction=0.5`) is the default: it gives ~75% of the log-growth
of full Kelly at ~half the variance, and — more importantly here — it is far
more forgiving when `p` is overstated, which is the realistic failure mode for
a model whose calibration is still being established. Kelly's edge is linear
in the probability error while its risk is quadratic, so a model that says 80%
when the truth is 70% is meaningfully over-betting at full Kelly and merely
sub-optimal at half.

Guards applied on top of raw Kelly:
  * no bet unless the edge clears `min_edge` (default 2%),
  * stake capped at `max_fraction` of the bank (default 5%) so one
    high-confidence pick cannot dominate the book,
  * stake capped at `max_stake`, floored at `min_stake` (exchange minimum),
  * `prob_shrink` optionally pulls the model probability toward the
    market-implied probability before sizing (0 = trust the model as-is).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

DEFAULT_COMMISSION = 0.05      # Betfair standard
DEFAULT_KELLY = 0.5            # half Kelly
DEFAULT_MIN_EDGE = 0.02
DEFAULT_MAX_FRACTION = 0.05    # never stake >5% of bank on one selection
DEFAULT_MIN_STAKE = 2.0        # exchange minimum


@dataclass
class StakeAdvice:
    odds: Optional[float]        # decimal back price used
    implied: Optional[float]     # 1/odds (gross, before commission)
    edge: Optional[float]        # model prob - implied prob
    ev_per_unit: Optional[float] # expected profit per 1.00 staked, after commission
    kelly_full: Optional[float]  # full-Kelly bankroll fraction
    kelly_used: Optional[float]  # fraction after applying `fraction` and caps
    stake: float                 # recommended stake (0 = no bet)
    reason: str                  # why the stake is what it is

    def as_dict(self) -> dict:
        return asdict(self)


def break_even_odds(prob: float, commission: float = DEFAULT_COMMISSION) -> Optional[float]:
    """Minimum decimal price at which this probability has zero EV.

    Solve  p * (o - 1) * (1 - c) - (1 - p) = 0  =>  o = 1 + (1-p) / (p * (1-c))

    Useful even with no odds to hand: it is the price to beat for the pick to
    be worth backing at all.
    """
    if not (0.0 < prob < 1.0):
        return None
    return 1.0 + (1.0 - prob) / (prob * (1.0 - commission))


def kelly_fraction(prob: float, odds: float,
                   commission: float = DEFAULT_COMMISSION) -> Optional[float]:
    """Full-Kelly fraction of bankroll for a back bet. None if inputs invalid."""
    if odds is None or odds <= 1.0:
        return None
    if not (0.0 < prob < 1.0):
        return None
    b = (odds - 1.0) * (1.0 - commission)
    if b <= 0:
        return None
    return (prob * b - (1.0 - prob)) / b


def size_bet(
    prob: float,
    odds: Optional[float],
    bank: float,
    fraction: float = DEFAULT_KELLY,
    commission: float = DEFAULT_COMMISSION,
    min_edge: float = DEFAULT_MIN_EDGE,
    max_fraction: float = DEFAULT_MAX_FRACTION,
    max_stake: Optional[float] = None,
    min_stake: float = DEFAULT_MIN_STAKE,
    prob_shrink: float = 0.0,
) -> StakeAdvice:
    """Recommended stake for one selection. stake=0 means 'do not back'."""
    if odds is None or not (odds > 1.0):
        return StakeAdvice(None, None, None, None, None, None, 0.0, "no_odds")
    if not (0.0 < prob < 1.0):
        return StakeAdvice(odds, 1.0 / odds, None, None, None, None, 0.0, "invalid_probability")

    implied = 1.0 / odds

    # Optionally shrink the model probability toward the market before sizing.
    p = (1.0 - prob_shrink) * prob + prob_shrink * implied

    edge = p - implied
    b = (odds - 1.0) * (1.0 - commission)
    ev = p * b - (1.0 - p)          # expected profit per unit staked
    f_full = kelly_fraction(p, odds, commission)

    if f_full is None or f_full <= 0:
        return StakeAdvice(odds, implied, edge, ev, f_full, 0.0, 0.0, "no_edge")
    if edge < min_edge:
        return StakeAdvice(odds, implied, edge, ev, f_full, 0.0, 0.0,
                           f"edge_below_min({min_edge:.1%})")

    f_used = f_full * fraction
    reason = "ok"
    if f_used > max_fraction:
        f_used = max_fraction
        reason = f"capped_at_{max_fraction:.0%}_of_bank"

    stake = bank * f_used
    if max_stake is not None and stake > max_stake:
        stake = max_stake
        reason = "capped_at_max_stake"
    if stake < min_stake:
        return StakeAdvice(odds, implied, edge, ev, f_full, f_used, 0.0,
                           f"below_min_stake({min_stake:g})")

    return StakeAdvice(odds, implied, edge, ev, f_full, f_used, round(stake, 2), reason)
