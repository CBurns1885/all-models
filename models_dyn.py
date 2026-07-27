# models_dyn.py
"""
Dynamic team-strength model (state-space Dixon-Coles).

Each team's attack and defence strength is a latent state that evolves as a
Gaussian random walk over time (Rue & Salvesen 2000 / Koopman & Lit 2015
style). Matches are processed chronologically; after each match the two
teams' states are updated with a scalar Kalman-style step:

    lam = exp(att_home - def_away + home_adv)     (home expected goals)
    mu  = exp(att_away - def_home)                (away expected goals)

    gain  = var / (var + R)                       (more uncertain -> bigger step)
    att  += gain * (goals_scored - expected)      (Poisson score gradient)
    var   = var * R / (var + R) + q * days_gap    (shrink on update, inflate with time)

This is the principled replacement for the ad-hoc "recent form multiplier"
in models_dc (last-5 goal ratios with hard clamps): form, momentum and
mean-reversion all fall out of the random-walk dynamics, and each team
carries an explicit uncertainty that grows during long gaps (new season,
cup teams) and shrinks with evidence.

Markets are priced from the shared DC score grid, so all goals markets stay
mutually consistent. Integrated as the "dyn" pseudo-base in models.py.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from models_dc import price_from_lambdas

MAX_GOALS = 8

# --- Model constants ---
INIT_VAR      = 0.30    # initial state variance (new team = very uncertain)
OBS_NOISE_R   = 1.2     # observation noise: one match tells you only so much
WALK_Q_PER_DAY = 0.0004 # variance added per day since the team last played
HOME_ADV_INIT = 0.25
HOME_ADV_LR   = 0.002   # slow per-league home advantage adaptation
STATE_CLIP    = 1.5     # |attack|, |defence| bound (log-scale sanity)
VAR_FLOOR     = 0.005
VAR_CEIL      = INIT_VAR

_DYN_CACHE: Dict[str, "DynStates"] = {}


@dataclass
class TeamState:
    att: float = 0.0
    deff: float = 0.0
    var_att: float = INIT_VAR
    var_def: float = INIT_VAR
    last_date: Optional[pd.Timestamp] = None
    n_matches: int = 0


@dataclass
class DynStates:
    teams: Dict[tuple, TeamState] = field(default_factory=dict)   # (league, team) -> state
    home_adv: Dict[str, float] = field(default_factory=dict)      # league -> home advantage
    base_rate: Dict[str, float] = field(default_factory=dict)     # league -> log avg goals/team

    def get(self, league: str, team: str) -> TeamState:
        return self.teams.setdefault((league, team), TeamState())


def _inflate(state: TeamState, date: pd.Timestamp):
    """Random-walk variance inflation for the time since the team last played."""
    if state.last_date is not None:
        days = max(0.0, (date - state.last_date).days)
        state.var_att = min(VAR_CEIL, state.var_att + WALK_Q_PER_DAY * days)
        state.var_def = min(VAR_CEIL, state.var_def + WALK_Q_PER_DAY * days)


def fit_dynamic(train_df: pd.DataFrame, use_cache: bool = True) -> Optional[DynStates]:
    """Run the filter over completed matches in date order."""
    sub = train_df.dropna(subset=["League", "Date", "HomeTeam", "AwayTeam",
                                  "FTHG", "FTAG"]).copy()
    if len(sub) < 200:
        return None
    sub["Date"] = pd.to_datetime(sub["Date"])
    sub = sub.sort_values("Date", kind="stable")

    key = hashlib.md5(
        f"dyn|{str(sub['Date'].min())[:10]}|{str(sub['Date'].max())[:10]}|{len(sub)}".encode()
    ).hexdigest()[:16]
    if use_cache and key in _DYN_CACHE:
        return _DYN_CACHE[key]

    states = DynStates()

    # Per-league scoring base rate (log of average goals per team per match)
    for lg, grp in sub.groupby("League"):
        avg = float((grp["FTHG"].mean() + grp["FTAG"].mean()) / 2.0)
        states.base_rate[lg] = np.log(max(avg, 0.3))
        states.home_adv[lg] = HOME_ADV_INIT

    for row in sub.itertuples(index=False):
        lg = row.League
        date = row.Date
        hg = float(row.FTHG)
        ag = float(row.FTAG)

        sh = states.get(lg, row.HomeTeam)
        sa = states.get(lg, row.AwayTeam)
        _inflate(sh, date)
        _inflate(sa, date)

        base = states.base_rate.get(lg, np.log(1.3))
        ha = states.home_adv.get(lg, HOME_ADV_INIT)

        lam = np.exp(np.clip(base + sh.att - sa.deff + ha, -3, 3))
        mu = np.exp(np.clip(base + sa.att - sh.deff, -3, 3))

        # Poisson score gradients (d loglik / d state)
        resid_h = hg - lam
        resid_a = ag - mu

        gain_att_h = sh.var_att / (sh.var_att + OBS_NOISE_R)
        gain_def_h = sh.var_def / (sh.var_def + OBS_NOISE_R)
        gain_att_a = sa.var_att / (sa.var_att + OBS_NOISE_R)
        gain_def_a = sa.var_def / (sa.var_def + OBS_NOISE_R)

        sh.att = float(np.clip(sh.att + gain_att_h * resid_h, -STATE_CLIP, STATE_CLIP))
        sa.deff = float(np.clip(sa.deff - gain_def_a * resid_h, -STATE_CLIP, STATE_CLIP))
        sa.att = float(np.clip(sa.att + gain_att_a * resid_a, -STATE_CLIP, STATE_CLIP))
        sh.deff = float(np.clip(sh.deff - gain_def_h * resid_a, -STATE_CLIP, STATE_CLIP))

        # Posterior variance shrink
        sh.var_att = max(VAR_FLOOR, sh.var_att * OBS_NOISE_R / (sh.var_att + OBS_NOISE_R))
        sh.var_def = max(VAR_FLOOR, sh.var_def * OBS_NOISE_R / (sh.var_def + OBS_NOISE_R))
        sa.var_att = max(VAR_FLOOR, sa.var_att * OBS_NOISE_R / (sa.var_att + OBS_NOISE_R))
        sa.var_def = max(VAR_FLOOR, sa.var_def * OBS_NOISE_R / (sa.var_def + OBS_NOISE_R))

        # Slow home-advantage adaptation from the home-goal residual sign
        states.home_adv[lg] = float(np.clip(
            ha + HOME_ADV_LR * np.tanh(resid_h - resid_a), 0.0, 0.6))

        sh.last_date = date
        sa.last_date = date
        sh.n_matches += 1
        sa.n_matches += 1

    if use_cache:
        _DYN_CACHE[key] = states
    return states


def price_fixture(states: DynStates, league: str, home: str, away: str,
                  max_goals: int = MAX_GOALS, min_matches: int = 3) -> dict:
    """Price a fixture from current states. Returns {} for unknown/thin teams
    (mirrors DC behaviour so downstream fallback handling is identical)."""
    kh, ka = (league, home), (league, away)
    if kh not in states.teams or ka not in states.teams:
        return {}
    sh, sa = states.teams[kh], states.teams[ka]
    if sh.n_matches < min_matches or sa.n_matches < min_matches:
        return {}

    base = states.base_rate.get(league, np.log(1.3))
    ha = states.home_adv.get(league, HOME_ADV_INIT)
    lam = float(np.exp(np.clip(base + sh.att - sa.deff + ha, -3, 3)))
    mu = float(np.exp(np.clip(base + sa.att - sh.deff, -3, 3)))

    # price_from_lambdas returns a cache-shared dict — copy before adding keys
    out = dict(price_from_lambdas(lam, mu, rho=-0.05, max_goals=max_goals))
    # Expose state uncertainty for diagnostics / gating
    out["_dyn_uncertainty"] = float(np.sqrt(
        sh.var_att + sh.var_def + sa.var_att + sa.var_def))
    return out


def dyn_prices_for_rows(train_df: pd.DataFrame, rows_df: pd.DataFrame,
                        max_goals: int = MAX_GOALS) -> list:
    """List of DC_*-keyed market dicts, one per fixture row."""
    states = fit_dynamic(train_df)
    if states is None:
        return [{} for _ in range(len(rows_df))]
    out = []
    for row in rows_df[["League", "HomeTeam", "AwayTeam"]].itertuples(index=False):
        out.append(price_fixture(states, row.League, row.HomeTeam, row.AwayTeam,
                                 max_goals=max_goals))
    return out
