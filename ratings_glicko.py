# ratings_glicko.py
"""
Glicko-2 team ratings with explicit uncertainty.

Why Glicko-2 on top of the existing Elo: it carries a RATING DEVIATION (RD)
per team — a principled "how little do we know about this team" number that
grows during inactivity (new season, cup entrants, promoted sides) and
shrinks with evidence. That gives the models a direct signal for match-up
strength AND for data sparsity, and gives the betting layer a clean gate:
refuse to bet fixtures where either team's RD is still high, instead of
relying on the exact-float fallback sentinel.

Ratings are GLOBAL across leagues (European cups connect the pools), updated
match-by-match in date order using the standard Glicko-2 algorithm
(Glickman 2013) with each match treated as its own rating period, plus
RD inflation proportional to days of inactivity.

Feature columns produced (all PRE-match values — no leakage):
    Home_Glicko, Away_Glicko      rating (1500 = average)
    Home_GlickoRD, Away_GlickoRD  rating deviation (350 = unknown, ~50 = solid)
    Glicko_Diff                   Home_Glicko - Away_Glicko
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

SCALE = 173.7178
BASE_RATING = 1500.0
BASE_RD = 350.0
MIN_RD = 30.0
BASE_VOL = 0.06
TAU = 0.5                 # system constant: volatility change speed
INACTIVITY_PERIOD_DAYS = 30.0   # one "missed rating period" worth of RD growth


class _Team:
    __slots__ = ("mu", "phi", "sigma", "last_date")

    def __init__(self):
        self.mu = 0.0                     # (rating - 1500) / SCALE
        self.phi = BASE_RD / SCALE
        self.sigma = BASE_VOL
        self.last_date = None

    @property
    def rating(self) -> float:
        return self.mu * SCALE + BASE_RATING

    @property
    def rd(self) -> float:
        return self.phi * SCALE


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi ** 2))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _new_volatility(phi: float, v: float, delta: float, sigma: float, tau: float = TAU) -> float:
    """Illinois-algorithm volatility update (Glickman 2013, step 5)."""
    a = math.log(sigma * sigma)
    phi2 = phi * phi
    delta2 = delta * delta

    def f(x):
        ex = math.exp(x)
        return (ex * (delta2 - phi2 - v - ex)) / (2.0 * (phi2 + v + ex) ** 2) - (x - a) / (tau * tau)

    A = a
    if delta2 > phi2 + v:
        B = math.log(delta2 - phi2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0 and k < 100:
            k += 1
        B = a - k * tau

    fA, fB = f(A), f(B)
    for _ in range(100):
        if abs(B - A) < 1e-6:
            break
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA = fA / 2.0
        B, fB = C, fC
    return math.exp(A / 2.0)


def _one_side_update(me: _Team, opp: _Team, s: float) -> tuple:
    """Compute (mu', phi', sigma') for one team against one opponent."""
    g_j = _g(opp.phi)
    E_j = _E(me.mu, opp.mu, opp.phi)
    v = 1.0 / (g_j * g_j * E_j * (1.0 - E_j))
    delta = v * g_j * (s - E_j)
    sigma_new = _new_volatility(me.phi, v, delta, me.sigma)
    phi_star = math.sqrt(me.phi * me.phi + sigma_new * sigma_new)
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_new = me.mu + phi_new * phi_new * g_j * (s - E_j)
    return mu_new, max(phi_new, MIN_RD / SCALE), sigma_new


def _update_pair(th: _Team, ta: _Team, score_home: float):
    """One-match Glicko-2 update for both teams (score_home: 1 / 0.5 / 0).
    Both updates are computed from the PRE-match states, then applied."""
    upd_h = _one_side_update(th, ta, score_home)
    upd_a = _one_side_update(ta, th, 1.0 - score_home)
    th.mu, th.phi, th.sigma = upd_h
    ta.mu, ta.phi, ta.sigma = upd_a


def _inflate_for_inactivity(t: _Team, date: pd.Timestamp):
    """RD grows with sqrt of missed rating periods (standard Glicko-2 decay)."""
    if t.last_date is None:
        return
    periods = max(0.0, (date - t.last_date).days) / INACTIVITY_PERIOD_DAYS
    if periods > 0:
        phi = math.sqrt(t.phi * t.phi + t.sigma * t.sigma * periods)
        t.phi = min(phi, BASE_RD / SCALE)


def add_glicko_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match Glicko-2 rating/RD columns to a match dataframe.

    Requires Date, HomeTeam, AwayTeam, FTR. Rows are processed in date order;
    the stored values are the ratings BEFORE that match's update.
    """
    needed = {"Date", "HomeTeam", "AwayTeam", "FTR"}
    if not needed.issubset(df.columns):
        print("   [GLICKO] Skipping — required columns missing")
        return df

    order = df.sort_values("Date", kind="stable").index
    teams: Dict[str, _Team] = {}

    h_r = pd.Series(np.nan, index=df.index)
    a_r = pd.Series(np.nan, index=df.index)
    h_rd = pd.Series(np.nan, index=df.index)
    a_rd = pd.Series(np.nan, index=df.index)

    for idx in order:
        row = df.loc[idx]
        home, away = row["HomeTeam"], row["AwayTeam"]
        date = row["Date"]
        th = teams.setdefault(home, _Team())
        ta = teams.setdefault(away, _Team())

        _inflate_for_inactivity(th, date)
        _inflate_for_inactivity(ta, date)

        # Record PRE-match values
        h_r[idx], a_r[idx] = th.rating, ta.rating
        h_rd[idx], a_rd[idx] = th.rd, ta.rd

        ftr = row.get("FTR")
        if pd.isna(ftr):
            continue
        score = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        _update_pair(th, ta, score)
        th.last_date = date
        ta.last_date = date

    out = df.copy()
    out["Home_Glicko"] = h_r
    out["Away_Glicko"] = a_r
    out["Home_GlickoRD"] = h_rd
    out["Away_GlickoRD"] = a_rd
    out["Glicko_Diff"] = out["Home_Glicko"] - out["Away_Glicko"]
    print(f"   Added Glicko-2 features for {len(teams)} teams "
          f"(median RD {np.nanmedian(pd.concat([h_rd, a_rd])):.0f})")
    return out
