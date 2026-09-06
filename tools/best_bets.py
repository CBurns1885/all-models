#!/usr/bin/env python3
"""
Unified best-bets output across all markets and leagues.
Score = confidence × hist_accuracy × (1 + hist_roi/100)
Reads: latest weekly_bets_full.csv + outputs/league_breakdown.csv
Writes: best_bets.csv / best_bets.html in the same dated folder.
"""
import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# This file lives in tools/ — outputs/ is at the repo root, one level up.
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT))

from staking import (size_bet, break_even_odds, DEFAULT_COMMISSION,
                     DEFAULT_KELLY, DEFAULT_MIN_EDGE, DEFAULT_MAX_FRACTION,
                     DEFAULT_MIN_STAKE)


def _load_odds(out_dir: Path, odds_file: str | None):
    """Betfair price snapshot keyed by (Date, Home, Away, Market, Bet).

    Produced by betfair/betfair_odds.py (read-only price fetch). Missing file
    is not an error — picks still get a break-even price, just no stake.
    """
    path = Path(odds_file) if odds_file else out_dir / "betfair_odds.csv"
    if not path.exists():
        return {}, path
    try:
        odds = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read odds file {path}: {e}")
        return {}, path
    lookup = {}
    for _, r in odds.iterrows():
        key = (str(r.get("Date", ""))[:10], str(r.get("Home", "")),
               str(r.get("Away", "")), str(r.get("Market", "")), str(r.get("Bet", "")))
        price = r.get("BackPrice")
        if price == price and price is not None:
            lookup[key] = float(price)
    return lookup, path

# Home-prior fallback sentinel (19/21): emitted for fixtures where neither
# ML nor DC had any training data for the teams (e.g. European qualifiers).
# These predictions are pure prior — they must NEVER reach the betting layer,
# where their fake 90% confidence would look like a huge edge.
FALLBACK_1X2_H = 19.0 / 21.0
_FALLBACK_TOL = 1e-4


def _is_fallback_row(match) -> bool:
    """True if this fixture's 1X2 probabilities are the no-data fallback prior."""
    for col in ("P_1X2_H", "BLEND_1X2_H"):
        if col in match.index:
            v = match[col]
            if v == v and abs(float(v) - FALLBACK_1X2_H) < _FALLBACK_TOL:
                return True
    return False

# (market_name_in_breakdown) -> [list of prediction columns to consider]
# For each market we take the column with the highest probability as the predicted bet.
MARKET_COLS = {
    "1X2":              ["BLEND_1X2_H", "BLEND_1X2_D", "BLEND_1X2_A"],
    "BTTS":             ["BLEND_BTTS_Y", "BLEND_BTTS_N"],
    "OU_0_5":           ["BLEND_OU_0_5_O", "BLEND_OU_0_5_U"],
    "OU_1_5":           ["BLEND_OU_1_5_O", "BLEND_OU_1_5_U"],
    "OU_2_5":           ["BLEND_OU_2_5_O", "BLEND_OU_2_5_U"],
    "OU_3_5":           ["BLEND_OU_3_5_O", "BLEND_OU_3_5_U"],
    "OU_4_5":           ["BLEND_OU_4_5_O", "BLEND_OU_4_5_U"],
    "HomeTG_0_5":       ["BLEND_HomeTG_0_5_O", "BLEND_HomeTG_0_5_U"],
    "HomeTG_1_5":       ["BLEND_HomeTG_1_5_O", "BLEND_HomeTG_1_5_U"],
    "AwayTG_0_5":       ["BLEND_AwayTG_0_5_O", "BLEND_AwayTG_0_5_U"],
    "AwayTG_1_5":       ["BLEND_AwayTG_1_5_O", "BLEND_AwayTG_1_5_U"],
    "TotalYC_O1_5":     ["P_TotalYC_O1_5_Y",  "P_TotalYC_O1_5_N"],
    "TotalYC_O2_5":     ["P_TotalYC_O2_5_Y",  "P_TotalYC_O2_5_N"],
    "TotalYC_O3_5":     ["P_TotalYC_O3_5_Y",  "P_TotalYC_O3_5_N"],
    "TotalYC_O4_5":     ["P_TotalYC_O4_5_Y",  "P_TotalYC_O4_5_N"],
    "TotalYC_O5_5":     ["P_TotalYC_O5_5_Y",  "P_TotalYC_O5_5_N"],
    "TotalYC_O6_5":     ["P_TotalYC_O6_5_Y",  "P_TotalYC_O6_5_N"],
    "BookingPts_O20_5": ["P_BookingPts_O20_5_Y", "P_BookingPts_O20_5_N"],
    "BookingPts_O30_5": ["P_BookingPts_O30_5_Y", "P_BookingPts_O30_5_N"],
    "BookingPts_O40_5": ["P_BookingPts_O40_5_Y", "P_BookingPts_O40_5_N"],
    "HomeTeam_Card":    ["P_HomeTeam_Card_Y",  "P_HomeTeam_Card_N"],
    "AwayTeam_Card":    ["P_AwayTeam_Card_Y",  "P_AwayTeam_Card_N"],
    "TotalCorners_O6_5":  ["P_TotalCorners_O6_5_Y",  "P_TotalCorners_O6_5_N"],
    "TotalCorners_O7_5":  ["P_TotalCorners_O7_5_Y",  "P_TotalCorners_O7_5_N"],
    "TotalCorners_O8_5":  ["P_TotalCorners_O8_5_Y",  "P_TotalCorners_O8_5_N"],
    "TotalCorners_O9_5":  ["P_TotalCorners_O9_5_Y",  "P_TotalCorners_O9_5_N"],
    "TotalCorners_O10_5": ["P_TotalCorners_O10_5_Y", "P_TotalCorners_O10_5_N"],
    "TotalCorners_O11_5": ["P_TotalCorners_O11_5_Y", "P_TotalCorners_O11_5_N"],
    "TotalCorners_O12_5": ["P_TotalCorners_O12_5_Y", "P_TotalCorners_O12_5_N"],
    "HomeCorners_O4_5":   ["P_HomeCorners_O4_5_Y",   "P_HomeCorners_O4_5_N"],
    "HomeCorners_O5_5":   ["P_HomeCorners_O5_5_Y",   "P_HomeCorners_O5_5_N"],
    "AwayCorners_O4_5":   ["P_AwayCorners_O4_5_Y",   "P_AwayCorners_O4_5_N"],
    "AwayCorners_O5_5":   ["P_AwayCorners_O5_5_Y",   "P_AwayCorners_O5_5_N"],
    "HomeCorners_Win":    ["P_HomeCorners_Win_Y",     "P_HomeCorners_Win_N"],
}

# Human-readable outcome labels extracted from column suffixes
def _outcome_label(col: str) -> str:
    parts = col.split("_")
    suffix = parts[-1]
    if suffix == "Y": return "Yes"
    if suffix == "N": return "No"
    if suffix == "O": return "Over"
    if suffix == "U": return "Under"
    if suffix == "H": return "Home"
    if suffix == "D": return "Draw"
    if suffix == "A": return "Away"
    return suffix


# Cards/corners markets suffer a season cold-start problem: rolling-form
# discipline features reset each season, and the model stays confidently
# wrong for each team's first ~4 matches of a new season (honest 52-week
# backtest: e.g. TotalYC_O1_5 accuracy 43% at 91% stated confidence for
# min(Home_SeasonGames, Away_SeasonGames) < 4, vs 85% accuracy at 92%
# confidence once both teams have played 4+ games — see CLAUDE.md
# 2026-07-28 session notes). Attempted probability corrections (bucketed
# recalibration, linear log-odds regression) were tested and found unsafe —
# the miscalibration direction is inconsistent across markets (some
# overconfident, some underconfident) so a blind correction makes roughly
# half of them worse. Excluding the bet entirely is the only fix validated
# not to backfire.
_SEASON_COLD_START_MARKETS = {
    m for m in [
        "TotalYC_O1_5", "TotalYC_O2_5", "TotalYC_O3_5", "TotalYC_O4_5", "TotalYC_O5_5", "TotalYC_O6_5",
        "BookingPts_O20_5", "BookingPts_O30_5", "BookingPts_O40_5",
        "HomeTeam_Card", "AwayTeam_Card",
        "TotalCorners_O6_5", "TotalCorners_O7_5", "TotalCorners_O8_5", "TotalCorners_O9_5",
        "TotalCorners_O10_5", "TotalCorners_O11_5", "TotalCorners_O12_5",
        "HomeCorners_O4_5", "HomeCorners_O5_5", "AwayCorners_O4_5", "AwayCorners_O5_5",
        "HomeCorners_Win",
    ]
}


def _season_too_thin(match, market: str, min_games: float) -> bool:
    """True when this is a cold-start cards/corners market and either team
    hasn't played enough games yet this season for its rolling-form
    discipline features to be reliable."""
    if market not in _SEASON_COLD_START_MARKETS:
        return False
    worst = None
    for col in ("Home_SeasonGames", "Away_SeasonGames"):
        if col in match.index:
            v = match[col]
            if v == v:
                worst = float(v) if worst is None else min(worst, float(v))
    return worst is not None and worst < min_games


def _rd_too_high(match, max_rd: float) -> bool:
    """True when either team's Glicko rating deviation says 'we barely know
    this team' — the principled unknown-team gate (RD 350 = brand new)."""
    worst = 0.0
    found = False
    for col in ("Home_GlickoRD", "Away_GlickoRD"):
        if col in match.index:
            v = match[col]
            if v == v:
                worst = max(worst, float(v))
                found = True
    return found and worst > max_rd


def generate_best_bets(
    top_n: int = 200,
    min_confidence: float = 0.70,
    min_hist_preds: int = 20,
    min_hist_accuracy: float = 0.55,
    max_rd: float = 200.0,
    min_games: float = 4.0,
    bank: float = 0.0,
    kelly: float = DEFAULT_KELLY,
    commission: float = DEFAULT_COMMISSION,
    min_edge: float = DEFAULT_MIN_EDGE,
    max_fraction: float = DEFAULT_MAX_FRACTION,
    max_stake: float = None,
    min_stake: float = DEFAULT_MIN_STAKE,
    prob_shrink: float = 0.0,
    odds_file: str = None,
):
    # --- Load league breakdown ---
    breakdown_path = OUTPUTS / "league_breakdown.csv"
    if not breakdown_path.exists():
        print("[ERROR] league_breakdown.csv not found — run market_backtest.py --by-league first")
        return None
    breakdown = pd.read_csv(breakdown_path).set_index(["League", "Market"])

    # --- Find latest predictions file ---
    dated_dirs = sorted(d for d in OUTPUTS.iterdir() if d.is_dir() and d.name[:4].isdigit())
    preds_file = None
    out_dir = None
    for d in reversed(dated_dirs):
        f = d / "weekly_bets_full.csv"
        if f.exists():
            preds_file, out_dir = f, d
            break
    if preds_file is None:
        print("[ERROR] No weekly_bets_full.csv found")
        return None

    print(f"Loading predictions: {preds_file}")
    preds = pd.read_csv(preds_file)
    print(f"  {len(preds)} matches")

    # --- Betfair price snapshot (optional; enables stake sizing) ---
    odds_lookup, odds_path = _load_odds(out_dir, odds_file)
    if bank > 0:
        if odds_lookup:
            print(f"Staking: bank £{bank:,.2f}, {kelly:g}x Kelly, {commission:.0%} commission, "
                  f"cap {max_fraction:.0%} of bank  ({len(odds_lookup)} prices from {odds_path.name})")
        else:
            print(f"[WARN] No Betfair prices at {odds_path} — showing MinOdds (break-even) "
                  f"but no stakes.\n       Fetch prices first: py betfair/betfair_odds.py")

    # --- Build bet rows ---
    rows = []
    fallback_skipped = 0
    season_thin_skipped = 0
    for _, match in preds.iterrows():
        league = str(match.get("League", ""))
        date   = str(match.get("Date", ""))[:10]
        home   = str(match.get("HomeTeam", ""))
        away   = str(match.get("AwayTeam", ""))
        time_  = str(match.get("Time", "")) if "Time" in match.index else ""

        # Never bet on no-data fallback predictions or barely-known teams
        if _is_fallback_row(match) or _rd_too_high(match, max_rd):
            fallback_skipped += 1
            continue

        for market, cols in MARKET_COLS.items():
            if _season_too_thin(match, market, min_games):
                season_thin_skipped += 1
                continue

            # Filter to columns that exist
            avail = [c for c in cols if c in match.index]
            if not avail:
                continue

            # Pick highest-probability outcome
            vals = {c: float(match[c]) for c in avail}
            best_col = max(vals, key=vals.get)
            confidence = vals[best_col]

            if confidence < min_confidence:
                continue

            # Historical performance lookup
            try:
                hist = breakdown.loc[(league, market)]
                hist_acc   = float(hist["Acc"])
                hist_roi   = float(hist["ROI_Fair"])
                hist_preds = int(hist["Preds"])
            except KeyError:
                continue

            if hist_preds < min_hist_preds or hist_acc < min_hist_accuracy:
                continue

            # Score: confidence × accuracy × upside factor
            score = confidence * hist_acc * (1 + max(hist_roi, 0) / 100)

            outcome = _outcome_label(best_col)

            # --- Stake sizing (half Kelly by default) ---
            # MinOdds is the break-even price and is always shown; Odds/Stake
            # need a Betfair price snapshot (betfair/betfair_odds.py).
            min_odds = break_even_odds(confidence, commission)
            price = odds_lookup.get((date, home, away, market, outcome))
            advice = size_bet(
                confidence, price, bank,
                fraction=kelly, commission=commission, min_edge=min_edge,
                max_fraction=max_fraction, max_stake=max_stake,
                min_stake=min_stake, prob_shrink=prob_shrink,
            ) if bank > 0 else None

            row = {
                "Score":      round(score, 4),
                "Confidence": f"{confidence:.1%}",
                "Prob":       round(confidence, 6),   # raw probability
                "Market":     market,
                "Bet":        outcome,
                "League":     league,
                "Date":       date,
                "Time":       time_,
                "Home":       home,
                "Away":       away,
                "MinOdds":    round(min_odds, 3) if min_odds else "",
                "Odds":       round(price, 2) if price else "",
                "Edge":       f"{advice.edge:+.1%}" if advice and advice.edge is not None else "",
                "EV":         f"{advice.ev_per_unit:+.3f}" if advice and advice.ev_per_unit is not None else "",
                "Stake":      round(advice.stake, 2) if advice else "",
                "StakeNote":  advice.reason if advice else "",
                "Hist_Acc":   f"{hist_acc:.1%}",
                "Hist_ROI":   f"{hist_roi:+.1f}%",
                "Hist_n":     hist_preds,
                "_score_raw": score,
            }
            rows.append(row)

    if fallback_skipped:
        print(f"[GUARD] Skipped {fallback_skipped} fixture(s) with no-data fallback predictions")
    if season_thin_skipped:
        print(f"[GUARD] Skipped {season_thin_skipped} cards/corners bet(s) — a team has played "
              f"< {min_games:.0f} games this season (cold-start miscalibration)")

    if not rows:
        print("[WARN] No qualifying bets found — try lowering min_confidence or min_hist_accuracy")
        return None

    df = (pd.DataFrame(rows)
          .sort_values("_score_raw", ascending=False)
          .drop(columns=["_score_raw"])
          .head(top_n)
          .reset_index(drop=True))
    df.index += 1

    # --- Save CSV ---
    csv_path = out_dir / "best_bets.csv"
    df.to_csv(csv_path, index_label="Rank")
    print(f"[OK] Saved {len(df)} bets -> {csv_path}")

    # --- Save HTML ---
    html_path = out_dir / "best_bets.html"
    _write_html(df, html_path)
    print(f"[OK] Saved HTML -> {html_path}")

    # --- Print top 30 ---
    print(f"\n{'Rk':>3}  {'Score':>6}  {'Conf':>6}  {'MinOdd':>6}  {'Odds':>6}  {'Edge':>6}  {'Stake':>7}  "
          f"{'Hist%':>6}  {'ROI':>7}  {'n':>5}  {'Market':<20}  {'Bet':<6}  {'League':<6}  {'Date':<11}  Match")
    print("-" * 165)
    for rank, row in df.head(30).iterrows():
        print(f"{rank:>3}  {row['Score']:>6}  {row['Confidence']:>6}  {str(row['MinOdds']):>6}  "
              f"{str(row['Odds']):>6}  {str(row['Edge']):>6}  {str(row['Stake']):>7}  "
              f"{row['Hist_Acc']:>6}  {row['Hist_ROI']:>7}  {row['Hist_n']:>5}  {row['Market']:<20}  "
              f"{row['Bet']:<6}  {row['League']:<6}  {row['Date']:<11}  {row['Home']} v {row['Away']}")

    # --- Staking summary ---
    if bank > 0:
        stakes = pd.to_numeric(df["Stake"], errors="coerce").fillna(0.0)
        backed = df[stakes > 0]
        total = float(stakes.sum())
        print(f"\n{'='*60}")
        print(f"STAKING SUMMARY ({kelly:g}x Kelly on £{bank:,.2f} bank)")
        print(f"{'='*60}")
        print(f"  Picks with a positive-edge price : {len(backed)} / {len(df)}")
        print(f"  Total staked                     : £{total:,.2f} ({total/bank:.1%} of bank)")
        if len(backed):
            print(f"  Largest single stake             : £{stakes.max():,.2f}")
            print(f"  Median stake                     : £{stakes[stakes > 0].median():,.2f}")
        no_price = (df["Odds"].astype(str) == "").sum()
        if no_price:
            print(f"  No Betfair price available       : {no_price} (fetch with betfair/betfair_odds.py)")
        skipped_reasons = df.loc[stakes == 0, "StakeNote"].value_counts()
        for reason, n in skipped_reasons.items():
            if reason and reason != "ok":
                print(f"  Not staked — {reason:<28}: {n}")

    return df


def _write_html(df: pd.DataFrame, path: Path):
    rows_html = ""
    for rank, row in df.iterrows():
        stake_val = row.get("Stake", "")
        stake_cell = f"<b>£{stake_val}</b>" if stake_val not in ("", 0, 0.0) else "—"
        rows_html += (
            f"<tr><td>{rank}</td>"
            f"<td>{row['Score']}</td>"
            f"<td>{row['Confidence']}</td>"
            f"<td>{row['Market']}</td>"
            f"<td><b>{row['Bet']}</b></td>"
            f"<td>{row['League']}</td>"
            f"<td>{row['Date']}</td>"
            f"<td>{row['Home']}</td>"
            f"<td>{row['Away']}</td>"
            f"<td>{row.get('MinOdds','')}</td>"
            f"<td>{row.get('Odds','')}</td>"
            f"<td>{row.get('Edge','')}</td>"
            f"<td class='stake'>{stake_cell}</td>"
            f"<td>{row['Hist_Acc']}</td>"
            f"<td>{row['Hist_ROI']}</td>"
            f"<td>{row['Hist_n']}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Best Bets</title>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 13px; background: #111; color: #eee; margin: 20px; }}
  h1 {{ color: #f0c040; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #222; color: #f0c040; padding: 8px 10px; text-align: left; position: sticky; top: 0; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #333; }}
  tr:hover {{ background: #1e1e2e; }}
  .score-high {{ color: #4caf50; font-weight: bold; }}
  .score-med  {{ color: #ff9800; }}
  .stake {{ color: #4caf50; }}
  caption {{ caption-side: top; text-align: left; color: #888; margin-bottom: 8px; font-size: 12px; }}
</style>
</head>
<body>
<h1>Best Bets — Ranked by Score</h1>
<p style="color:#aaa">Score = Confidence × Historical Accuracy × (1 + Historical ROI). All markets and leagues unified.</p>
<p style="color:#aaa">Min Odds = break-even price for the model's probability (after commission). Stake = half-Kelly
recommendation against the Betfair back price; blank means no price was available or the price offered no edge.</p>
<table>
<caption>{len(df)} bets shown, ranked highest score first</caption>
<thead>
<tr>
  <th>#</th><th>Score</th><th>Confidence</th><th>Market</th><th>Bet</th>
  <th>League</th><th>Date</th><th>Home</th><th>Away</th>
  <th>Min Odds</th><th>Odds</th><th>Edge</th><th>Stake</th>
  <th>Hist Acc</th><th>Hist ROI</th><th>Hist n</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate unified best-bets ranking")
    parser.add_argument("--top", type=int, default=200, help="Number of bets to output")
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--min-hist-preds", type=int, default=20, help="Min historical predictions for a league/market pair")
    parser.add_argument("--min-hist-accuracy", type=float, default=0.55, help="Min historical accuracy (0-1)")
    parser.add_argument("--max-rd", type=float, default=200.0,
                        help="Skip fixtures where either team's Glicko RD exceeds this (350 = unknown team)")
    parser.add_argument("--min-games", type=float, default=4.0,
                        help="Skip cards/corners bets where either team has played fewer than this "
                             "many games this season (season cold-start guard)")

    stake_group = parser.add_argument_group("stake sizing (picks only — nothing is placed)")
    stake_group.add_argument("--bank", type=float, default=0.0,
                             help="Bankroll for Kelly sizing. 0 (default) = no stakes, show break-even odds only")
    stake_group.add_argument("--kelly", type=float, default=DEFAULT_KELLY,
                             help=f"Kelly fraction (default {DEFAULT_KELLY} = half Kelly)")
    stake_group.add_argument("--commission", type=float, default=DEFAULT_COMMISSION,
                             help=f"Exchange commission on net winnings (default {DEFAULT_COMMISSION})")
    stake_group.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE,
                             help=f"Minimum model-vs-market edge to stake (default {DEFAULT_MIN_EDGE})")
    stake_group.add_argument("--max-fraction", type=float, default=DEFAULT_MAX_FRACTION,
                             help=f"Cap any single stake at this fraction of bank (default {DEFAULT_MAX_FRACTION})")
    stake_group.add_argument("--max-stake", type=float, default=None,
                             help="Absolute cap on a single stake")
    stake_group.add_argument("--min-stake", type=float, default=DEFAULT_MIN_STAKE,
                             help=f"Exchange minimum stake (default {DEFAULT_MIN_STAKE})")
    stake_group.add_argument("--prob-shrink", type=float, default=0.0,
                             help="Shrink model probability toward market implied before sizing "
                                  "(0 = trust model, 0.3 = a defensive hedge against overconfidence)")
    stake_group.add_argument("--odds-file", type=str, default=None,
                             help="Betfair price CSV (default: betfair_odds.csv in the same dated folder)")
    args = parser.parse_args()

    if not (0 < args.kelly <= 1):
        parser.error("--kelly must be in (0, 1]; >1 is super-Kelly and is not supported")

    generate_best_bets(
        top_n=args.top,
        min_confidence=args.min_confidence,
        min_hist_preds=args.min_hist_preds,
        min_hist_accuracy=args.min_hist_accuracy,
        max_rd=args.max_rd,
        min_games=args.min_games,
        bank=args.bank,
        kelly=args.kelly,
        commission=args.commission,
        min_edge=args.min_edge,
        max_fraction=args.max_fraction,
        max_stake=args.max_stake,
        min_stake=args.min_stake,
        prob_shrink=args.prob_shrink,
        odds_file=args.odds_file,
    )
