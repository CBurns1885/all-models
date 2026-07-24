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

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

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


def generate_best_bets(
    top_n: int = 200,
    min_confidence: float = 0.70,
    min_hist_preds: int = 20,
    min_hist_accuracy: float = 0.55,
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

    # --- Build bet rows ---
    rows = []
    for _, match in preds.iterrows():
        league = str(match.get("League", ""))
        date   = str(match.get("Date", ""))[:10]
        home   = str(match.get("HomeTeam", ""))
        away   = str(match.get("AwayTeam", ""))

        for market, cols in MARKET_COLS.items():
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

            rows.append({
                "Score":      round(score, 4),
                "Confidence": f"{confidence:.1%}",
                "Market":     market,
                "Bet":        _outcome_label(best_col),
                "League":     league,
                "Date":       date,
                "Home":       home,
                "Away":       away,
                "Hist_Acc":   f"{hist_acc:.1%}",
                "Hist_ROI":   f"{hist_roi:+.1f}%",
                "Hist_n":     hist_preds,
                "_score_raw": score,
            })

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
    print(f"\n{'Rk':>3}  {'Score':>6}  {'Conf':>6}  {'Hist%':>6}  {'ROI':>7}  {'n':>5}  {'Market':<22}  {'Bet':<6}  {'League':<6}  {'Date':<11}  Match")
    print("-" * 130)
    for rank, row in df.head(30).iterrows():
        print(f"{rank:>3}  {row['Score']:>6}  {row['Confidence']:>6}  {row['Hist_Acc']:>6}  "
              f"{row['Hist_ROI']:>7}  {row['Hist_n']:>5}  {row['Market']:<22}  {row['Bet']:<6}  "
              f"{row['League']:<6}  {row['Date']:<11}  {row['Home']} v {row['Away']}")

    return df


def _write_html(df: pd.DataFrame, path: Path):
    rows_html = ""
    for rank, row in df.iterrows():
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
  caption {{ caption-side: top; text-align: left; color: #888; margin-bottom: 8px; font-size: 12px; }}
</style>
</head>
<body>
<h1>Best Bets — Ranked by Score</h1>
<p style="color:#aaa">Score = Confidence × Historical Accuracy × (1 + Historical ROI). All markets and leagues unified.</p>
<table>
<caption>{len(df)} bets shown, ranked highest score first</caption>
<thead>
<tr>
  <th>#</th><th>Score</th><th>Confidence</th><th>Market</th><th>Bet</th>
  <th>League</th><th>Date</th><th>Home</th><th>Away</th>
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
    args = parser.parse_args()

    generate_best_bets(
        top_n=args.top,
        min_confidence=args.min_confidence,
        min_hist_preds=args.min_hist_preds,
        min_hist_accuracy=args.min_hist_accuracy,
    )
