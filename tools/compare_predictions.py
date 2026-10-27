"""
compare_predictions.py — Compare our model vs API-Football's own predictions.

Outputs a HTML report and prints a terminal summary showing:
  - Our 1X2 probabilities vs APIF's percentages
  - Where we agree vs disagree (divergence signal)
  - Historical accuracy comparison (once results are in)

Usage:
    py tools/compare_predictions.py                        # this week's fixtures
    py tools/compare_predictions.py --csv path/to/bets.csv
    py tools/compare_predictions.py --historical --weeks 8 # past N weeks accuracy
"""

import sys
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import API_FOOTBALL_DB, OUTPUT_DIR


def _load_predictions_from_db(fixture_ids=None):
    conn = sqlite3.connect(API_FOOTBALL_DB)
    if fixture_ids:
        placeholders = ",".join("?" * len(fixture_ids))
        df = pd.read_sql_query(
            f"SELECT * FROM fixture_predictions WHERE fixture_id IN ({placeholders})",
            conn, params=fixture_ids
        )
    else:
        df = pd.read_sql_query("SELECT * FROM fixture_predictions", conn)
    conn.close()
    return df


def _load_results_from_db(fixture_ids):
    conn = sqlite3.connect(API_FOOTBALL_DB)
    placeholders = ",".join("?" * len(fixture_ids))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, home_goals, away_goals FROM fixtures
            WHERE fixture_id IN ({placeholders}) AND status = 'FT'""",
        conn, params=fixture_ids
    )
    conn.close()
    return df


def _actual_result(home_goals, away_goals):
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    if home_goals > away_goals:
        return "H"
    elif home_goals == away_goals:
        return "D"
    return "A"


def _our_pick(row):
    """Return H/D/A based on highest P_1X2 column."""
    h = row.get("P_1X2_H", 0) or 0
    d = row.get("P_1X2_D", 0) or 0
    a = row.get("P_1X2_A", 0) or 0
    mx = max(h, d, a)
    if mx == h:
        return "H", round(h * 100, 1)
    elif mx == d:
        return "D", round(d * 100, 1)
    return "A", round(a * 100, 1)


def _apif_pick(row):
    """Return H/D/A based on highest APIF percentage."""
    h = row.get("apif_home_pct") or 0
    d = row.get("apif_draw_pct") or 0
    a = row.get("apif_away_pct") or 0
    mx = max(h, d, a)
    if mx == h:
        return "H", round(h, 1)
    elif mx == d:
        return "D", round(d, 1)
    return "A", round(a, 1)


def build_comparison(csv_path=None, weeks_back=None):
    if csv_path is None:
        # Find most recent weekly_bets_full.csv
        candidates = sorted(OUTPUT_DIR.glob("*/weekly_bets_full.csv"), reverse=True)
        if not candidates:
            candidates = sorted(OUTPUT_DIR.glob("*/weekly_bets.csv"), reverse=True)
        if not candidates:
            print("[ERROR] No weekly_bets_full.csv found in outputs/")
            return None
        csv_path = candidates[0]

    print(f"[INFO] Loading: {csv_path}")
    bets = pd.read_csv(csv_path)

    if "fixture_id" not in bets.columns:
        print("[WARN] No fixture_id column in CSV — cannot join with APIF predictions")
        return None

    fixture_ids = bets["fixture_id"].dropna().astype(int).tolist()
    apif = _load_predictions_from_db(fixture_ids)

    if apif.empty:
        print("[WARN] No APIF predictions in DB yet for these fixtures. Run:")
        print("  py fetch/fetch_predictions.py --upcoming")
        return None

    merged = bets.merge(apif, on="fixture_id", how="inner")
    print(f"[INFO] {len(merged)} fixtures with both our predictions and APIF's")

    rows = []
    for _, r in merged.iterrows():
        our_side, our_pct = _our_pick(r)
        apif_side, apif_pct = _apif_pick(r)
        agree = our_side == apif_side
        rows.append({
            "Date":         r.get("Date", ""),
            "League":       r.get("League", ""),
            "Home":         r.get("HomeTeam", r.get("Home", "")),
            "Away":         r.get("AwayTeam", r.get("Away", "")),
            "Our_Pick":     our_side,
            "Our_Pct":      our_pct,
            "APIF_Pick":    apif_side,
            "APIF_Pct":     apif_pct,
            "Agree":        agree,
            "APIF_Advice":  r.get("apif_advice", ""),
            "fixture_id":   int(r["fixture_id"]),
        })

    df = pd.DataFrame(rows)
    agree_count = df["Agree"].sum()
    print(f"\n[SUMMARY] Agreement: {agree_count}/{len(df)} picks ({agree_count/len(df)*100:.1f}%)")
    disagree = df[~df["Agree"]]
    if not disagree.empty:
        print(f"\nDisagreements ({len(disagree)}):")
        for _, r in disagree.iterrows():
            print(f"  {r['Date']} {r['League']:4s}  {r['Home']} vs {r['Away']}")
            print(f"    Ours: {r['Our_Pick']} ({r['Our_Pct']}%)  |  APIF: {r['APIF_Pick']} ({r['APIF_Pct']}%)  |  {r['APIF_Advice']}")

    return df


def build_html(df, output_path):
    agree_count = int(df["Agree"].sum())
    total = len(df)

    rows_html = ""
    for _, r in df.sort_values(["Date", "League"]).iterrows():
        agree_cls = "agree" if r["Agree"] else "disagree"
        badge = "✓ AGREE" if r["Agree"] else "✗ SPLIT"
        rows_html += f"""
        <tr class="{agree_cls}">
          <td>{r['Date']}</td>
          <td>{r['League']}</td>
          <td>{r['Home']}</td>
          <td>{r['Away']}</td>
          <td><strong>{r['Our_Pick']}</strong> <span class="pct">{r['Our_Pct']}%</span></td>
          <td><strong>{r['APIF_Pick']}</strong> <span class="pct">{r['APIF_Pct']}%</span></td>
          <td><span class="badge {agree_cls}">{badge}</span></td>
          <td class="advice">{r['APIF_Advice'] or ''}</td>
        </tr>"""

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Prediction Comparison — Our Model vs API-Football</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0f0f1a; color: #e0e0e0; padding: 20px; }}
  h1 {{ color: #fff; }}
  .summary {{ background: #1a1a2e; border-radius: 8px; padding: 16px; margin-bottom: 20px; display: flex; gap: 40px; }}
  .stat {{ text-align: center; }}
  .stat .val {{ font-size: 2rem; font-weight: bold; color: #4ade80; }}
  .stat .lbl {{ font-size: 0.8rem; color: #888; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #1a1a2e; padding: 10px 8px; text-align: left; color: #888; font-size: 0.8rem; text-transform: uppercase; }}
  td {{ padding: 9px 8px; border-bottom: 1px solid #222; font-size: 0.85rem; }}
  tr.agree td {{ background: #0a1a0a; }}
  tr.disagree td {{ background: #1a0a0a; }}
  .pct {{ color: #888; font-size: 0.75rem; margin-left: 4px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; }}
  .badge.agree {{ background: #14532d; color: #4ade80; }}
  .badge.disagree {{ background: #450a0a; color: #f87171; }}
  .advice {{ color: #666; font-size: 0.75rem; max-width: 200px; }}
</style>
</head><body>
<h1>Prediction Comparison — Our Model vs API-Football</h1>
<div class="summary">
  <div class="stat"><div class="val">{total}</div><div class="lbl">Fixtures</div></div>
  <div class="stat"><div class="val" style="color:#4ade80">{agree_count}</div><div class="lbl">Agree</div></div>
  <div class="stat"><div class="val" style="color:#f87171">{total - agree_count}</div><div class="lbl">Disagree</div></div>
  <div class="stat"><div class="val">{agree_count/total*100:.0f}%</div><div class="lbl">Agreement Rate</div></div>
</div>
<table>
  <thead><tr>
    <th>Date</th><th>League</th><th>Home</th><th>Away</th>
    <th>Our Pick</th><th>APIF Pick</th><th>Status</th><th>APIF Advice</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<p style="color:#444;font-size:0.75rem;margin-top:20px">
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
</p>
</body></html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"[OK] Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    df = build_comparison(csv_path=csv_path)
    if df is None or df.empty:
        return

    if args.output:
        out = Path(args.output)
    else:
        # Save alongside the CSV
        if csv_path:
            out = csv_path.parent / "prediction_comparison.html"
        else:
            out = OUTPUT_DIR / "prediction_comparison.html"

    build_html(df, out)


if __name__ == "__main__":
    main()
