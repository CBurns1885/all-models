#!/usr/bin/env python3
"""
Generate an HTML predictions sheet showing only markets where the backtest
found genuine edge. Reads backtest_summary.csv for the best markets, then
pulls those columns from weekly_bets.csv to build a focused betting card.

Usage:
    python generate_predictions_html.py
    python generate_predictions_html.py --backtest outputs/backtest_summary.csv --predictions outputs/weekly_bets.csv
    python generate_predictions_html.py --min-edge 5          # only markets with >=5% edge
    python generate_predictions_html.py --worldcup             # use World Cup paths
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Market name -> prediction column mapping
# ---------------------------------------------------------------------------

def _market_to_pred_columns(market: str):
    """
    Given a backtest market name (e.g. '1X2', 'BTTS', 'OU_2_5', 'AH_-0_5'),
    return (list_of_P_columns, list_of_BLEND_columns, list_of_outcome_labels).
    """
    m = market.strip()

    if m == '1X2':
        return (['P_1X2_H', 'P_1X2_D', 'P_1X2_A'],
                ['BLEND_1X2_H', 'BLEND_1X2_D', 'BLEND_1X2_A'],
                ['Home', 'Draw', 'Away'])
    if m == 'BTTS':
        return (['P_BTTS_Y', 'P_BTTS_N'],
                ['BLEND_BTTS_Y', 'BLEND_BTTS_N'],
                ['Yes', 'No'])
    if m == 'GOAL_RANGE':
        labels = ['0', '1', '2', '3', '4', '5+']
        return ([f'P_GR_{k}' for k in labels],
                [f'BLEND_GR_{k}' for k in labels],
                labels)

    # Over/Under
    ou_match = re.match(r'^OU_(\d+_\d+)$', m)
    if ou_match:
        line = ou_match.group(1)
        pretty = line.replace('_', '.')
        return ([f'P_OU_{line}_O', f'P_OU_{line}_U'],
                [f'BLEND_OU_{line}_O', f'BLEND_OU_{line}_U'],
                [f'Over {pretty}', f'Under {pretty}'])

    # Asian Handicap
    ah_match = re.match(r'^AH_(.+)$', m)
    if ah_match:
        line = ah_match.group(1)
        pretty = line.replace('_', '.').replace('+', '+')
        return ([f'P_AH_{line}_H', f'P_AH_{line}_A', f'P_AH_{line}_P'],
                [f'BLEND_AH_{line}_H', f'BLEND_AH_{line}_A', f'BLEND_AH_{line}_P'],
                [f'Home {pretty}', f'Away {pretty}', f'Push {pretty}'])

    # Double Chance
    if m in ('DC_1X', 'DC_X2', 'DC_12'):
        tag = m.replace('DC_', '')
        return ([f'P_DC_{tag}_Y', f'P_DC_{tag}_N'],
                [], [f'{tag} Yes', f'{tag} No'])

    # Draw No Bet
    if m == 'DNB_H':
        return (['P_DNB_H_Y', 'P_DNB_H_N'], [], ['Home', 'Away'])
    if m == 'DNB_A':
        return (['P_DNB_A_Y', 'P_DNB_A_N'], [], ['Home', 'Away'])

    # Win to Nil
    if m == 'HomeWTN':
        return (['P_HomeWTN_Y', 'P_HomeWTN_N'], [], ['Yes', 'No'])
    if m == 'AwayWTN':
        return (['P_AwayWTN_Y', 'P_AwayWTN_N'], [], ['Yes', 'No'])

    # Clean Sheets
    if m == 'HomeCS':
        return (['P_HomeCS_Y', 'P_HomeCS_N'], [], ['Yes', 'No'])
    if m == 'AwayCS':
        return (['P_AwayCS_Y', 'P_AwayCS_N'], [], ['Yes', 'No'])

    # To Score
    if m == 'HomeToScore':
        return (['P_HomeToScore_Y', 'P_HomeToScore_N'], [], ['Yes', 'No'])
    if m == 'AwayToScore':
        return (['P_AwayToScore_Y', 'P_AwayToScore_N'], [], ['Yes', 'No'])

    # Half-time result
    if m == 'HT':
        return (['P_HT_H', 'P_HT_D', 'P_HT_A'], [], ['Home', 'Draw', 'Away'])

    # HT/FT
    if m == 'HTFT':
        combos = [f'{a}/{b}' for a in ['H', 'D', 'A'] for b in ['H', 'D', 'A']]
        p_cols = [f'P_HTFT_{a}_{b}' for a in ['H', 'D', 'A'] for b in ['H', 'D', 'A']]
        return (p_cols, [], combos)

    # Half-time O/U
    ht_ou = re.match(r'^HT_OU_(\d+_\d+)$', m)
    if ht_ou:
        line = ht_ou.group(1)
        pretty = line.replace('_', '.')
        return ([f'P_HT_OU_{line}_O', f'P_HT_OU_{line}_U'],
                [], [f'Over {pretty}', f'Under {pretty}'])

    # HT BTTS
    if m == 'HT_BTTS':
        return (['P_HT_BTTS_Y', 'P_HT_BTTS_N'], [], ['Yes', 'No'])

    # 2nd Half O/U
    h2_ou = re.match(r'^2H_OU_(\d+_\d+)$', m)
    if h2_ou:
        line = h2_ou.group(1)
        pretty = line.replace('_', '.')
        return ([f'P_2H_OU_{line}_O', f'P_2H_OU_{line}_U'],
                [], [f'Over {pretty}', f'Under {pretty}'])

    # 2nd Half BTTS
    if m == '2H_BTTS':
        return (['P_2H_BTTS_Y', 'P_2H_BTTS_N'], [], ['Yes', 'No'])

    # Odd/Even
    if m == 'TotalOddEven':
        return (['P_TotalOddEven_Odd', 'P_TotalOddEven_Even'], [], ['Odd', 'Even'])
    if m == 'HomeOddEven':
        return (['P_HomeOddEven_Odd', 'P_HomeOddEven_Even'], [], ['Odd', 'Even'])
    if m == 'AwayOddEven':
        return (['P_AwayOddEven_Odd', 'P_AwayOddEven_Even'], [], ['Odd', 'Even'])

    # Multi-Goal
    mg = re.match(r'^Match(\d+)\+Goals$', m)
    if mg:
        n = mg.group(1)
        return ([f'P_Match{n}+Goals_Y', f'P_Match{n}+Goals_N'], [],
                [f'{n}+ Goals Yes', f'{n}+ Goals No'])

    # Result+BTTS combos
    rb = re.match(r'^(HomeWin|AwayWin|Draw)_BTTS_(Y|N)$', m)
    if rb:
        res, btts = rb.group(1), rb.group(2)
        return ([f'P_{res}_BTTS_{btts}_Y', f'P_{res}_BTTS_{btts}_N'], [],
                ['Yes', 'No'])

    # Result+O/U combos
    rou = re.match(r'^(HomeWin|AwayWin|Draw)_(O25|U25)$', m)
    if rou:
        res, ou = rou.group(1), rou.group(2)
        return ([f'P_{res}_{ou}_Y', f'P_{res}_{ou}_N'], [], ['Yes', 'No'])

    # DC + O/U combos
    dcou = re.match(r'^DC(1X|X2|12)_(O25|U25)$', m)
    if dcou:
        dc, ou = dcou.group(1), dcou.group(2)
        return ([f'P_DC{dc}_{ou}_Y', f'P_DC{dc}_{ou}_N'], [], ['Yes', 'No'])

    # DC + BTTS combos
    dcb = re.match(r'^DC(1X|X2)_BTTS_(Y|N)$', m)
    if dcb:
        dc, btts = dcb.group(1), dcb.group(2)
        return ([f'P_DC{dc}_BTTS_{btts}_Y', f'P_DC{dc}_BTTS_{btts}_N'], [],
                ['Yes', 'No'])

    # Higher Half
    if m == 'HigherHalf':
        return (['P_HigherHalf_1H', 'P_HigherHalf_2H', 'P_HigherHalf_EQ'],
                [], ['1st Half', '2nd Half', 'Equal'])

    # Goals Both Halves
    if m == 'GoalsBothHalves':
        return (['P_GoalsBothHalves_Y', 'P_GoalsBothHalves_N'], [], ['Yes', 'No'])

    # Home/Away Scores Both Halves
    if m == 'HomeScoresBothHalves':
        return (['P_HomeScoresBothHalves_Y', 'P_HomeScoresBothHalves_N'], [], ['Yes', 'No'])
    if m == 'AwayScoresBothHalves':
        return (['P_AwayScoresBothHalves_Y', 'P_AwayScoresBothHalves_N'], [], ['Yes', 'No'])

    # Win Either/Both Halves
    for team in ['Home', 'Away']:
        for scope in ['EitherHalf', 'BothHalves']:
            if m == f'{team}Win{scope}':
                return ([f'P_{team}Win{scope}_Y', f'P_{team}Win{scope}_N'], [], ['Yes', 'No'])

    # First to Score
    if m == 'FirstToScore':
        return (['P_FirstToScore_H', 'P_FirstToScore_A', 'P_FirstToScore_None'],
                [], ['Home', 'Away', 'No Goal'])

    # Team Goals O/U
    tg = re.match(r'^(Home|Away)TG_(\d+_\d+)$', m)
    if tg:
        team, line = tg.group(1), tg.group(2)
        pretty = line.replace('_', '.')
        return ([f'P_{team}TG_{line}_O', f'P_{team}TG_{line}_U'],
                [], [f'Over {pretty}', f'Under {pretty}'])

    # Exact Team Goals
    te = re.match(r'^(Home|Away)Exact_(\d+\+?)$', m)
    if te:
        team, n = te.group(1), te.group(2)
        return ([f'P_{team}Exact_{n}_Y', f'P_{team}Exact_{n}_N'],
                [], [f'{n} Goals Yes', f'{n} Goals No'])

    # Exact Total Goals
    et = re.match(r'^ExactTotal_(\d+\+?)$', m)
    if et:
        n = et.group(1)
        return ([f'P_ExactTotal_{n}_Y', f'P_ExactTotal_{n}_N'],
                [], [f'{n} Goals Yes', f'{n} Goals No'])

    # No Goal
    if m == 'NoGoal':
        return (['P_NoGoal_Y', 'P_NoGoal_N'], [], ['Yes', 'No'])

    # Win by Margin
    wm = re.match(r'^(Home|Away)Win(By\d+\+?|2\+)$', m)
    if wm:
        team, margin = wm.group(1), wm.group(2)
        return ([f'P_{team}Win{margin}_Y', f'P_{team}Win{margin}_N'],
                [], ['Yes', 'No'])

    # European Handicap
    eh = re.match(r'^EH_(m\d+|p\d+)_(H|D|A)$', m)
    if eh:
        line, outcome = eh.group(1), eh.group(2)
        return ([f'P_EH_{line}_{outcome}_Y', f'P_EH_{line}_{outcome}_N'],
                [], ['Yes', 'No'])

    return ([], [], [])


def _pick_best_outcome(row, p_cols, blend_cols, labels):
    """Return (best_label, best_probability, source) using BLEND if available, else P_."""
    best_prob = 0.0
    best_label = '?'
    source = 'ML'

    # Try BLEND first
    for bc, label in zip(blend_cols, labels):
        if bc in row.index:
            val = row[bc]
            if pd.notna(val) and val > best_prob:
                best_prob = val
                best_label = label
                source = 'Blend'

    if best_prob > 0:
        return best_label, best_prob, source

    # Fall back to P_ columns
    for pc, label in zip(p_cols, labels):
        if pc in row.index:
            val = row[pc]
            if pd.notna(val) and val > best_prob:
                best_prob = val
                best_label = label
                source = 'ML'

    return best_label, best_prob, source


# ---------------------------------------------------------------------------
# Market display name
# ---------------------------------------------------------------------------

def _pretty_market_name(market: str) -> str:
    """Human-readable market name."""
    pretty = {
        '1X2': 'Match Result (1X2)',
        'BTTS': 'Both Teams To Score',
        'GOAL_RANGE': 'Goal Range',
        'HT': 'Half-Time Result',
        'HTFT': 'HT/FT Double',
        'HT_BTTS': 'HT Both Teams Score',
        '2H_BTTS': '2nd Half BTTS',
        'TotalOddEven': 'Total Goals Odd/Even',
        'HomeOddEven': 'Home Goals Odd/Even',
        'AwayOddEven': 'Away Goals Odd/Even',
        'GoalsBothHalves': 'Goals in Both Halves',
        'HomeScoresBothHalves': 'Home Scores Both Halves',
        'AwayScoresBothHalves': 'Away Scores Both Halves',
        'HigherHalf': 'Higher Scoring Half',
        'FirstToScore': 'First to Score',
        'NoGoal': 'No Goal (0-0)',
        'HomeToScore': 'Home to Score',
        'AwayToScore': 'Away to Score',
        'HomeCS': 'Home Clean Sheet',
        'AwayCS': 'Away Clean Sheet',
        'HomeWTN': 'Home Win to Nil',
        'AwayWTN': 'Away Win to Nil',
        'DNB_H': 'Draw No Bet - Home',
        'DNB_A': 'Draw No Bet - Away',
    }
    if market in pretty:
        return pretty[market]
    m = market
    m = re.sub(r'^OU_(\d+)_(\d+)$', r'Over/Under \1.\2', m)
    m = re.sub(r'^AH_(.+)$', lambda x: f'Asian Handicap {x.group(1).replace("_", ".").replace("+", "+")}', m)
    m = re.sub(r'^HT_OU_(\d+)_(\d+)$', r'HT Over/Under \1.\2', m)
    m = re.sub(r'^2H_OU_(\d+)_(\d+)$', r'2H Over/Under \1.\2', m)
    m = re.sub(r'^DC_(1X|X2|12)$', r'Double Chance \1', m)
    m = re.sub(r'^(Home|Away)TG_(\d+)_(\d+)$', r'\1 Goals O/U \2.\3', m)
    m = re.sub(r'^(Home|Away)Exact_(.+)$', r'\1 Exact \2 Goals', m)
    m = re.sub(r'^ExactTotal_(.+)$', r'Exact Total \1 Goals', m)
    m = re.sub(r'^Match(\d+)\+Goals$', r'\1+ Match Goals', m)
    m = re.sub(r'^(Home|Away)Win(By\d+\+?)$', r'\1 Win \2', m)
    m = re.sub(r'^(Home|Away)Win(2\+)$', r'\1 Win \2', m)
    m = re.sub(r'^(Home|Away)Win(EitherHalf)$', r'\1 Win Either Half', m)
    m = re.sub(r'^(Home|Away)Win(BothHalves)$', r'\1 Win Both Halves', m)
    m = re.sub(r'^(HomeWin|AwayWin|Draw)_BTTS_(Y|N)$', r'\1 & BTTS \2', m)
    m = re.sub(r'^(HomeWin|AwayWin|Draw)_(O25|U25)$', r'\1 & \2', m)
    m = re.sub(r'^DC(1X|X2|12)_(O25|U25)$', r'DC \1 & \2', m)
    m = re.sub(r'^DC(1X|X2)_BTTS_(Y|N)$', r'DC \1 & BTTS \2', m)
    m = re.sub(r'^EH_(m|p)(\d+)_(H|D|A)$',
               lambda x: f'Euro H/C {"−" if x.group(1)=="m" else "+"}{x.group(2)} {x.group(3)}', m)
    return m


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(backtest_path: Path, predictions_path: Path, output_path: Path,
                  min_edge: float = 0.0, min_matches: int = 10):
    """Build the HTML predictions sheet."""

    # Load backtest summary
    summary = pd.read_csv(backtest_path, index_col=0)
    print(f"Loaded backtest summary: {len(summary)} markets")

    # Filter to positive-edge markets with enough data
    if 'Edge_%' not in summary.columns:
        print("[ERROR] Backtest summary missing Edge_% column. Re-run backtest with latest code.")
        sys.exit(1)

    edge_markets = summary[
        (summary['Edge_%'] >= min_edge) &
        (summary['Total_Matches'] >= min_matches)
    ].sort_values('Edge_%', ascending=False)

    print(f"Markets with edge >= {min_edge}% and >= {min_matches} matches: {len(edge_markets)}")
    if edge_markets.empty:
        print("[WARN] No markets meet the criteria. Try lowering --min-edge.")
        return

    # Load predictions
    preds = pd.read_csv(predictions_path)
    print(f"Loaded predictions: {len(preds)} fixtures")

    if preds.empty:
        print("[ERROR] No predictions found.")
        return

    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Build per-fixture market cards
    fixture_cards = []
    for fix_idx, fix_row in preds.iterrows():
        home = fix_row.get('HomeTeam', '?')
        away = fix_row.get('AwayTeam', '?')
        date = fix_row.get('Date', '?')
        league = fix_row.get('League', '')

        bets = []
        for market_name, market_row in edge_markets.iterrows():
            p_cols, blend_cols, labels = _market_to_pred_columns(market_name)
            if not p_cols:
                continue

            available_p = [c for c in p_cols if c in preds.columns]
            available_b = [c for c in blend_cols if c in preds.columns]

            if not available_p and not available_b:
                continue

            pick, prob, source = _pick_best_outcome(fix_row, p_cols, blend_cols, labels)
            if prob <= 0:
                continue

            edge = market_row['Edge_%']
            accuracy = market_row['Accuracy_%']
            baseline = market_row['Baseline_%']
            matches = int(market_row['Total_Matches'])
            n_outcomes = int(market_row.get('Outcomes', 2))

            # Confidence tier
            if edge >= 15:
                tier = 'elite'
            elif edge >= 10:
                tier = 'strong'
            elif edge >= 5:
                tier = 'good'
            else:
                tier = 'marginal'

            # All probabilities for the mini-bar
            all_probs = []
            cols_to_use = available_b if available_b else available_p
            labels_to_use = labels[:len(cols_to_use)]
            for col, label in zip(cols_to_use, labels_to_use):
                val = fix_row.get(col, 0)
                if pd.notna(val) and val > 0:
                    all_probs.append((label, float(val)))

            bets.append({
                'market': market_name,
                'pretty_market': _pretty_market_name(market_name),
                'pick': pick,
                'prob': prob,
                'source': source,
                'edge': edge,
                'accuracy': accuracy,
                'baseline': baseline,
                'matches': matches,
                'n_outcomes': n_outcomes,
                'tier': tier,
                'all_probs': all_probs,
            })

        fixture_cards.append({
            'home': home,
            'away': away,
            'date': str(date)[:10],
            'league': league,
            'bets': sorted(bets, key=lambda b: -b['edge']),
        })

    # Count stats
    total_bets = sum(len(fc['bets']) for fc in fixture_cards)
    strong_bets = sum(1 for fc in fixture_cards for b in fc['bets'] if b['tier'] in ('elite', 'strong'))

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Edge-Based Predictions</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0f1117; color: #e1e4e8; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
.header {{ background: linear-gradient(135deg, #1a1f36 0%, #2d1b69 50%, #1a1f36 100%); padding: 30px; border-radius: 16px; margin-bottom: 24px; border: 1px solid #30363d; }}
.header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
.header .subtitle {{ color: #8b949e; font-size: 14px; }}
.stats-bar {{ display: flex; gap: 16px; margin-top: 16px; flex-wrap: wrap; }}
.stat {{ background: rgba(255,255,255,0.06); padding: 12px 20px; border-radius: 10px; text-align: center; min-width: 120px; }}
.stat .val {{ font-size: 24px; font-weight: 700; color: #58a6ff; }}
.stat .lbl {{ font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}

.edge-legend {{ display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }}
.edge-legend .chip {{ padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }}
.chip.elite {{ background: #1a472a; color: #3fb950; border: 1px solid #238636; }}
.chip.strong {{ background: #0d2748; color: #58a6ff; border: 1px solid #1f6feb; }}
.chip.good {{ background: #2a1e00; color: #d29922; border: 1px solid #9e6a03; }}
.chip.marginal {{ background: #1c1c1c; color: #8b949e; border: 1px solid #30363d; }}

.fixture {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; margin-bottom: 20px; overflow: hidden; }}
.fixture-header {{ padding: 16px 20px; background: #1c2333; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; }}
.fixture-header .teams {{ font-size: 18px; font-weight: 700; }}
.fixture-header .vs {{ color: #8b949e; font-weight: 400; margin: 0 8px; }}
.fixture-header .meta {{ font-size: 13px; color: #8b949e; }}
.fixture-body {{ padding: 16px 20px; }}

.bet-row {{ display: grid; grid-template-columns: 2fr 1fr 1fr 2fr; align-items: center; padding: 10px 0; border-bottom: 1px solid #21262d; }}
.bet-row:last-child {{ border-bottom: none; }}
.bet-market {{ font-weight: 600; font-size: 14px; }}
.bet-pick {{ text-align: center; }}
.bet-pick .pick-val {{ font-size: 15px; font-weight: 700; }}
.bet-pick .pick-prob {{ font-size: 12px; color: #8b949e; margin-top: 2px; }}
.bet-edge {{ text-align: center; }}
.bet-edge .edge-badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 700; }}
.bet-edge .edge-badge.elite {{ background: #1a472a; color: #3fb950; }}
.bet-edge .edge-badge.strong {{ background: #0d2748; color: #58a6ff; }}
.bet-edge .edge-badge.good {{ background: #2a1e00; color: #d29922; }}
.bet-edge .edge-badge.marginal {{ background: #1c1c1c; color: #8b949e; }}
.bet-probs {{ display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }}
.prob-chip {{ font-size: 11px; padding: 2px 8px; border-radius: 8px; background: #21262d; white-space: nowrap; }}
.prob-chip.is-pick {{ background: #1f6feb; color: #fff; font-weight: 600; }}
.bet-detail {{ font-size: 11px; color: #8b949e; margin-top: 2px; }}

.summary-table {{ width: 100%; border-collapse: collapse; margin-top: 24px; }}
.summary-table th {{ text-align: left; padding: 10px 12px; background: #1c2333; color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 2px solid #30363d; }}
.summary-table td {{ padding: 10px 12px; border-bottom: 1px solid #21262d; font-size: 14px; }}
.summary-table tr:hover {{ background: #1c2333; }}
.summary-table .num {{ text-align: right; font-variant-numeric: tabular-nums; }}

.section-title {{ font-size: 18px; font-weight: 700; margin: 30px 0 12px; padding-bottom: 8px; border-bottom: 1px solid #30363d; }}

.no-bets {{ padding: 20px; color: #8b949e; font-style: italic; text-align: center; }}

@media (max-width: 768px) {{
    .bet-row {{ grid-template-columns: 1fr 1fr; gap: 8px; }}
    .bet-probs {{ justify-content: flex-start; }}
    .stats-bar {{ gap: 8px; }}
    .stat {{ min-width: 90px; padding: 10px; }}
    .fixture-header {{ flex-direction: column; gap: 8px; }}
}}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>Edge-Based Predictions</h1>
    <p class="subtitle">Only markets where backtest found genuine edge | Generated {now}</p>
    <div class="stats-bar">
        <div class="stat"><div class="val">{len(fixture_cards)}</div><div class="lbl">Fixtures</div></div>
        <div class="stat"><div class="val">{len(edge_markets)}</div><div class="lbl">Edge Markets</div></div>
        <div class="stat"><div class="val">{total_bets}</div><div class="lbl">Total Bets</div></div>
        <div class="stat"><div class="val">{strong_bets}</div><div class="lbl">Strong+ Bets</div></div>
    </div>
    <div class="edge-legend" style="margin-top:12px;">
        <span class="chip elite">Elite: 15%+</span>
        <span class="chip strong">Strong: 10-15%</span>
        <span class="chip good">Good: 5-10%</span>
        <span class="chip marginal">Marginal: 0-5%</span>
    </div>
</div>
"""

    # Fixture cards
    for fc in fixture_cards:
        html += f"""
<div class="fixture">
    <div class="fixture-header">
        <div class="teams">{fc['home']}<span class="vs">vs</span>{fc['away']}</div>
        <div class="meta">{fc['date']} &middot; {fc['league']}</div>
    </div>
    <div class="fixture-body">
"""
        if not fc['bets']:
            html += '<div class="no-bets">No edge-based predictions available for this fixture</div>'
        else:
            html += '<div class="bet-row" style="font-weight:600;color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;border-bottom:2px solid #30363d;">'
            html += '<div>Market</div><div style="text-align:center">Pick</div><div style="text-align:center">Edge</div><div style="text-align:right">Probabilities</div></div>'

            for bet in fc['bets']:
                prob_chips = ''
                for lbl, p in bet['all_probs']:
                    is_pick = 'is-pick' if lbl == bet['pick'] else ''
                    prob_chips += f'<span class="prob-chip {is_pick}">{lbl} {p:.0%}</span>'

                html += f"""
        <div class="bet-row">
            <div class="bet-market">{bet['pretty_market']}<div class="bet-detail">{bet['accuracy']:.0f}% acc vs {bet['baseline']:.0f}% base ({bet['matches']} matches) &middot; {bet['source']}</div></div>
            <div class="bet-pick"><div class="pick-val">{bet['pick']}</div><div class="pick-prob">{bet['prob']:.0%}</div></div>
            <div class="bet-edge"><span class="edge-badge {bet['tier']}">+{bet['edge']:.1f}%</span></div>
            <div class="bet-probs">{prob_chips}</div>
        </div>"""

        html += """
    </div>
</div>"""

    # Backtest summary table
    html += """
<div class="section-title">Backtest Edge Summary (all qualifying markets)</div>
<table class="summary-table">
<thead><tr>
    <th>#</th><th>Market</th><th class="num">Matches</th><th class="num">Accuracy</th>
    <th class="num">Baseline</th><th class="num">Edge</th><th class="num">Brier</th>
    <th class="num">Outcomes</th><th>Source</th>
</tr></thead>
<tbody>
"""
    for rank, (market_name, row) in enumerate(edge_markets.iterrows(), 1):
        tier = 'elite' if row['Edge_%'] >= 15 else ('strong' if row['Edge_%'] >= 10 else ('good' if row['Edge_%'] >= 5 else 'marginal'))
        html += f"""<tr>
    <td>{rank}</td>
    <td><strong>{_pretty_market_name(market_name)}</strong></td>
    <td class="num">{int(row['Total_Matches'])}</td>
    <td class="num">{row['Accuracy_%']:.1f}%</td>
    <td class="num">{row['Baseline_%']:.1f}%</td>
    <td class="num"><span class="edge-badge {tier}">+{row['Edge_%']:.1f}%</span></td>
    <td class="num">{row['Brier_Score']:.3f}</td>
    <td class="num">{int(row.get('Outcomes', 2))}</td>
    <td>{row.get('Source', '?')}</td>
</tr>"""

    html += """
</tbody>
</table>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    print(f"\n[OK] HTML predictions saved to: {output_path}")
    print(f"     {len(fixture_cards)} fixtures x {len(edge_markets)} edge markets = {total_bets} bets")
    print(f"     Strong+ bets: {strong_bets}")


def main():
    parser = argparse.ArgumentParser(description="Generate edge-based predictions HTML")
    parser.add_argument('--backtest', type=str, help='Path to backtest_summary.csv')
    parser.add_argument('--predictions', type=str, help='Path to weekly_bets.csv')
    parser.add_argument('--output', type=str, help='Output HTML path')
    parser.add_argument('--min-edge', type=float, default=0.0, help='Minimum edge %% to include (default: 0 = all positive)')
    parser.add_argument('--min-matches', type=int, default=10, help='Minimum backtest matches to trust a market (default: 10)')
    parser.add_argument('--worldcup', action='store_true', help='Use World Cup output paths')
    args = parser.parse_args()

    if args.worldcup:
        default_bt = BASE_DIR / "outputs" / "worldcup" / "backtest_summary.csv"
        default_pred = BASE_DIR / "outputs" / "worldcup" / "weekly_bets.csv"
        default_out = BASE_DIR / "outputs" / "worldcup" / "edge_predictions.html"
    else:
        out_dir = BASE_DIR / "outputs" / datetime.now().strftime('%Y-%m-%d')
        default_bt = out_dir / "backtest_summary.csv"
        default_pred = out_dir / "weekly_bets.csv"
        default_out = out_dir / "edge_predictions.html"

    bt_path = Path(args.backtest) if args.backtest else default_bt
    pred_path = Path(args.predictions) if args.predictions else default_pred
    out_path = Path(args.output) if args.output else default_out

    if not bt_path.exists():
        print(f"[ERROR] Backtest summary not found: {bt_path}")
        print("Run the pipeline first, or use --backtest to specify the path.")
        sys.exit(1)
    if not pred_path.exists():
        print(f"[ERROR] Predictions file not found: {pred_path}")
        print("Run predictions first, or use --predictions to specify the path.")
        sys.exit(1)

    generate_html(bt_path, pred_path, out_path,
                  min_edge=args.min_edge, min_matches=args.min_matches)


if __name__ == '__main__':
    main()
