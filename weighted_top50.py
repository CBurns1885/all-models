# weighted_top50.py
"""
Generate weighted Top 50 bets based on historical market performance
Includes smart conflict resolution
"""

import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import json

OUTPUT_DIR = Path("outputs")
DB_PATH = OUTPUT_DIR / "accuracy_database.db"
WEIGHTS_PATH = OUTPUT_DIR / "blend_weights.json"

def calculate_market_weights() -> Dict[str, float]:
    """Calculate weights for each market based on historical accuracy"""
    
    if not DB_PATH.exists():
        print("âš ï¸ No accuracy database found, using default weights")
        return _default_weights()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # FIXED query - use correct column names
        query = """
        SELECT market, 
               COUNT(*) as total,
               SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
               AVG(CAST(predicted_prob AS FLOAT)) as avg_confidence
        FROM predictions
        WHERE actual_result IS NOT NULL
        GROUP BY market
        HAVING total >= 10
        """
        
        # If predicted_prob doesn't exist, use simpler query
        try:
            df = pd.read_sql_query(query, conn)
        except:
            # Fallback query without avg_confidence
            query = """
            SELECT market, 
                   COUNT(*) as total,
                   SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct
            FROM predictions
            WHERE actual_result IS NOT NULL
            GROUP BY market
            HAVING total >= 10
            """
            df = pd.read_sql_query(query, conn)
            df['avg_confidence'] = 0.7  # Default
        
        conn.close()
        
        if len(df) == 0:
            print("âš ï¸ Not enough historical data, using default weights")
            return _default_weights()
        
        # Calculate weights based on accuracy
        df['accuracy'] = df['correct'] / df['total']
        df['weight'] = df['accuracy'] * df.get('avg_confidence', 0.7)
        
        # Normalize weights
        total_weight = df['weight'].sum()
        df['weight'] = df['weight'] / total_weight
        
        weights = dict(zip(df['market'], df['weight']))
        
        # Save weights
        with open(WEIGHTS_PATH, 'w') as f:
            json.dump(weights, f, indent=2)
        
        print(f"âœ… Calculated weights for {len(weights)} markets")
        return weights
        
    except Exception as e:
        print(f"âš ï¸ Error calculating weights: {e}")
        return _default_weights()


def _default_weights() -> Dict[str, float]:
    """Default weights when no historical data available"""
    return {
        '1X2_H': 1.0,
        '1X2_D': 0.8,
        '1X2_A': 1.0,
        'BTTS_Y': 1.2,
        'BTTS_N': 1.0,
        'O_0_5_O': 0.2,
        'O_0_5_U': 0.2,
        'O_1_5_O': 2.0,
        'O_1_5_U': 2.0,
        'OU_2_5_O': 1.1,
        'OU_2_5_U': 1.1,
        'OU_3_5_O': 1.1,
        'OU_3_5_U': 1.1,
        'CS_1_0': 0.6,
        'CS_2_0': 0.6,
        'CS_2_1': 0.7,
        'CS_0_0': 0.5,
        'CS_1_1': 0.6,
    }


def extract_predictions_from_csv(csv_path: Path) -> List[Dict]:
    """Extract all predictions from weekly_bets.csv"""
    
    df = pd.read_csv(csv_path)
    
    predictions = []
    
    for idx, row in df.iterrows():
        match_id = f"{row['HomeTeam']}_vs_{row['AwayTeam']}"
        
        # Extract all probability columns
        for col in df.columns:
            if col.startswith('BLEND_') or col.startswith('P_'):
                prob = row[col]
                
                if pd.notna(prob) and prob > 0:
                    # Parse market name
                    market = col.replace('BLEND_', '').replace('P_', '')
                    
                    predictions.append({
                        'match_id': match_id,
                        'league': row.get('League', ''),
                        'date': row.get('Date', ''),
                        'home_team': row['HomeTeam'],
                        'away_team': row['AwayTeam'],
                        'market': market,
                        'probability': prob,
                        'odds': 1 / prob if prob > 0 else 999,
                    })
    
    return predictions


def resolve_conflicts(predictions: List[Dict]) -> List[Dict]:
    """
    Resolve conflicts where same match appears multiple times
    Keep only the highest weighted prediction per match
    """
    
    # Group by match
    matches = {}
    for pred in predictions:
        match_id = pred['match_id']
        if match_id not in matches:
            matches[match_id] = []
        matches[match_id].append(pred)
    
    # Keep best prediction per match
    resolved = []
    for match_id, preds in matches.items():
        if len(preds) == 1:
            resolved.append(preds[0])
        else:
            # Keep highest weighted score
            best = max(preds, key=lambda x: x['weighted_score'])
            resolved.append(best)
    
    return resolved


def generate_weighted_top50(csv_path: Path, output_html: Path = None, output_csv: Path = None):
    """
    Generate weighted Top 50 predictions with conflict resolution
    """
    
    print("\nðŸ† GENERATING WEIGHTED TOP 50")
    print("="*45)
    
    if output_html is None:
        output_html = OUTPUT_DIR / "top50_weighted_lite.html"
    if output_csv is None:
        output_csv = OUTPUT_DIR / "top50_weighted_lite.csv"
    
    # Calculate market weights
    weights = calculate_market_weights()
    
    # Extract predictions
    predictions = extract_predictions_from_csv(csv_path)
    print(f"ðŸ“Š Extracted {len(predictions)} predictions")
    
    # Add weighted scores
    for pred in predictions:
        market = pred['market']
        weight = weights.get(market, 0.5)  # Default weight if market not found
        pred['weight'] = weight
        pred['weighted_score'] = pred['probability'] * weight
    
    # Resolve conflicts (keep best prediction per match)
    predictions = resolve_conflicts(predictions)
    print(f"âœ… Resolved conflicts, {len(predictions)} unique matches")
    
    # Sort by weighted score
    predictions.sort(key=lambda x: x['weighted_score'], reverse=True)
    
    # Take top 50
    top50 = predictions[:50]
    
    # Generate HTML
    _generate_html(top50, output_html, weights)
    
    # Generate CSV
    df = pd.DataFrame(top50)
    df.to_csv(output_csv, index=False)
    
    print(f"âœ… Generated top50_weighted.html")
    print(f"âœ… Generated top50_weighted.csv")
    
    return top50


def _generate_split_html(top50_ou: List[Dict], top50_btts: List[Dict], 
                         top50_result: List[Dict], output_path: Path, weights: Dict[str, float]):
    """Generate HTML report with split categories"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Top 50 Weighted Predictions (Split)</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2d3748;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .subtitle {{
            text-align: center;
            color: #718096;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0;
        }}
        .tab {{
            padding: 12px 24px;
            cursor: pointer;
            border: none;
            background: #f7fafc;
            color: #4a5568;
            font-size: 1em;
            font-weight: 600;
            border-radius: 8px 8px 0 0;
            transition: all 0.3s;
        }}
        .tab:hover {{
            background: #edf2f7;
        }}
        .tab.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
            animation: fadeIn 0.3s;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .category-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .category-header h2 {{
            margin: 0;
            font-size: 1.8em;
        }}
        .category-header .count {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 8px;
        }}
        thead th {{
            background: #2d3748;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        thead th:first-child {{ border-radius: 8px 0 0 8px; }}
        thead th:last-child {{ border-radius: 0 8px 8px 0; }}
        
        tbody tr {{
            background: white;
            transition: all 0.2s;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        tbody tr:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        td {{
            padding: 15px;
            border-top: 1px solid #e2e8f0;
            border-bottom: 1px solid #e2e8f0;
        }}
        td:first-child {{ 
            border-left: 1px solid #e2e8f0;
            border-radius: 8px 0 0 8px;
        }}
        td:last-child {{ 
            border-right: 1px solid #e2e8f0;
            border-radius: 0 8px 8px 0;
        }}
        
        .rank {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.2em;
        }}
        .match {{
            font-weight: 600;
            color: #2d3748;
        }}
        .league {{
            color: #718096;
            font-size: 0.9em;
        }}
        .market {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .market-ou {{ background: #bee3f8; color: #2c5282; }}
        .market-btts {{ background: #c6f6d5; color: #22543d; }}
        .market-1x2 {{ background: #fed7d7; color: #742a2a; }}
        
        .prob {{
            font-weight: bold;
            font-size: 1.1em;
        }}
        .prob-high {{ color: #38a169; }}
        .prob-medium {{ color: #d69e2e; }}
        
        .score {{
            font-weight: 600;
            color: #667eea;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class='container'>
        <h1>âš½ Top 50 Weighted Predictions</h1>
        <div class='subtitle'>Split by Category: Over/Under, BTTS, 1X2</div>
        
        <div class='stats'>
            <div class='stat-card'>
                <div class='stat-label'>ðŸŽ¯ OVER/UNDER</div>
                <div class='stat-value'>{len(top50_ou)}</div>
                <div class='stat-label'>Predictions (excl. 0.5)</div>
            </div>
            <div class='stat-card'>
                <div class='stat-label'>âš½ BTTS</div>
                <div class='stat-value'>{len(top50_btts)}</div>
                <div class='stat-label'>Both Teams Score</div>
            </div>
            <div class='stat-card'>
                <div class='stat-label'>ðŸ† 1X2</div>
                <div class='stat-value'>{len(top50_result)}</div>
                <div class='stat-label'>Match Results</div>
            </div>
        </div>
        
        <div class='tabs'>
            <button class='tab active' onclick='showTab(0)'>ðŸŽ¯ Over/Under ({len(top50_ou)})</button>
            <button class='tab' onclick='showTab(1)'>âš½ BTTS ({len(top50_btts)})</button>
            <button class='tab' onclick='showTab(2)'>ðŸ† 1X2 ({len(top50_result)})</button>
        </div>
        
        <div class='tab-content active' id='tab-0'>
            {_generate_category_table(top50_ou, 'ðŸŽ¯ Over/Under Markets', 'ou')}
        </div>
        
        <div class='tab-content' id='tab-1'>
            {_generate_category_table(top50_btts, 'âš½ Both Teams To Score', 'btts')}
        </div>
        
        <div class='tab-content' id='tab-2'>
            {_generate_category_table(top50_result, 'ðŸ† Match Results (1X2)', '1x2')}
        </div>
    </div>
    
    <script>
        function showTab(index) {{
            // Hide all tabs
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            // Show selected tab
            document.querySelectorAll('.tab')[index].classList.add('active');
            document.getElementById('tab-' + index).classList.add('active');
        }}
    </script>
</body>
</html>
"""
    
    output_path.write_text(html, encoding='utf-8')


def _generate_category_table(predictions: List[Dict], title: str, category: str) -> str:
    """Generate HTML table for a category"""
    
    if not predictions:
        return f"<p style='text-align: center; padding: 40px; color: #718096;'>No predictions in this category</p>"
    
    market_class = f'market-{category}'
    
    rows = ""
    for i, pred in enumerate(predictions, 1):
        prob = pred['probability']
        prob_class = 'prob-high' if prob >= 0.75 else 'prob-medium'
        
        rows += f"""
        <tr>
            <td class='rank'>#{i}</td>
            <td>
                <div class='match'>{pred['home_team']} vs {pred['away_team']}</div>
                <div class='league'>{pred['league']} | {pred['date']}</div>
            </td>
            <td><span class='market {market_class}'>{pred['selection']}</span></td>
            <td class='prob {prob_class}'>{prob:.1%}</td>
            <td class='score'>{pred['weighted_score']:.3f}</td>
        </tr>
        """
    
    table = f"""
    <div class='category-header'>
        <h2>{title}</h2>
        <div class='count'>{len(predictions)} predictions</div>
    </div>
    <table>
        <thead>
            <tr>
                <th width='60'>Rank</th>
                <th>Match</th>
                <th width='200'>Market</th>
                <th width='100'>Probability</th>
                <th width='120'>Weighted Score</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """
    
    return table


def _generate_html(predictions: List[Dict], output_path: Path, weights: Dict[str, float]):
    """Generate HTML report (legacy function, kept for compatibility)"""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Top 50 Weighted Predictions</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            h1 { color: #2c3e50; }
            table { width: 100%; border-collapse: collapse; background: white; }
            th { background: #3498db; color: white; padding: 12px; text-align: left; }
            td { padding: 10px; border-bottom: 1px solid #ddd; }
            tr:hover { background: #f0f0f0; }
            .high { background: #d4edda; font-weight: bold; }
            .medium { background: #fff3cd; }
            .low { background: #f8d7da; }
            .info { background: #e8f4f8; padding: 15px; margin: 20px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>âš½ Top 50 Weighted Predictions</h1>
        <div class="info">
            <strong>Smart Conflict Resolution Active</strong><br>
            Only the best-weighted prediction per match is shown.<br>
            Weights calculated from historical accuracy data.
        </div>
        <table>
            <tr>
                <th>Rank</th>
                <th>Date</th>
                <th>League</th>
                <th>Match</th>
                <th>Market</th>
                <th>Probability</th>
                <th>Odds</th>
                <th>Weight</th>
                <th>Score</th>
            </tr>
    """
    
    for i, pred in enumerate(predictions, 1):
        prob = pred['probability']
        score = pred['weighted_score']
        
        # Color coding
        if score >= 0.8:
            row_class = "high"
        elif score >= 0.6:
            row_class = "medium"
        else:
            row_class = "low"
        
        html += f"""
            <tr class="{row_class}">
                <td>{i}</td>
                <td>{pred['date']}</td>
                <td>{pred['league']}</td>
                <td>{pred['home_team']} vs {pred['away_team']}</td>
                <td>{pred['market']}</td>
                <td>{prob:.1%}</td>
                <td>{pred['odds']:.2f}</td>
                <td>{pred['weight']:.2f}</td>
                <td>{score:.2f}</td>
            </tr>
        """
    
    html += """
        </table>
        <div class="info" style="margin-top: 20px;">
            <strong>Legend:</strong><br>
            ðŸŸ¢ High confidence (score â‰¥ 0.8)<br>
            ðŸŸ¡ Medium confidence (0.6 â‰¤ score < 0.8)<br>
            ðŸ”´ Lower confidence (score < 0.6)
        </div>
    </body>
    </html>
    """
    
    output_path.write_text(html, encoding='utf-8')


if __name__ == "__main__":
    # Test
    csv_path = OUTPUT_DIR / "weekly_bets.csv"
    if csv_path.exists():
        generate_weighted_top50(csv_path)
        print("âœ… Test complete!")
    else:
        print("âŒ weekly_bets.csv not found")
