#!/usr/bin/env python3
"""
Over/Under Analyzer - Integrated with existing pipeline
Extracts and organizes O/U predictions from weekly_bets.csv
Tracks DC vs Blend performance, generates focused O/U report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sqlite3

# ============================================================================
# CONFIGURATION - Uses existing pipeline paths
# ============================================================================

from config import OUTPUT_DIR

# File paths (integrates with existing pipeline)
WEEKLY_BETS_FILE = OUTPUT_DIR / "weekly_bets_full.csv"
ACCURACY_DB = OUTPUT_DIR / "accuracy_database.db"
OU_REPORT_HTML = OUTPUT_DIR / "ou_analysis.html"
OU_REPORT_CSV = OUTPUT_DIR / "ou_analysis.csv"
OU_REPORT_XLSX = OUTPUT_DIR / "ou_analysis.xlsx"

# Thresholds
DEFAULT_CONFIDENCE = 0.90  # 90% minimum confidence
HIGH_CONFIDENCE = 0.92     # 92%+ = high confidence
ELITE_CONFIDENCE = 0.95    # 95%+ = elite

# O/U Lines to analyze
OU_LINES = [ '0_5','1_5', '2_5', '3_5', '4_5']

# ============================================================================
# DATA EXTRACTION
# ============================================================================

def load_weekly_bets_lite() -> pd.DataFrame:
    """Load predictions from weekly_bets_full.csv"""
    if not WEEKLY_BETS_FILE.exists():
        raise FileNotFoundError(f"Weekly bets file not found: {WEEKLY_BETS_FILE}")

    df = pd.read_csv(WEEKLY_BETS_FILE)
    print(f"Loaded {len(df)} matches from {WEEKLY_BETS_FILE.name}")
    return df


def extract_ou_predictions(df: pd.DataFrame, min_confidence: float = DEFAULT_CONFIDENCE) -> pd.DataFrame:
    """
    Extract all O/U predictions from weekly_bets.csv
    
    Returns DataFrame with columns:
    - Match info (League, Date, HomeTeam, AwayTeam)
    - Line (0.5, 1.5, 2.5, etc.)
    - Selection (Over/Under)
    - DC_Probability
    - Blend_Probability (if available)
    - Best_Probability (higher of DC/Blend)
    """
    
    ou_predictions = []
    
    for idx, row in df.iterrows():
        match_info = {
            'League': row.get('League', ''),
            'Date': row.get('Date', ''),
            'HomeTeam': row.get('HomeTeam', ''),
            'AwayTeam': row.get('AwayTeam', ''),
            'Match': f"{row.get('HomeTeam', '')} vs {row.get('AwayTeam', '')}"
        }
        
        # Check each O/U line
        for line in OU_LINES:
            line_display = line.replace('_', '.')
            
            # Over predictions
            dc_over_col = f'DC_OU_{line}_O'
            blend_over_col = f'BLEND_OU_{line}_O'
            
            if dc_over_col in df.columns:
                dc_prob = row.get(dc_over_col, 0)
                blend_prob = row.get(blend_over_col, 0) if blend_over_col in df.columns else 0
                
                if pd.notna(dc_prob) and dc_prob >= min_confidence:
                    pred = match_info.copy()
                    pred['Line'] = line_display
                    pred['Selection'] = 'Over'
                    pred['DC_Prob'] = dc_prob
                    pred['Blend_Prob'] = blend_prob if pd.notna(blend_prob) else dc_prob
                    pred['Best_Prob'] = max(dc_prob, blend_prob if pd.notna(blend_prob) else 0)
                    pred['Source'] = 'DC' if dc_prob > blend_prob else 'Blend'
                    ou_predictions.append(pred)
            
            # Under predictions
            dc_under_col = f'DC_OU_{line}_U'
            blend_under_col = f'BLEND_OU_{line}_U'
            
            if dc_under_col in df.columns:
                dc_prob = row.get(dc_under_col, 0)
                blend_prob = row.get(blend_under_col, 0) if blend_under_col in df.columns else 0
                
                if pd.notna(dc_prob) and dc_prob >= min_confidence:
                    pred = match_info.copy()
                    pred['Line'] = line_display
                    pred['Selection'] = 'Under'
                    pred['DC_Prob'] = dc_prob
                    pred['Blend_Prob'] = blend_prob if pd.notna(blend_prob) else dc_prob
                    pred['Best_Prob'] = max(dc_prob, blend_prob if pd.notna(blend_prob) else 0)
                    pred['Source'] = 'DC' if dc_prob > blend_prob else 'Blend'
                    ou_predictions.append(pred)
    
    if ou_predictions:
        return pd.DataFrame(ou_predictions)
    else:
        return pd.DataFrame()


# ============================================================================
# HISTORICAL PERFORMANCE ANALYSIS
# ============================================================================

def get_ou_historical_performance() -> dict:
    """
    Query accuracy database for O/U historical performance
    Returns accuracy by line and selection
    """
    
    if not ACCURACY_DB.exists():
        print("No accuracy database found - skipping historical stats")
        return {}
    
    try:
        conn = sqlite3.connect(ACCURACY_DB)
        
        # Query O/U accuracy by market
        query = """
        SELECT 
            market,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
            AVG(CASE WHEN correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
            AVG(predicted_probability) as avg_confidence
        FROM predictions
        WHERE market LIKE '%OU_%'
        AND actual_outcome IS NOT NULL
        GROUP BY market
        ORDER BY accuracy DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Parse into dict
        performance = {}
        for _, row in df.iterrows():
            market = row['market']
            performance[market] = {
                'total': row['total_predictions'],
                'correct': row['correct_predictions'],
                'accuracy': row['accuracy'],
                'avg_confidence': row['avg_confidence']
            }
        
        return performance
    
    except Exception as e:
        print(f"Warning: Could not load historical performance: {e}")
        return {}


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_ou_report(df_ou: pd.DataFrame, min_confidence: float = DEFAULT_CONFIDENCE):
    """Generate comprehensive O/U analysis report"""
    
    if df_ou.empty:
        print(f"\nNo O/U predictions above {min_confidence:.0%} confidence")
        _generate_empty_report(min_confidence)
        return
    
    # Sort by confidence
    df_ou = df_ou.sort_values('Best_Prob', ascending=False).reset_index(drop=True)
    
    # Get historical performance
    historical = get_ou_historical_performance()
    
    # Generate outputs
    _generate_html_report(df_ou, historical, min_confidence)
    _generate_csv_report(df_ou)
    _generate_excel_report(df_ou, historical)
    _print_console_summary(df_ou, historical)


def _generate_html_report(df: pd.DataFrame, historical: dict, min_conf: float):
    """Generate interactive HTML report"""
    
    # Calculate stats
    total = len(df)
    high_conf = len(df[df['Best_Prob'] >= HIGH_CONFIDENCE])
    elite_conf = len(df[df['Best_Prob'] >= ELITE_CONFIDENCE])
    avg_conf = df['Best_Prob'].mean()
    max_conf = df['Best_Prob'].max()
    
    # Count by line
    line_counts = df.groupby(['Line', 'Selection']).size().to_dict()
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Over/Under Analysis - {total} Predictions</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 32px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
            margin: 5px 0;
        }}
        .stat-label {{
            color: #666;
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .elite {{
            background-color: #d4edda;
            font-weight: bold;
        }}
        .high {{
            background-color: #fff3cd;
        }}
        .prob {{
            color: #e74c3c;
            font-weight: bold;
            font-size: 16px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            margin-left: 5px;
        }}
        .dc-badge {{
            background-color: #e74c3c;
            color: white;
        }}
        .blend-badge {{
            background-color: #3498db;
            color: white;
        }}
        .over-badge {{
            background-color: #27ae60;
            color: white;
        }}
        .under-badge {{
            background-color: #f39c12;
            color: white;
        }}
        .historical {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .section-title {{
            color: #2c3e50;
            font-size: 24px;
            margin: 30px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
    </style>
</head>
<body>
    <div class='header'>
        <h1>âš½ Over/Under Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p>Minimum Confidence: {min_conf:.0%}</p>
    </div>
    
    <div class='stats-grid'>
        <div class='stat-card'>
            <div class='stat-label'>Total O/U Predictions</div>
            <div class='stat-value'>{total}</div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>Elite Confidence (85%+)</div>
            <div class='stat-value'>{elite_conf}</div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>High Confidence (75%+)</div>
            <div class='stat-value'>{high_conf}</div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>Average Confidence</div>
            <div class='stat-value'>{avg_conf:.1%}</div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>Highest Confidence</div>
            <div class='stat-value'>{max_conf:.1%}</div>
        </div>
    </div>
"""
    
    # Add historical performance if available
    if historical:
        html += """
    <div class='historical'>
        <h3>ðŸ“Š Historical O/U Performance</h3>
        <table style='margin-top: 10px;'>
            <tr>
                <th>Market</th>
                <th>Total Bets</th>
                <th>Accuracy</th>
                <th>Avg Confidence</th>
            </tr>
"""
        for market, stats in sorted(historical.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            if 'OU_' in market:
                html += f"""
            <tr>
                <td>{market}</td>
                <td>{stats['total']}</td>
                <td><strong>{stats['accuracy']:.1%}</strong></td>
                <td>{stats['avg_confidence']:.1%}</td>
            </tr>
"""
        html += """
        </table>
    </div>
"""
    
    # Predictions table
    html += f"""
    <h2 class='section-title'>All O/U Predictions (Ranked by Confidence)</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Date</th>
            <th>League</th>
            <th>Match</th>
            <th>Market</th>
            <th>DC Prob</th>
            <th>Blend Prob</th>
            <th>Best</th>
        </tr>
"""
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        # Determine row class
        row_class = ''
        if row['Best_Prob'] >= ELITE_CONFIDENCE:
            row_class = 'elite'
        elif row['Best_Prob'] >= HIGH_CONFIDENCE:
            row_class = 'high'
        
        # Selection badge
        sel_badge_class = 'over-badge' if row['Selection'] == 'Over' else 'under-badge'
        
        # Source badge
        src_badge_class = 'dc-badge' if row['Source'] == 'DC' else 'blend-badge'
        
        html += f"""
        <tr class='{row_class}'>
            <td><strong>#{i}</strong></td>
            <td>{row['Date']}</td>
            <td>{row['League']}</td>
            <td>{row['Match']}</td>
            <td>
                O/U {row['Line']} 
                <span class='badge {sel_badge_class}'>{row['Selection']}</span>
            </td>
            <td>{row['DC_Prob']:.1%}</td>
            <td>{row['Blend_Prob']:.1%}</td>
            <td>
                <span class='prob'>{row['Best_Prob']:.1%}</span>
                <span class='badge {src_badge_class}'>{row['Source']}</span>
            </td>
        </tr>
"""
    
    html += """
    </table>
    
    <div style='margin-top: 30px; padding: 20px; background: white; border-radius: 8px;'>
        <h3>ðŸ“Œ Legend</h3>
        <p><span class='badge dc-badge'>DC</span> Dixon-Coles model probability</p>
        <p><span class='badge blend-badge'>BLEND</span> Blended ML + DC probability</p>
        <p><span class='badge over-badge'>Over</span> Over line prediction</p>
        <p><span class='badge under-badge'>Under</span> Under line prediction</p>
        <p><strong>Green rows:</strong> Elite confidence (85%+)</p>
        <p><strong>Yellow rows:</strong> High confidence (75-84%)</p>
    </div>
</body>
</html>
"""
    
    OU_REPORT_HTML.write_text(html, encoding='utf-8')
    print(f"\nHTML report: {OU_REPORT_HTML}")


def _generate_csv_report(df: pd.DataFrame):
    """Save CSV report with percentages and sorted by date"""
    df_export = df[[
        'Date', 'League', 'HomeTeam', 'AwayTeam', 
        'Line', 'Selection', 'DC_Prob', 'Blend_Prob', 'Best_Prob', 'Source'
    ]].copy()
    
    # Convert probabilities to percentages
    for col in ['DC_Prob', 'Blend_Prob', 'Best_Prob']:
        if col in df_export.columns:
            df_export[col] = (df_export[col] * 100).round(2)
    
    # Rename columns to show they're percentages
    df_export.rename(columns={
        'DC_Prob': 'DC_Prob_%',
        'Blend_Prob': 'Blend_Prob_%',
        'Best_Prob': 'Best_Prob_%'
    }, inplace=True)
    
    # Sort by date, league, and probability (descending)
    df_export['Date'] = pd.to_datetime(df_export['Date'], errors='coerce')
    df_export = df_export.sort_values(['Date', 'League', 'Best_Prob_%'], ascending=[True, True, False])
    
    df_export.to_csv(OU_REPORT_CSV, index=False)
    print(f"CSV report: {OU_REPORT_CSV}")

    print(f"CSV report: {OU_REPORT_CSV}")


def _generate_excel_report(df: pd.DataFrame, historical: dict):
    """Save Excel report with multiple sheets, percentages, and sorted by date"""
    try:
        # Prepare data with percentages
        df_formatted = df.copy()
        
        # Convert probabilities to percentages
        for col in ['DC_Prob', 'Blend_Prob', 'Best_Prob']:
            if col in df_formatted.columns:
                df_formatted[col] = (df_formatted[col] * 100).round(2)
        
        # Rename columns
        df_formatted.rename(columns={
            'DC_Prob': 'DC_%',
            'Blend_Prob': 'Blend_%',
            'Best_Prob': 'Best_%'
        }, inplace=True)
        
        # Sort by date
        df_formatted['Date'] = pd.to_datetime(df_formatted['Date'], errors='coerce')
        df_formatted = df_formatted.sort_values(['Date', 'League'], ascending=[True, True])
        
        with pd.ExcelWriter(OU_REPORT_XLSX, engine='openpyxl') as writer:
            # Main predictions sheet
            df_formatted.to_excel(writer, sheet_name='All Predictions', index=False)
            
            # Format the main sheet
            worksheet = writer.sheets['All Predictions']
            for col in ['DC_%', 'Blend_%', 'Best_%']:
                col_letter = None
                for idx, cell in enumerate(worksheet[1]):
                    if cell.value == col:
                        col_letter = cell.column_letter
                        break
                if col_letter:
                    for cell in worksheet[col_letter][1:]:
                        cell.number_format = '0.00'
            
            # By line breakdown (also sorted)
            for line in ['0.5', '1.5', '2.5', '3.5', '4.5']:
                line_df = df_formatted[df_formatted['Line'] == line].copy()
                if not line_df.empty:
                    line_df = line_df.sort_values('Date')
                    sheet_name = f'O_U {line}'.replace('.', '_')
                    line_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Historical performance sheet (with percentages)
            if historical:
                hist_df = pd.DataFrame([
                    {
                        'Market': market,
                        'Total_Bets': stats['total'],
                        'Correct': stats['correct'],
                        'Accuracy_%': round(stats['accuracy'] * 100, 2),
                        'Avg_Confidence_%': round(stats['avg_confidence'] * 100, 2)
                    }
                    for market, stats in historical.items()
                    if 'OU_' in market
                ])
                hist_df.to_excel(writer, sheet_name='Historical', index=False)
        
        print(f"Excel report: {OU_REPORT_XLSX}")
    except Exception as e:
        print(f"Warning: Could not create Excel file: {e}")


def _generate_empty_report(min_conf: float):
    """Generate report when no predictions found"""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>O/U Analysis - No Predictions</title>
    <style>
        body {{font-family: Arial; margin: 40px; text-align: center; color: #666;}}
        .empty {{padding: 50px; background: #f5f5f5; border-radius: 10px;}}
    </style>
</head>
<body>
    <div class='empty'>
        <h2>No O/U Predictions Found</h2>
        <p>No Over/Under predictions above {min_conf:.0%} confidence threshold</p>
        <p>Try lowering the threshold or wait for next week's predictions</p>
    </div>
</body>
</html>
"""
    OU_REPORT_HTML.write_text(html, encoding='utf-8')


def _print_console_summary(df: pd.DataFrame, historical: dict):
    """Print summary to console"""
    print("\n" + "="*60)
    print("OVER/UNDER ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nTotal O/U Predictions: {len(df)}")
    print(f"Elite (85%+): {len(df[df['Best_Prob'] >= ELITE_CONFIDENCE])}")
    print(f"High (75%+): {len(df[df['Best_Prob'] >= HIGH_CONFIDENCE])}")
    print(f"Average Confidence: {df['Best_Prob'].mean():.1%}")
    
    print("\nBy Line:")
    for line in ['0.5', '1.5', '2.5', '3.5', '4.5']:
        line_df = df[df['Line'] == line]
        if not line_df.empty:
            over = len(line_df[line_df['Selection'] == 'Over'])
            under = len(line_df[line_df['Selection'] == 'Under'])
            print(f"  O/U {line}: {over} Over, {under} Under")
    
    print("\nBy Source:")
    print(f"  DC Best: {len(df[df['Source'] == 'DC'])}")
    print(f"  Blend Best: {len(df[df['Source'] == 'Blend'])}")
    
    if historical:
        print("\nHistorical O/U Accuracy:")
        ou_hist = {k: v for k, v in historical.items() if 'OU_' in k}
        for market in sorted(ou_hist, key=lambda x: ou_hist[x]['accuracy'], reverse=True)[:5]:
            stats = ou_hist[market]
            print(f"  {market}: {stats['accuracy']:.1%} ({stats['total']} bets)")
    
    print("\n" + "="*60)
    print(f"Reports saved to {OUTPUT_DIR}/")
    print("="*60)


# ============================================================================
# MAIN FUNCTION (integrates with pipeline)
# ============================================================================

def analyze_ou_predictions(min_confidence: float = DEFAULT_CONFIDENCE):
    """
    Main function - analyzes O/U predictions from weekly_bets.csv
    
    Args:
        min_confidence: Minimum confidence threshold (default 65%)
    
    Returns:
        DataFrame of O/U predictions (or empty if none found)
    """
    
    print("\n" + "="*60)
    print("OVER/UNDER ANALYZER")
    print("="*60)
    
    # Load data
    try:
        df_bets = load_weekly_bets_lite()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Run RUN_WEEKLY.py first to generate predictions")
        return pd.DataFrame()
    
    # Extract O/U predictions
    print(f"\nExtracting O/U predictions (min confidence: {min_confidence:.0%})...")
    df_ou = extract_ou_predictions(df_bets, min_confidence)
    
    if df_ou.empty:
        print(f"No O/U predictions found above {min_confidence:.0%}")
        _generate_empty_report(min_confidence)
        return df_ou
    
    print(f"Found {len(df_ou)} O/U predictions")
    
    # Generate reports
    generate_ou_report(df_ou, min_confidence)
    
    return df_ou


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Over/Under predictions")
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.90,
        help='Minimum confidence threshold (default: 0.90)'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    df_ou = analyze_ou_predictions(min_confidence=args.confidence)
    
    if not df_ou.empty:
        print(f"\nâœ… Analysis complete!")
        print(f"ðŸ“Š Open {OU_REPORT_HTML.name} to view results")
    else:
        print("\nâš ï¸ No predictions to analyze")
