# betting_accuracy_analyzer.py
"""
Betting Accuracy Analyzer
- Takes 1X2 bets from P_ columns
- Takes O/U markets from DC_ columns
- Filters for >= 90% confidence
- Provides accuracy breakdown by market and league
"""

import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from config import OUTPUT_DIR, BASE_DIR


class BettingAccuracyAnalyzer:
    """
    Analyze betting accuracy using:
    - P_ columns for 1X2 market
    - DC_ columns for O/U markets
    - 90%+ confidence threshold
    """

    def __init__(self, db_path: Path = None, confidence_threshold: float = 0.90):
        if db_path is None:
            db_path = BASE_DIR / "outputs" / "accuracy_database.db"
        self.db_path = db_path
        self.confidence_threshold = confidence_threshold

        # Define which column prefix to use for each market
        self.market_column_config = {
            # 1X2 uses P_ columns (base ML model)
            '1X2': {
                'prefix': 'P_',
                'columns': ['1X2_H', '1X2_D', '1X2_A'],
                'outcome_col': 'y_1X2'
            },
            # O/U markets use DC_ columns (Dixon-Coles ensemble)
            'OU_0_5': {
                'prefix': 'DC_',
                'columns': ['OU_0_5_O', 'OU_0_5_U'],
                'outcome_col': 'y_OU_0_5'
            },
            'OU_1_5': {
                'prefix': 'DC_',
                'columns': ['OU_1_5_O', 'OU_1_5_U'],
                'outcome_col': 'y_OU_1_5'
            },
            'OU_2_5': {
                'prefix': 'DC_',
                'columns': ['OU_2_5_O', 'OU_2_5_U'],
                'outcome_col': 'y_OU_2_5'
            },
            'OU_3_5': {
                'prefix': 'DC_',
                'columns': ['OU_3_5_O', 'OU_3_5_U'],
                'outcome_col': 'y_OU_3_5'
            },
            'OU_4_5': {
                'prefix': 'DC_',
                'columns': ['OU_4_5_O', 'OU_4_5_U'],
                'outcome_col': 'y_OU_4_5'
            },
        }

    def extract_qualifying_bets(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract bets that meet the confidence threshold using the correct column prefixes.

        Returns DataFrame with columns:
            Date, League, HomeTeam, AwayTeam, Market, Prediction, Confidence
        """
        bets = []

        for market, config in self.market_column_config.items():
            prefix = config['prefix']
            columns = config['columns']

            # Build full column names
            full_columns = [f"{prefix}{col}" for col in columns]

            # Check which columns exist
            available_cols = [col for col in full_columns if col in predictions_df.columns]

            if not available_cols:
                continue

            # For each match, find the best prediction for this market
            for idx, row in predictions_df.iterrows():
                best_prob = 0
                best_outcome = None

                for col in available_cols:
                    prob = row.get(col, 0)
                    if pd.notna(prob) and prob > best_prob:
                        best_prob = prob
                        # Extract outcome (last part after underscore)
                        best_outcome = col.split('_')[-1]

                # Only include if meets confidence threshold
                if best_prob >= self.confidence_threshold and best_outcome is not None:
                    bets.append({
                        'Date': row.get('Date'),
                        'League': row.get('League'),
                        'HomeTeam': row.get('HomeTeam'),
                        'AwayTeam': row.get('AwayTeam'),
                        'Market': market,
                        'Prediction': best_outcome,
                        'Confidence': best_prob,
                        'Source': prefix.rstrip('_')  # 'P' or 'DC'
                    })

        return pd.DataFrame(bets)

    def get_accuracy_from_database(self,
                                    start_date: str = None,
                                    end_date: str = None) -> pd.DataFrame:
        """
        Get accuracy data from the accuracy database, filtered by our market/column config.

        Returns DataFrame with prediction results.
        """
        if not self.db_path.exists():
            print(f"[WARN] Database not found: {self.db_path}")
            return pd.DataFrame()

        conn = sqlite3.connect(self.db_path)

        # Get markets we care about
        markets = list(self.market_column_config.keys())
        placeholders = ','.join(['?' for _ in markets])

        query = f"""
            SELECT
                match_date,
                league,
                home_team,
                away_team,
                market,
                predicted_outcome,
                predicted_probability,
                actual_outcome,
                correct
            FROM predictions
            WHERE market IN ({placeholders})
            AND predicted_probability >= ?
            AND correct IS NOT NULL
        """

        params = markets + [self.confidence_threshold]

        if start_date:
            query += " AND match_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND match_date <= ?"
            params.append(end_date)

        query += " ORDER BY match_date DESC"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def calculate_accuracy_breakdown(self,
                                      start_date: str = None,
                                      end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate accuracy breakdown by market and league.

        Returns:
            Dictionary with:
            - 'by_market': DataFrame with accuracy by market
            - 'by_league': DataFrame with accuracy by league
            - 'by_market_league': DataFrame with accuracy by market AND league
            - 'summary': Overall summary stats
        """
        df = self.get_accuracy_from_database(start_date, end_date)

        if df.empty:
            print("[WARN] No data found for accuracy analysis")
            return {
                'by_market': pd.DataFrame(),
                'by_league': pd.DataFrame(),
                'by_market_league': pd.DataFrame(),
                'summary': {}
            }

        # Overall summary
        total = len(df)
        correct = df['correct'].sum()
        accuracy = correct / total if total > 0 else 0
        avg_confidence = df['predicted_probability'].mean()

        summary = {
            'total_bets': total,
            'correct': int(correct),
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'profit_units': correct - total,  # Simple flat stake profit
            'roi': ((correct - total) / total * 100) if total > 0 else 0
        }

        # By Market
        by_market = df.groupby('market').agg({
            'correct': ['sum', 'count'],
            'predicted_probability': 'mean'
        }).reset_index()
        by_market.columns = ['Market', 'Correct', 'Total', 'Avg_Confidence']
        by_market['Accuracy'] = (by_market['Correct'] / by_market['Total'] * 100).round(1)
        by_market['Profit'] = by_market['Correct'] - by_market['Total']
        by_market['ROI'] = ((by_market['Profit'] / by_market['Total']) * 100).round(1)
        by_market = by_market.sort_values('Accuracy', ascending=False)

        # By League
        by_league = df.groupby('league').agg({
            'correct': ['sum', 'count'],
            'predicted_probability': 'mean'
        }).reset_index()
        by_league.columns = ['League', 'Correct', 'Total', 'Avg_Confidence']
        by_league['Accuracy'] = (by_league['Correct'] / by_league['Total'] * 100).round(1)
        by_league['Profit'] = by_league['Correct'] - by_league['Total']
        by_league['ROI'] = ((by_league['Profit'] / by_league['Total']) * 100).round(1)
        by_league = by_league.sort_values('Accuracy', ascending=False)

        # By Market AND League (detailed breakdown)
        by_market_league = df.groupby(['market', 'league']).agg({
            'correct': ['sum', 'count'],
            'predicted_probability': 'mean'
        }).reset_index()
        by_market_league.columns = ['Market', 'League', 'Correct', 'Total', 'Avg_Confidence']
        by_market_league['Accuracy'] = (by_market_league['Correct'] / by_market_league['Total'] * 100).round(1)
        by_market_league['Profit'] = by_market_league['Correct'] - by_market_league['Total']
        by_market_league['ROI'] = ((by_market_league['Profit'] / by_market_league['Total']) * 100).round(1)
        by_market_league = by_market_league.sort_values(['Market', 'Accuracy'], ascending=[True, False])

        return {
            'by_market': by_market,
            'by_league': by_league,
            'by_market_league': by_market_league,
            'summary': summary
        }

    def print_accuracy_report(self,
                               start_date: str = None,
                               end_date: str = None):
        """Print a formatted accuracy report to console."""

        results = self.calculate_accuracy_breakdown(start_date, end_date)

        if results['by_market'].empty:
            print("\n[INFO] No accuracy data available yet.")
            print("       Run predictions and update results first.")
            return results

        summary = results['summary']

        print("\n" + "=" * 70)
        print(" BETTING ACCURACY ANALYSIS")
        print(" P_ columns for 1X2 | DC_ columns for O/U | >= 90% confidence")
        print("=" * 70)

        if start_date or end_date:
            print(f" Period: {start_date or 'start'} to {end_date or 'now'}")

        # Overall Summary
        print(f"\n OVERALL SUMMARY")
        print("-" * 40)
        print(f" Total Bets:      {summary['total_bets']}")
        print(f" Correct:         {summary['correct']}")
        print(f" Accuracy:        {summary['accuracy']:.1%}")
        print(f" Avg Confidence:  {summary['avg_confidence']:.1%}")
        print(f" Profit (units):  {summary['profit_units']:+.0f}")
        print(f" ROI:             {summary['roi']:+.1f}%")

        # By Market
        print(f"\n ACCURACY BY MARKET")
        print("-" * 70)
        print(f"{'Market':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'ROI':>10} {'Source':>8}")
        print("-" * 70)

        for _, row in results['by_market'].iterrows():
            market = row['Market']
            source = self.market_column_config.get(market, {}).get('prefix', 'P_').rstrip('_')
            print(f"{market:<12} {int(row['Correct']):>8} {int(row['Total']):>8} "
                  f"{row['Accuracy']:>9.1f}% {row['ROI']:>+9.1f}% {source:>8}")

        # By League
        print(f"\n ACCURACY BY LEAGUE")
        print("-" * 70)
        print(f"{'League':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'ROI':>10}")
        print("-" * 70)

        for _, row in results['by_league'].iterrows():
            print(f"{row['League']:<12} {int(row['Correct']):>8} {int(row['Total']):>8} "
                  f"{row['Accuracy']:>9.1f}% {row['ROI']:>+9.1f}%")

        # Top Market/League Combinations
        print(f"\n TOP PERFORMING MARKET/LEAGUE COMBINATIONS (min 5 bets)")
        print("-" * 70)

        top_combos = results['by_market_league'][results['by_market_league']['Total'] >= 5]
        top_combos = top_combos.nlargest(10, 'Accuracy')

        print(f"{'Market':<12} {'League':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
        print("-" * 70)

        for _, row in top_combos.iterrows():
            print(f"{row['Market']:<12} {row['League']:<10} {int(row['Correct']):>8} "
                  f"{int(row['Total']):>8} {row['Accuracy']:>9.1f}%")

        # Worst Performing (to avoid)
        print(f"\n WORST PERFORMING MARKET/LEAGUE COMBINATIONS (min 5 bets)")
        print("-" * 70)

        worst_combos = results['by_market_league'][results['by_market_league']['Total'] >= 5]
        worst_combos = worst_combos.nsmallest(5, 'Accuracy')

        print(f"{'Market':<12} {'League':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
        print("-" * 70)

        for _, row in worst_combos.iterrows():
            print(f"{row['Market']:<12} {row['League']:<10} {int(row['Correct']):>8} "
                  f"{int(row['Total']):>8} {row['Accuracy']:>9.1f}%")

        print("\n" + "=" * 70)

        return results

    def export_report(self,
                      output_dir: Path = None,
                      start_date: str = None,
                      end_date: str = None) -> List[Path]:
        """
        Export accuracy report to CSV and HTML files.

        Returns list of created file paths.
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.calculate_accuracy_breakdown(start_date, end_date)

        if results['by_market'].empty:
            print("[WARN] No data to export")
            return []

        created_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')

        # Export CSVs
        csv_path = output_dir / f"betting_accuracy_by_market_{timestamp}.csv"
        results['by_market'].to_csv(csv_path, index=False)
        created_files.append(csv_path)

        csv_path = output_dir / f"betting_accuracy_by_league_{timestamp}.csv"
        results['by_league'].to_csv(csv_path, index=False)
        created_files.append(csv_path)

        csv_path = output_dir / f"betting_accuracy_detailed_{timestamp}.csv"
        results['by_market_league'].to_csv(csv_path, index=False)
        created_files.append(csv_path)

        # Generate HTML Report
        html_path = output_dir / f"betting_accuracy_report_{timestamp}.html"
        self._generate_html_report(results, html_path, start_date, end_date)
        created_files.append(html_path)

        print(f"\n[OK] Exported {len(created_files)} files to {output_dir}")
        for f in created_files:
            print(f"     - {f.name}")

        return created_files

    def _generate_html_report(self,
                               results: Dict,
                               output_path: Path,
                               start_date: str = None,
                               end_date: str = None):
        """Generate HTML report."""

        summary = results['summary']
        by_market = results['by_market']
        by_league = results['by_league']
        by_market_league = results['by_market_league']

        period_str = ""
        if start_date or end_date:
            period_str = f"Period: {start_date or 'start'} to {end_date or 'now'}"

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Betting Accuracy Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 25px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 28px;
        }}
        .header .subtitle {{
            opacity: 0.9;
            font-size: 14px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            text-align: center;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: 700;
            color: #1e3a5f;
        }}
        .stat-card .label {{
            color: #666;
            font-size: 13px;
            margin-top: 5px;
        }}
        .stat-card.positive .value {{ color: #16a34a; }}
        .stat-card.negative .value {{ color: #dc2626; }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }}
        .section h2 {{
            margin: 0 0 20px 0;
            color: #1e3a5f;
            font-size: 18px;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #f1f5f9;
            color: #1e3a5f;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
            font-size: 14px;
        }}
        tr:hover {{
            background: #f8fafc;
        }}
        .accuracy-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 13px;
        }}
        .accuracy-high {{ background: #dcfce7; color: #16a34a; }}
        .accuracy-medium {{ background: #fef3c7; color: #d97706; }}
        .accuracy-low {{ background: #fee2e2; color: #dc2626; }}
        .roi-positive {{ color: #16a34a; font-weight: 600; }}
        .roi-negative {{ color: #dc2626; font-weight: 600; }}
        .source-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .source-p {{ background: #dbeafe; color: #2563eb; }}
        .source-dc {{ background: #f3e8ff; color: #9333ea; }}
        .config-note {{
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 13px;
            color: #0369a1;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Betting Accuracy Analysis</h1>
        <div class="subtitle">
            P_ columns for 1X2 | DC_ columns for O/U | Confidence >= {self.confidence_threshold:.0%}
            {f'<br>{period_str}' if period_str else ''}
            <br>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>

    <div class="config-note">
        <strong>Column Configuration:</strong>
        1X2 market uses <span class="source-badge source-p">P_</span> (base ML model) |
        O/U markets use <span class="source-badge source-dc">DC_</span> (Dixon-Coles ensemble)
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="value">{summary['total_bets']}</div>
            <div class="label">Total Bets</div>
        </div>
        <div class="stat-card">
            <div class="value">{summary['correct']}</div>
            <div class="label">Correct</div>
        </div>
        <div class="stat-card {'positive' if summary['accuracy'] >= 0.55 else 'negative' if summary['accuracy'] < 0.50 else ''}">
            <div class="value">{summary['accuracy']:.1%}</div>
            <div class="label">Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="value">{summary['avg_confidence']:.1%}</div>
            <div class="label">Avg Confidence</div>
        </div>
        <div class="stat-card {'positive' if summary['profit_units'] >= 0 else 'negative'}">
            <div class="value">{summary['profit_units']:+.0f}</div>
            <div class="label">Profit (units)</div>
        </div>
        <div class="stat-card {'positive' if summary['roi'] >= 0 else 'negative'}">
            <div class="value">{summary['roi']:+.1f}%</div>
            <div class="label">ROI</div>
        </div>
    </div>

    <div class="section">
        <h2>Accuracy by Market</h2>
        <table>
            <thead>
                <tr>
                    <th>Market</th>
                    <th>Source</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                    <th>Avg Conf</th>
                    <th>ROI</th>
                </tr>
            </thead>
            <tbody>
"""

        for _, row in by_market.iterrows():
            market = row['Market']
            source = self.market_column_config.get(market, {}).get('prefix', 'P_').rstrip('_')
            source_class = 'source-p' if source == 'P' else 'source-dc'

            acc = row['Accuracy']
            acc_class = 'accuracy-high' if acc >= 55 else 'accuracy-low' if acc < 50 else 'accuracy-medium'
            roi_class = 'roi-positive' if row['ROI'] >= 0 else 'roi-negative'

            html += f"""
                <tr>
                    <td><strong>{market}</strong></td>
                    <td><span class="source-badge {source_class}">{source}_</span></td>
                    <td>{int(row['Correct'])}</td>
                    <td>{int(row['Total'])}</td>
                    <td><span class="accuracy-badge {acc_class}">{acc:.1f}%</span></td>
                    <td>{row['Avg_Confidence']:.1%}</td>
                    <td class="{roi_class}">{row['ROI']:+.1f}%</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Accuracy by League</h2>
        <table>
            <thead>
                <tr>
                    <th>League</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                    <th>Avg Conf</th>
                    <th>ROI</th>
                </tr>
            </thead>
            <tbody>
"""

        for _, row in by_league.iterrows():
            acc = row['Accuracy']
            acc_class = 'accuracy-high' if acc >= 55 else 'accuracy-low' if acc < 50 else 'accuracy-medium'
            roi_class = 'roi-positive' if row['ROI'] >= 0 else 'roi-negative'

            html += f"""
                <tr>
                    <td><strong>{row['League']}</strong></td>
                    <td>{int(row['Correct'])}</td>
                    <td>{int(row['Total'])}</td>
                    <td><span class="accuracy-badge {acc_class}">{acc:.1f}%</span></td>
                    <td>{row['Avg_Confidence']:.1%}</td>
                    <td class="{roi_class}">{row['ROI']:+.1f}%</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Detailed Breakdown (Market + League)</h2>
        <table>
            <thead>
                <tr>
                    <th>Market</th>
                    <th>League</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                    <th>ROI</th>
                </tr>
            </thead>
            <tbody>
"""

        for _, row in by_market_league.iterrows():
            acc = row['Accuracy']
            acc_class = 'accuracy-high' if acc >= 55 else 'accuracy-low' if acc < 50 else 'accuracy-medium'
            roi_class = 'roi-positive' if row['ROI'] >= 0 else 'roi-negative'

            html += f"""
                <tr>
                    <td><strong>{row['Market']}</strong></td>
                    <td>{row['League']}</td>
                    <td>{int(row['Correct'])}</td>
                    <td>{int(row['Total'])}</td>
                    <td><span class="accuracy-badge {acc_class}">{acc:.1f}%</span></td>
                    <td class="{roi_class}">{row['ROI']:+.1f}%</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        output_path.write_text(html, encoding='utf-8')
        print(f"[OK] HTML report: {output_path.name}")


def analyze_betting_accuracy(confidence_threshold: float = 0.90,
                              start_date: str = None,
                              end_date: str = None,
                              export: bool = True) -> Dict:
    """
    Main function to analyze betting accuracy.

    Args:
        confidence_threshold: Minimum confidence to include (default 0.90)
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        export: Whether to export reports to files

    Returns:
        Dictionary with accuracy breakdown results
    """
    analyzer = BettingAccuracyAnalyzer(confidence_threshold=confidence_threshold)

    # Print console report
    results = analyzer.print_accuracy_report(start_date, end_date)

    # Export if requested
    if export and results['by_market'] is not None and not results['by_market'].empty:
        analyzer.export_report(start_date=start_date, end_date=end_date)

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze betting accuracy using P_ for 1X2 and DC_ for O/U markets"
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.90,
        help='Confidence threshold (default: 0.90 = 90%%)'
    )
    parser.add_argument(
        '--start', '-s',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', '-e',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Skip exporting CSV/HTML reports'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" BETTING ACCURACY ANALYZER")
    print(f" Configuration: P_ for 1X2 | DC_ for O/U | >= {args.threshold:.0%} confidence")
    print("=" * 70)

    results = analyze_betting_accuracy(
        confidence_threshold=args.threshold,
        start_date=args.start,
        end_date=args.end,
        export=not args.no_export
    )
