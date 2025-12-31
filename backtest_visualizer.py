# backtest_visualizer.py
"""
Visualize backtest results with charts and insights
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

class BacktestVisualizer:
    """Create visualizations from backtest results"""
    
    def __init__(self, 
                 summary_path: Path = Path("outputs/backtest_summary.csv"),
                 detailed_path: Path = Path("outputs/backtest_detailed.csv")):
        self.summary_path = summary_path
        self.detailed_path = detailed_path
        
        self.summary_df = None
        self.detailed_df = None
        
        self._load_data()
    
    def _load_data(self):
        """Load backtest results"""
        if self.summary_path.exists():
            self.summary_df = pd.read_csv(self.summary_path, index_col=0)
            print(f"✅ Loaded summary: {len(self.summary_df)} markets")
        
        if self.detailed_path.exists():
            self.detailed_df = pd.read_csv(self.detailed_path)
            self.detailed_df['period_start'] = pd.to_datetime(self.detailed_df['period_start'])
            print(f"✅ Loaded detailed: {len(self.detailed_df)} periods")
    
    def plot_accuracy_by_market(self, save_path: Path = None):
        """Bar chart of accuracy by market"""
        if self.summary_df is None or self.summary_df.empty:
            print("⚠️ No summary data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        markets = self.summary_df.index
        accuracies = self.summary_df['Accuracy'] * 100
        
        bars = ax.bar(markets, accuracies, color='steelblue', alpha=0.7)
        
        # Color bars by profitability
        for i, (market, row) in enumerate(self.summary_df.iterrows()):
            if row['ROI'] > 0:
                bars[i].set_color('green')
            elif row['ROI'] < -5:
                bars[i].set_color('red')
        
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Backtest Accuracy by Market')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roi_by_market(self, save_path: Path = None):
        """Bar chart of ROI by market"""
        if self.summary_df is None or self.summary_df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        markets = self.summary_df.index
        rois = self.summary_df['ROI']
        
        colors = ['green' if x > 0 else 'red' for x in rois]
        ax.bar(markets, rois, color=colors, alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_ylabel('ROI (%)')
        ax.set_title('Return on Investment by Market')
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_accuracy_over_time(self, save_path: Path = None):
        """Line chart showing accuracy trends over time"""
        if self.detailed_df is None or self.detailed_df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        markets = self.detailed_df['market'].unique()
        
        for market in markets:
            market_data = self.detailed_df[self.detailed_df['market'] == market]
            market_data = market_data.sort_values('period_start')
            
            ax.plot(market_data['period_start'], 
                   market_data['accuracy'] * 100,
                   marker='o', label=market, linewidth=2, alpha=0.7)
        
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Trends Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_cumulative_profit(self, save_path: Path = None):
        """Plot cumulative profit over time"""
        if self.detailed_df is None or self.detailed_df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        markets = self.detailed_df['market'].unique()
        
        for market in markets:
            market_data = self.detailed_df[self.detailed_df['market'] == market].copy()
            market_data = market_data.sort_values('period_start')
            
            # Calculate cumulative profit
            market_data['cumulative_profit'] = market_data['profit'].cumsum()
            
            ax.plot(market_data['period_start'], 
                   market_data['cumulative_profit'],
                   marker='o', label=market, linewidth=2, alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Profit (Units)')
        ax.set_title('Cumulative Profit Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_brier_score_comparison(self, save_path: Path = None):
        """Compare calibration across markets"""
        if self.summary_df is None or self.summary_df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        markets = self.summary_df.index
        brier_scores = self.summary_df['Avg_Brier_Score']
        
        colors = ['green' if x < 0.2 else 'orange' if x < 0.3 else 'red' for x in brier_scores]
        ax.bar(markets, brier_scores, color=colors, alpha=0.7)
        
        ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Good (<0.2)')
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Poor (>0.3)')
        ax.set_ylabel('Brier Score')
        ax.set_title('Prediction Calibration by Market (Lower = Better)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_performance_heatmap(self, save_path: Path = None):
        """Heatmap of performance metrics"""
        if self.summary_df is None or self.summary_df.empty:
            return
        
        # Select key metrics
        metrics = ['Accuracy', 'ROI', 'Avg_Brier_Score']
        plot_data = self.summary_df[metrics].copy()
        
        # Normalize for comparison
        plot_data['Accuracy'] = plot_data['Accuracy'] * 100
        plot_data['Avg_Brier_Score'] = (1 - plot_data['Avg_Brier_Score']) * 100  # Invert so higher is better
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(plot_data.T, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=50, ax=ax, cbar_kws={'label': 'Score'})
        
        ax.set_xlabel('Market')
        ax.set_ylabel('Metric')
        ax.set_title('Performance Heatmap (Higher = Better)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_all_charts(self, output_dir: Path = Path("outputs")):
        """Generate all visualization charts"""
        print("\n📊 GENERATING BACKTEST VISUALIZATIONS")
        print("="*50)
        
        output_dir.mkdir(exist_ok=True)
        
        self.plot_accuracy_by_market(output_dir / "backtest_accuracy.png")
        self.plot_roi_by_market(output_dir / "backtest_roi.png")
        self.plot_accuracy_over_time(output_dir / "backtest_accuracy_trends.png")
        self.plot_cumulative_profit(output_dir / "backtest_cumulative_profit.png")
        self.plot_brier_score_comparison(output_dir / "backtest_calibration.png")
        self.plot_performance_heatmap(output_dir / "backtest_heatmap.png")
        
        print("\n✅ All charts generated!")
        print(f"📂 Saved to: {output_dir}/")
    
    def generate_html_report(self, output_path: Path = Path("outputs/backtest_report.html")):
        """Generate interactive HTML report"""
        if self.summary_df is None:
            print("⚠️ No data for report")
            return
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; background: white; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        .positive {{ color: green; font-weight: bold; }}
        .negative {{ color: red; font-weight: bold; }}
        .metric {{ display: inline-block; margin: 10px 20px; padding: 15px; background: #fff; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-label {{ font-size: 14px; color: #666; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>🔬 Backtest Performance Report</h1>
    <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>📊 Overall Summary</h2>
"""
        
        # Overall metrics
        total_matches = self.summary_df['Total_Matches'].sum()
        avg_accuracy = self.summary_df['Accuracy'].mean() * 100
        total_profit = self.summary_df['Total_Profit_Units'].sum()
        avg_roi = self.summary_df['ROI'].mean()
        
        html += f"""
        <div class="metric">
            <div class="metric-label">Total Matches</div>
            <div class="metric-value">{total_matches:.0f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Avg Accuracy</div>
            <div class="metric-value">{avg_accuracy:.1f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Profit</div>
            <div class="metric-value" class="{'positive' if total_profit > 0 else 'negative'}">{total_profit:+.1f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Avg ROI</div>
            <div class="metric-value" class="{'positive' if avg_roi > 0 else 'negative'}">{avg_roi:+.1f}%</div>
        </div>
    </div>
    
    <div class="summary">
        <h2>📈 Market Performance</h2>
        <table>
            <tr>
                <th>Market</th>
                <th>Matches</th>
                <th>Accuracy</th>
                <th>ROI</th>
                <th>Profit</th>
                <th>Brier Score</th>
            </tr>
"""
        
        for market, row in self.summary_df.iterrows():
            profit_class = 'positive' if row['ROI'] > 0 else 'negative'
            html += f"""
            <tr>
                <td><strong>{market}</strong></td>
                <td>{row['Total_Matches']:.0f}</td>
                <td>{row['Accuracy']*100:.1f}%</td>
                <td class="{profit_class}">{row['ROI']:+.1f}%</td>
                <td class="{profit_class}">{row['Total_Profit_Units']:+.1f}</td>
                <td>{row['Avg_Brier_Score']:.3f}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
    
    <div class="summary">
        <h2>📊 Visualizations</h2>
        <img src="backtest_accuracy.png" alt="Accuracy by Market">
        <img src="backtest_roi.png" alt="ROI by Market">
        <img src="backtest_accuracy_trends.png" alt="Accuracy Trends">
        <img src="backtest_cumulative_profit.png" alt="Cumulative Profit">
        <img src="backtest_calibration.png" alt="Calibration">
        <img src="backtest_heatmap.png" alt="Performance Heatmap">
    </div>
    
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"✅ HTML report generated: {output_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    print("📊 Backtest Visualizer")
    print("="*50)
    
    viz = BacktestVisualizer()
    
    if viz.summary_df is not None:
        viz.generate_all_charts()
        viz.generate_html_report()
        
        print("\n✅ Complete! Open backtest_report.html in your browser")
    else:
        print("❌ No backtest results found")
        print("Run 'python run_backtest.py' first")
