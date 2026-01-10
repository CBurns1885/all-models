# run_backtest.py
"""
Easy backtesting interface
Run this to validate your prediction system on historical data
"""

from datetime import datetime, timedelta
from pathlib import Path
from backtest import BacktestEngine
import pandas as pd

# ============================================================================
# BACKTEST CONFIGURATIONS
# ============================================================================

BACKTEST_CONFIGS = {
    'last_3_months': {
        'name': 'Last 3 Months',
        'days_back': 90,
        'window_days': 7,
        'description': 'Quick validation on recent data'
    },
    'last_6_months': {
        'name': 'Last 6 Months', 
        'days_back': 180,
        'window_days': 7,
        'description': 'Standard backtest period'
    },
    'last_season': {
        'name': 'Last Full Season',
        'days_back': 365,
        'window_days': 7,
        'description': 'Full season validation'
    },
    'last_2_seasons': {
        'name': 'Last 2 Seasons',
        'days_back': 730,
        'window_days': 7,
        'description': 'Long-term performance validation'
    },
    'custom': {
        'name': 'Custom Period',
        'start_date': '2024-01-01',  # Set your dates
        'end_date': '2024-12-31',
        'window_days': 7,
        'description': 'User-defined period'
    }
}

# ============================================================================
# INTERACTIVE RUNNER
# ============================================================================

def select_backtest_config():
    """Interactive configuration selection"""
    print("\n🔬 BACKTESTING CONFIGURATION")
    print("="*50)
    print("\nAvailable configurations:")
    
    configs = list(BACKTEST_CONFIGS.keys())
    for i, key in enumerate(configs, 1):
        config = BACKTEST_CONFIGS[key]
        print(f"\n{i}. {config['name']}")
        print(f"   {config['description']}")
        
        if 'days_back' in config:
            end = datetime.now()
            start = end - timedelta(days=config['days_back'])
            print(f"   Period: {start.date()} to {end.date()}")
        elif 'start_date' in config:
            print(f"   Period: {config['start_date']} to {config['end_date']}")
    
    choice = input(f"\nSelect configuration (1-{len(configs)}, default=2): ").strip() or "2"
    
    try:
        selected_key = configs[int(choice) - 1]
        return BACKTEST_CONFIGS[selected_key], selected_key
    except:
        print("Invalid choice, using default (Last 6 Months)")
        return BACKTEST_CONFIGS['last_6_months'], 'last_6_months'


def run_backtest_with_config(config: dict, config_name: str):
    """Run backtest with selected configuration"""
    print(f"\n{'='*60}")
    print(f"🚀 Running: {config['name']}")
    print(f"{'='*60}")
    
    # Calculate dates
    if 'days_back' in config:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config['days_back'])
    else:
        start_date = datetime.strptime(config['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(config['end_date'], '%Y-%m-%d')
    
    # Create engine
    engine = BacktestEngine(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        test_window_days=config['window_days']
    )
    
    # Run backtest
    summary = engine.run_backtest()
    
    # Export results
    engine.export_detailed_results()
    
    # Print key insights
    print_backtest_insights(summary, config_name)
    
    return summary


def print_backtest_insights(summary: pd.DataFrame, config_name: str):
    """Print actionable insights from backtest"""
    if summary.empty:
        return
    
    print("\n" + "="*60)
    print("💡 KEY INSIGHTS")
    print("="*60)
    
    # Best performing market
    best_market = summary['Accuracy'].idxmax()
    best_acc = summary.loc[best_market, 'Accuracy']
    best_roi = summary.loc[best_market, 'ROI']
    
    print(f"\n🏆 Best Market: {best_market}")
    print(f"   * Accuracy: {best_acc:.1%}")
    print(f"   * ROI: {best_roi:+.1f}%")
    print(f"   * Matches: {summary.loc[best_market, 'Total_Matches']:.0f}")
    
    # Worst performing
    worst_market = summary['Accuracy'].idxmin()
    worst_acc = summary.loc[worst_market, 'Accuracy']
    
    print(f"\n[WARN] Weakest Market: {worst_market}")
    print(f"   * Accuracy: {worst_acc:.1%}")
    print(f"   * Needs improvement")
    
    # Overall profitability
    total_profit = summary['Total_Profit_Units'].sum()
    total_matches = summary['Total_Matches'].sum()
    overall_roi = (total_profit / total_matches * 100) if total_matches > 0 else 0
    
    print(f"\n💰 Overall Performance:")
    print(f"   * Total Profit/Loss: {total_profit:+.1f} units")
    print(f"   * Overall ROI: {overall_roi:+.1f}%")
    print(f"   * Status: {'PROFITABLE [OK]' if total_profit > 0 else 'LOSING [ERROR]'}")
    
    # Recommendations
    print(f"\n📋 Recommendations:")
    
    profitable_markets = summary[summary['ROI'] > 0].index.tolist()
    if profitable_markets:
        print(f"   [OK] Focus on: {', '.join(profitable_markets)}")
    
    unprofitable_markets = summary[summary['ROI'] < 0].index.tolist()
    if unprofitable_markets:
        print(f"   [ERROR] Avoid/improve: {', '.join(unprofitable_markets)}")
    
    # Calibration check
    avg_brier = summary['Avg_Brier_Score'].mean()
    if avg_brier < 0.2:
        print(f"   [OK] Well-calibrated predictions (Brier: {avg_brier:.3f})")
    elif avg_brier > 0.3:
        print(f"   [WARN] Poor calibration (Brier: {avg_brier:.3f}) - consider recalibration")
    
    print("\n" + "="*60)


# ============================================================================
# COMPARISON RUNNER
# ============================================================================

def compare_multiple_periods():
    """Compare backtest performance across multiple periods"""
    print("\n📊 MULTI-PERIOD COMPARISON")
    print("="*50)
    
    periods = ['last_3_months', 'last_6_months', 'last_season']
    all_summaries = {}
    
    for period_key in periods:
        config = BACKTEST_CONFIGS[period_key]
        print(f"\n▶ Testing: {config['name']}")
        
        try:
            summary = run_backtest_with_config(config, period_key)
            all_summaries[period_key] = summary
        except Exception as e:
            print(f"   [ERROR] Failed: {e}")
    
    # Generate comparison report
    print("\n" + "="*60)
    print("📈 PERIOD COMPARISON")
    print("="*60)
    
    for period_key, summary in all_summaries.items():
        config = BACKTEST_CONFIGS[period_key]
        print(f"\n{config['name']}:")
        
        if not summary.empty:
            avg_acc = summary['Accuracy'].mean()
            total_roi = summary['ROI'].mean()
            print(f"   * Avg Accuracy: {avg_acc:.1%}")
            print(f"   * Avg ROI: {total_roi:+.1f}%")
        else:
            print("   * No data")


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Main backtest runner"""
    print("="*60)
    print("🔬 FOOTBALL PREDICTION BACKTESTING SYSTEM")
    print("="*60)
    
    print("\nOptions:")
    print("1. Single backtest (choose period)")
    print("2. Compare multiple periods")
    print("3. Custom date range")
    
    choice = input("\nSelect option (1-3, default=1): ").strip() or "1"
    
    if choice == "1":
        config, config_name = select_backtest_config()
        run_backtest_with_config(config, config_name)
        
    elif choice == "2":
        compare_multiple_periods()
        
    elif choice == "3":
        start = input("Start date (YYYY-MM-DD): ")
        end = input("End date (YYYY-MM-DD): ")
        window = input("Window days (default=7): ").strip() or "7"
        
        custom_config = {
            'name': 'Custom Period',
            'start_date': start,
            'end_date': end,
            'window_days': int(window),
            'description': f'{start} to {end}'
        }
        
        run_backtest_with_config(custom_config, 'custom')
    
    print("\n[OK] Backtesting complete!")
    print("📂 Results saved in outputs/")
    print("   * backtest_summary.csv - Overall stats")
    print("   * backtest_detailed.csv - Period-by-period results")


if __name__ == "__main__":
    main()
