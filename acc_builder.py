"""
Current Season Accumulator Builder - Updates Weekly
Uses only this season's data, refreshes automatically each week
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from itertools import combinations
import sqlite3
from datetime import datetime, timedelta

class SeasonalAccumulatorBuilder:
    def __init__(self, 
                 weekly_bets_path: str = 'outputs/weekly_bets_full.csv',
                 accuracy_db_path: str = 'outputs/accuracy_database.db'):
        """
        Initialize with current season data only
        Updates automatically each week when update_results.py runs
        """
        self.df = pd.read_csv(weekly_bets_path)
        self.accuracy_db = accuracy_db_path
        
        # Get current season start date (August 1st of current season)
        today = datetime.now()
        if today.month >= 8:
            self.season_start = datetime(today.year, 8, 1)
        else:
            self.season_start = datetime(today.year - 1, 8, 1)
        
        # Load ONLY current season performance
        self.league_performance = self._load_current_season_performance()
        
        print(f"✅ Loaded {len(self.df)} upcoming fixtures")
        print(f"📅 Season: {self.season_start.date()} onwards")
        print(f"📊 Historical data: {self._get_total_matches()} matches this season")
    
    def _get_total_matches(self) -> int:
        """Get total matches tracked this season"""
        try:
            conn = sqlite3.connect(self.accuracy_db)
            query = """
            SELECT COUNT(*) as total
            FROM weekly_accuracy
            WHERE prediction_date >= ?
            """
            result = pd.read_sql_query(query, conn, params=(self.season_start.strftime('%Y-%m-%d'),))
            conn.close()
            return result['total'].iloc[0]
        except:
            return 0
    
    def _load_current_season_performance(self) -> Dict:
        """Load ONLY current season accuracy by league and market"""
        try:
            conn = sqlite3.connect(self.accuracy_db)
            
            # Query ONLY current season data
            query = """
            SELECT 
                league,
                market_type,
                COUNT(*) as total_bets,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_bets,
                CAST(SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as accuracy,
                AVG(profit_loss) as avg_roi
            FROM weekly_accuracy
            WHERE prediction_date >= ?
            GROUP BY league, market_type
            HAVING COUNT(*) >= 5
            """
            
            results = pd.read_sql_query(
                query, 
                conn, 
                params=(self.season_start.strftime('%Y-%m-%d'),)
            )
            conn.close()
            
            # Build nested dict
            performance = {}
            for _, row in results.iterrows():
                market_type = row['market_type']
                league = row['league']
                
                # Map database market types to our naming
                market_map = {
                    'OU_4_5': 'Under_4_5',
                    'OU_1_5': 'Over_1_5',
                    '1X2': 'Home_Win'
                }
                
                market = market_map.get(market_type, market_type)
                
                if market not in performance:
                    performance[market] = {}
                
                performance[market][league] = {
                    'accuracy': row['accuracy'],
                    'roi': row['avg_roi'] if pd.notna(row['avg_roi']) else 0,
                    'sample_size': row['total_bets']
                }
            
            if performance:
                print(f"✅ Current season data loaded for {len(performance)} markets")
                for market, leagues in performance.items():
                    print(f"   {market}: {len(leagues)} leagues tracked")
            else:
                print("⚠️ No season data yet - using bootstrap values")
                return self._get_bootstrap_performance()
            
            return performance
            
        except Exception as e:
            print(f"⚠️ Database error: {e}")
            print("⚠️ Using bootstrap performance values")
            return self._get_bootstrap_performance()
    
    def _get_bootstrap_performance(self) -> Dict:
        """
        Bootstrap values for early season (first 2-3 weeks)
        Based on typical league patterns, will be replaced by real data
        """
        return {
            'Under_4_5': {
                'E0': {'accuracy': 0.92, 'roi': 0.03, 'sample_size': 5},
                'D1': {'accuracy': 0.94, 'roi': 0.04, 'sample_size': 5},
                'F1': {'accuracy': 0.90, 'roi': 0.02, 'sample_size': 5},
                'E1': {'accuracy': 0.88, 'roi': 0.02, 'sample_size': 5},
            },
            'Over_1_5': {
                'I1': {'accuracy': 0.85, 'roi': 0.03, 'sample_size': 5},
                'I2': {'accuracy': 0.90, 'roi': 0.04, 'sample_size': 5},
                'SC0': {'accuracy': 0.92, 'roi': 0.04, 'sample_size': 5},
                'D2': {'accuracy': 0.88, 'roi': 0.03, 'sample_size': 5},
            },
            'Home_Win': {
                'N1': {'accuracy': 0.80, 'roi': 0.08, 'sample_size': 5},
                'E3': {'accuracy': 0.82, 'roi': 0.08, 'sample_size': 5},
            }
        }
    
    def extract_percentage(self, val):
        """Convert percentage string to float"""
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            return float(val.strip('%')) / 100
        return float(val)
    
    def get_realistic_odds(self, market: str, probability: float) -> float:
        """Realistic market odds"""
        odds_map = {
            'Under_4_5': 1.10,
            'Over_1_5': 1.20,
            'Home_Win': 1.50 if probability >= 0.85 else 1.60
        }
        return odds_map.get(market, 1.50)
    
    def identify_value_bets(self, 
                           confidence_threshold: float = 0.85,
                           min_sample_size: int = 5) -> List[Dict]:
        """
        Identify value bets using current season performance
        Uses P_OU columns and BLEND_1X2 columns from your system
        
        Args:
            confidence_threshold: Minimum model confidence (85%)
            min_sample_size: Minimum historical matches (5)
        """
        value_bets = []
        
        for idx, row in self.df.iterrows():
            league = row['League']
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            date = row['Date']
            
            # Check each market using correct column names
            markets_to_check = [
                ('Under_4_5', 'P_OU_4_5_U'),     # Under 4.5 goals
                ('Over_1_5', 'P_OU_1_5_O'),      # Over 1.5 goals  
                ('Home_Win', 'BLEND_1X2_H')      # Home win
            ]
            
            for market, column in markets_to_check:
                if market not in self.league_performance:
                    continue
                    
                if league not in self.league_performance[market]:
                    continue
                
                hist_data = self.league_performance[market][league]
                
                # Skip if insufficient sample size
                if hist_data['sample_size'] < min_sample_size:
                    continue
                
                model_prob = self.extract_percentage(row.get(column, 0))
                
                if model_prob >= confidence_threshold:
                    odds = self.get_realistic_odds(market, model_prob)
                    ev = (hist_data['accuracy'] * odds - 1) * 100
                    
                    # Only include positive EV bets
                    if ev > 0:
                        value_bets.append({
                            'match': f"{home_team} vs {away_team}",
                            'league': league,
                            'date': date,
                            'market': market,
                            'model_probability': model_prob,
                            'season_accuracy': hist_data['accuracy'],
                            'season_roi': hist_data['roi'],
                            'sample_size': hist_data['sample_size'],
                            'odds': odds,
                            'ev': ev
                        })
        
        # Sort by EV
        value_bets.sort(key=lambda x: x['ev'], reverse=True)
        
        print(f"\n📊 Found {len(value_bets)} value bets this week:")
        for market in ['Under_4_5', 'Over_1_5', 'Home_Win']:
            count = len([b for b in value_bets if b['market'] == market])
            if count > 0:
                print(f"   {market}: {count} bets")
        
        return value_bets
    
    def build_accumulators(self, 
                          strategy: str = 'mixed',
                          max_accumulators: int = 10) -> List[Dict]:
        """
        Build accumulators using current season data
        
        Strategies:
            - 'mixed': 2 Under 4.5 + 2 Over 1.5 + 1 Home Win (5-fold)
            - 'under_45': 6 Under 4.5 from best leagues (6-fold)
            - 'volume': 3 Under 4.5 + 3 Over 1.5 (6-fold)
        """
        value_bets = self.identify_value_bets()
        
        if len(value_bets) == 0:
            print("❌ No value bets found this week")
            return []
        
        if strategy == 'mixed':
            return self._build_mixed(value_bets, max_accumulators)
        elif strategy == 'under_45':
            return self._build_under45(value_bets, max_accumulators)
        elif strategy == 'volume':
            return self._build_volume(value_bets, max_accumulators)
        else:
            print(f"❌ Unknown strategy: {strategy}")
            return []
    
    def _build_mixed(self, value_bets: List[Dict], max_acc: int) -> List[Dict]:
        """Mixed 5-fold: 2U45 + 2O15 + 1HW"""
        under_45 = [b for b in value_bets if b['market'] == 'Under_4_5'][:10]
        over_15 = [b for b in value_bets if b['market'] == 'Over_1_5'][:10]
        home_win = [b for b in value_bets if b['market'] == 'Home_Win'][:5]
        
        accumulators = []
        
        for u45_pair in combinations(under_45, 2):
            for o15_pair in combinations(over_15, 2):
                for hw in home_win:
                    legs = list(u45_pair) + list(o15_pair) + [hw]
                    matches = [leg['match'] for leg in legs]
                    
                    if len(set(matches)) == 5:
                        acc = self._calculate_accumulator(legs)
                        accumulators.append(acc)
                        
                        if len(accumulators) >= max_acc:
                            return sorted(accumulators, key=lambda x: x['ev'], reverse=True)
        
        return sorted(accumulators, key=lambda x: x['ev'], reverse=True)[:max_acc]
    
    def _build_under45(self, value_bets: List[Dict], max_acc: int) -> List[Dict]:
        """Under 4.5 6-fold from top leagues"""
        under_45 = sorted(
            [b for b in value_bets if b['market'] == 'Under_4_5' and b['season_accuracy'] >= 0.90],
            key=lambda x: x['season_accuracy'],
            reverse=True
        )[:15]
        
        accumulators = []
        for combo in combinations(under_45, 6):
            legs = list(combo)
            matches = [leg['match'] for leg in legs]
            
            if len(set(matches)) == 6:
                acc = self._calculate_accumulator(legs)
                accumulators.append(acc)
                
                if len(accumulators) >= max_acc:
                    break
        
        return sorted(accumulators, key=lambda x: x['ev'], reverse=True)[:max_acc]
    
    def _build_volume(self, value_bets: List[Dict], max_acc: int) -> List[Dict]:
        """Volume 6-fold: 3U45 + 3O15"""
        under_45 = [b for b in value_bets if b['market'] == 'Under_4_5'][:12]
        over_15 = [b for b in value_bets if b['market'] == 'Over_1_5'][:12]
        
        accumulators = []
        
        for u45_combo in combinations(under_45, 3):
            for o15_combo in combinations(over_15, 3):
                legs = list(u45_combo) + list(o15_combo)
                matches = [leg['match'] for leg in legs]
                
                if len(set(matches)) == 6:
                    acc = self._calculate_accumulator(legs)
                    accumulators.append(acc)
                    
                    if len(accumulators) >= max_acc:
                        return sorted(accumulators, key=lambda x: x['ev'], reverse=True)
        
        return sorted(accumulators, key=lambda x: x['ev'], reverse=True)[:max_acc]
    
    def _calculate_accumulator(self, legs: List[Dict]) -> Dict:
        """Calculate accumulator statistics"""
        total_odds = np.prod([leg['odds'] for leg in legs])
        win_probability = np.prod([leg['season_accuracy'] for leg in legs])
        ev = (win_probability * total_odds - 1) * 100
        
        return {
            'legs': legs,
            'num_legs': len(legs),
            'total_odds': round(total_odds, 2),
            'win_probability': round(win_probability, 4),
            'ev': round(ev, 2),
            'avg_sample_size': int(np.mean([leg['sample_size'] for leg in legs]))
        }
    
    def generate_weekly_report(self, filename: str = 'weekly_accumulators.html'):
        """Generate HTML report for all three strategies"""
        
        print("\n" + "="*60)
        print("📊 GENERATING WEEKLY ACCUMULATOR REPORT")
        print("="*60)
        
        strategies = {
            'mixed': 'Mixed 5-Fold (Best EV)',
            'under_45': 'Under 4.5 6-Fold (Safest)',
            'volume': 'Volume 6-Fold (Most Bets)'
        }
        
        all_results = {}
        for strategy_key, strategy_name in strategies.items():
            print(f"\n🔄 Building {strategy_name}...")
            accs = self.build_accumulators(strategy_key, max_accumulators=10)
            all_results[strategy_key] = {
                'name': strategy_name,
                'accumulators': accs
            }
            print(f"✅ Found {len(accs)} accumulators")
        
        # Generate HTML
        html = self._generate_html(all_results)
        
        output_path = Path('outputs') / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"\n✅ Report saved: {output_path}")
        return output_path
    
    def _generate_html(self, results: Dict) -> str:
        """Generate HTML report"""
        
        total_accs = sum(len(r['accumulators']) for r in results.values())
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Weekly Accumulators - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        body {{ font-family: Arial; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #2c3e50; text-align: center; }}
        .summary {{ background: #3498db; color: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .strategy {{ margin: 30px 0; border-top: 3px solid #3498db; padding-top: 20px; }}
        .accumulator {{ border: 2px solid #3498db; border-radius: 8px; padding: 15px; margin: 15px 0; background: #f8f9fa; }}
        .stats {{ background: #27ae60; color: white; padding: 10px; border-radius: 5px; display: inline-block; margin: 10px 0; }}
        .leg {{ background: white; padding: 10px; margin: 5px 0; border-left: 4px solid #3498db; border-radius: 4px; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; margin: 0 5px; }}
        .under {{ background: #3498db; color: white; }}
        .over {{ background: #e67e22; color: white; }}
        .home {{ background: #9b59b6; color: white; }}
        .warning {{ background: #f39c12; color: white; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Weekly Accumulators - {datetime.now().strftime('%B %d, %Y')}</h1>
        
        <div class="summary">
            <h3>📊 This Week's Summary</h3>
            <p><strong>Total Accumulators:</strong> {total_accs}</p>
            <p><strong>Season Start:</strong> {self.season_start.strftime('%B %d, %Y')}</p>
            <p><strong>Matches Tracked:</strong> {self._get_total_matches()}</p>
        </div>
        
        <div class="warning">
            ⚠️ <strong>Note:</strong> Accuracies based on current season data only. 
            Early season (first 2-3 weeks) uses bootstrap values until sufficient data is collected.
        </div>
"""
        
        for strategy_key, data in results.items():
            accs = data['accumulators']
            
            html += f"""
        <div class="strategy">
            <h2>{data['name']}</h2>
            <p><strong>{len(accs)} accumulators found</strong></p>
"""
            
            for i, acc in enumerate(accs[:5], 1):  # Show top 5
                html += f"""
            <div class="accumulator">
                <h3>Accumulator #{i}</h3>
                <div class="stats">
                    Odds: {acc['total_odds']} | 
                    Win Rate: {acc['win_probability']*100:.1f}% | 
                    EV: +{acc['ev']:.1f}% | 
                    Samples: {acc['avg_sample_size']}
                </div>
"""
                
                for j, leg in enumerate(acc['legs'], 1):
                    market_class = leg['market'].lower().replace('_', '')
                    html += f"""
                <div class="leg">
                    <strong>{j}. {leg['match']}</strong> ({leg['league']}) - {leg['date']}<br>
                    <span class="badge {market_class}">{leg['market'].replace('_', ' ')}</span>
                    Odds: {leg['odds']} | 
                    Season Accuracy: {leg['season_accuracy']*100:.1f}% ({leg['sample_size']} matches)
                </div>
"""
                
                html += """
            </div>
"""
            
            html += """
        </div>
"""
        
        html += """
    </div>
</body>
</html>"""
        
        return html


# Main execution
if __name__ == "__main__":
    builder = SeasonalAccumulatorBuilder()
    builder.generate_weekly_report('weekly_accumulators_full.html')
    
    print("\n✅ DONE! Open outputs/weekly_accumulators.html to view results")
