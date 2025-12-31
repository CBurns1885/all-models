# generate_and_load_stats.py
"""
PHASE 3: Stadium/Referee Integration
1. Generate team_statistics.csv & referee_statistics.csv from historical data
2. Load them into prediction pipeline as features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# ============================================================================
# PART 1: GENERATE STATISTICS FROM HISTORICAL DATA
# ============================================================================

class StatsGenerator:
    """Generate team and referee statistics from historical CSVs"""
    
    def __init__(self, data_dir: Path = Path("downloaded_data")):
        self.data_dir = data_dir
        self.df = pd.DataFrame()
        self.team_stats = {}
        self.referee_stats = {}
        self.league_stats = {}
    
    def load_historical_data(self):
        """Load all historical CSV files"""
        print("📂 Loading historical data...")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Standardize league column
                if 'Div' in df.columns and 'League' not in df.columns:
                    df['League'] = df['Div']
                all_data.append(df)
            except Exception as e:
                print(f"   ⚠️ Skipping {csv_file.name}: {e}")
        
        self.df = pd.concat(all_data, ignore_index=True)
        print(f"✅ Loaded {len(self.df)} matches from {len(csv_files)} files")
    
    def calculate_team_stats(self):
        """Calculate per-team statistics"""
        print("📊 Calculating team statistics...")
        
        teams = set(self.df['HomeTeam'].unique()) | set(self.df['AwayTeam'].unique())
        
        for team in teams:
            home_matches = self.df[self.df['HomeTeam'] == team]
            away_matches = self.df[self.df['AwayTeam'] == team]
            
            if len(home_matches) == 0 and len(away_matches) == 0:
                continue
            
            # Home stats
            home_wins = len(home_matches[home_matches['FTR'] == 'H'])
            home_draws = len(home_matches[home_matches['FTR'] == 'D'])
            home_losses = len(home_matches[home_matches['FTR'] == 'A'])
            home_total = len(home_matches)
            
            # Away stats
            away_wins = len(away_matches[away_matches['FTR'] == 'A'])
            away_draws = len(away_matches[away_matches['FTR'] == 'D'])
            away_losses = len(away_matches[away_matches['FTR'] == 'H'])
            away_total = len(away_matches)
            
            self.team_stats[team] = {
                'total_matches': home_total + away_total,
                'home_matches': home_total,
                'away_matches': away_total,
                'home_win_rate': home_wins / home_total if home_total > 0 else 0,
                'away_win_rate': away_wins / away_total if away_total > 0 else 0,
                'home_advantage': ((home_wins / home_total if home_total > 0 else 0) - 
                                 (away_wins / away_total if away_total > 0 else 0)),
                'points_per_game': ((home_wins + away_wins) * 3 + (home_draws + away_draws)) / 
                                  (home_total + away_total) if (home_total + away_total) > 0 else 0,
                'home_goals_per_game': home_matches['FTHG'].mean() if home_total > 0 else 0,
                'away_goals_per_game': away_matches['FTAG'].mean() if away_total > 0 else 0,
                'home_goals_against_per_game': home_matches['FTAG'].mean() if home_total > 0 else 0,
                'away_goals_against_per_game': away_matches['FTHG'].mean() if away_total > 0 else 0
            }
        
        print(f"✅ Calculated stats for {len(self.team_stats)} teams")
    
    def calculate_referee_stats(self):
        """Calculate per-referee statistics"""
        print("🟨 Calculating referee statistics...")
        
        if 'Referee' not in self.df.columns:
            print("⚠️ No Referee column found, skipping")
            return
        
        referees = self.df['Referee'].dropna().unique()
        
        for referee in referees:
            ref_matches = self.df[self.df['Referee'] == referee]
            
            if len(ref_matches) < 5:  # Minimum 5 matches
                continue
            
            total = len(ref_matches)
            home_wins = len(ref_matches[ref_matches['FTR'] == 'H'])
            away_wins = len(ref_matches[ref_matches['FTR'] == 'A'])
            
            # Cards
            total_yellows = 0
            total_reds = 0
            home_yellows = 0
            away_yellows = 0
            
            if 'HY' in ref_matches.columns and 'AY' in ref_matches.columns:
                total_yellows = ref_matches[['HY', 'AY']].sum().sum()
                home_yellows = ref_matches['HY'].sum()
                away_yellows = ref_matches['AY'].sum()
            
            if 'HR' in ref_matches.columns and 'AR' in ref_matches.columns:
                total_reds = ref_matches[['HR', 'AR']].sum().sum()
            
            self.referee_stats[referee] = {
                'total_matches': total,
                'home_win_rate': home_wins / total,
                'away_win_rate': away_wins / total,
                'home_bias': ((home_wins - away_wins) / total) * 100,
                'cards_per_match': (total_yellows + total_reds) / total if total > 0 else 0,
                'yellows_per_match': total_yellows / total if total > 0 else 0,
                'reds_per_match': total_reds / total if total > 0 else 0
            }
        
        print(f"✅ Calculated stats for {len(self.referee_stats)} referees")
    
    def export_to_csv(self, output_dir: Path):
        """Export stats to CSV files"""
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Team stats
        if self.team_stats:
            team_df = pd.DataFrame.from_dict(self.team_stats, orient='index')
            team_df.index.name = 'Team'
            team_path = output_dir / 'team_statistics.csv'
            team_df.to_csv(team_path)
            print(f"✅ Exported: {team_path}")
        
        # Referee stats
        if self.referee_stats:
            ref_df = pd.DataFrame.from_dict(self.referee_stats, orient='index')
            ref_df.index.name = 'Referee'
            ref_path = output_dir / 'referee_statistics.csv'
            ref_df.to_csv(ref_path)
            print(f"✅ Exported: {ref_path}")
    
    def generate_all(self, output_dir: Path):
        """Run full generation pipeline"""
        self.load_historical_data()
        self.calculate_team_stats()
        self.calculate_referee_stats()
        self.export_to_csv(output_dir)
        return self.team_stats, self.referee_stats


# ============================================================================
# PART 2: LOAD STATISTICS INTO FEATURES
# ============================================================================

class StatsLoader:
    """Load pre-generated statistics CSVs"""
    
    def __init__(self, stats_dir: Path = Path("outputs/statistics")):
        self.stats_dir = stats_dir
        self.team_stats = {}
        self.referee_stats = {}
        self._load_stats()
    
    def _load_stats(self):
        """Load team and referee CSVs"""
        team_file = self.stats_dir / "team_statistics.csv"
        ref_file = self.stats_dir / "referee_statistics.csv"
        
        if team_file.exists():
            df = pd.read_csv(team_file, index_col=0)
            self.team_stats = df.to_dict('index')
            print(f"✅ Loaded {len(self.team_stats)} team stats")
        else:
            print(f"⚠️ Team stats not found: {team_file}")
        
        if ref_file.exists():
            df = pd.read_csv(ref_file, index_col=0)
            self.referee_stats = df.to_dict('index')
            print(f"✅ Loaded {len(self.referee_stats)} referee stats")
        else:
            print(f"⚠️ Referee stats not found: {ref_file}")
    
    def get_team_features(self, team_name: str, venue: str) -> Dict[str, float]:
        """
        Get features for a team
        venue: 'home' or 'away'
        """
        if team_name not in self.team_stats:
            # Default values
            return {
                f'{venue}_stat_win_rate': 0.40,
                f'{venue}_stat_goals_pg': 1.3,
                f'{venue}_stat_goals_against': 1.3,
                f'{venue}_stat_advantage': 0.0
            }
        
        stats = self.team_stats[team_name]
        
        if venue == 'home':
            return {
                'home_stat_win_rate': stats.get('home_win_rate', 0.40),
                'home_stat_goals_pg': stats.get('home_goals_per_game', 1.3),
                'home_stat_goals_against': stats.get('home_goals_against_per_game', 1.3),
                'home_stat_advantage': stats.get('home_advantage', 0.0)
            }
        else:
            return {
                'away_stat_win_rate': stats.get('away_win_rate', 0.40),
                'away_stat_goals_pg': stats.get('away_goals_per_game', 1.3),
                'away_stat_goals_against': stats.get('away_goals_against_per_game', 1.3),
                'away_stat_advantage': -stats.get('home_advantage', 0.0)
            }
    
    def get_referee_features(self, referee_name: str) -> Dict[str, float]:
        """Get features for a referee"""
        if pd.isna(referee_name) or referee_name not in self.referee_stats:
            # Default values
            return {
                'ref_cards_per_match': 3.5,
                'ref_yellows_per_match': 3.0,
                'ref_home_bias': 0.0
            }
        
        stats = self.referee_stats[referee_name]
        return {
            'ref_cards_per_match': stats.get('cards_per_match', 3.5),
            'ref_yellows_per_match': stats.get('yellows_per_match', 3.0),
            'ref_home_bias': stats.get('home_bias', 0.0)
        }
    
    def add_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team/referee stats to existing dataframe"""
        df = df.copy()
        
        # Add home team stats
        for idx, row in df.iterrows():
            home_features = self.get_team_features(row.get('HomeTeam', ''), 'home')
            away_features = self.get_team_features(row.get('AwayTeam', ''), 'away')
            ref_features = self.get_referee_features(row.get('Referee', None))
            
            for key, val in {**home_features, **away_features, **ref_features}.items():
                df.at[idx, key] = val
        
        return df


# ============================================================================
# INTEGRATION FUNCTIONS
# ============================================================================

def generate_statistics(data_dir: Path = None,
                       output_dir: Path = Path("outputs/statistics")) -> bool:
    """
    Generate team_statistics.csv and referee_statistics.csv from historical data
    """
    print("\n🏟️ GENERATING TEAM & REFEREE STATISTICS")
    print("="*45)
    
    # Auto-detect data location
    if data_dir is None:
        possible_dirs = [
            Path("downloaded_data"),
            Path("data/raw"),
            Path("data"),
        ]
        for d in possible_dirs:
            if d.exists() and list(d.glob("*.csv")):
                data_dir = d
                print(f"📂 Using data from: {data_dir}")
                break
        
        if data_dir is None:
            print("⚠️ No CSV files found - skipping statistics generation")
            print("   (This is OK for first run)")
            return False
    
    try:
        generator = StatsGenerator(data_dir)
        generator.generate_all(output_dir)
        print("✅ Statistics generation complete\n")
        return True
    except Exception as e:
        print(f"⚠️ Error generating statistics: {e}")
        print("   Continuing without stats features...")
        return False
def add_stats_features(features_df: pd.DataFrame,
                      stats_dir: Path = Path("outputs/statistics")) -> pd.DataFrame:
    """
    Add team/referee stats to existing features dataframe
    Call this in features.py after building base features
    """
    print("📊 Adding team/referee statistics features...")
    loader = StatsLoader(stats_dir)
    enhanced = loader.add_to_dataframe(features_df)
    new_cols = len(enhanced.columns) - len(features_df.columns)
    print(f"✅ Added {new_cols} statistical feature columns")
    return enhanced


def get_stats_feature_names() -> list:
    """Get list of new feature names for model training"""
    return [
        'home_stat_win_rate',
        'home_stat_goals_pg',
        'home_stat_goals_against',
        'home_stat_advantage',
        'away_stat_win_rate',
        'away_stat_goals_pg',
        'away_stat_goals_against',
        'away_stat_advantage',
        'ref_cards_per_match',
        'ref_yellows_per_match',
        'ref_home_bias'
    ]


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example: Generate statistics
    print("Testing statistics generation...\n")
    
    success = generate_statistics()
    
    if success:
        print("\n✅ Statistics generated successfully!")
        print("   Files created in: outputs/statistics/")
        print("   - team_statistics.csv")
        print("   - referee_statistics.csv")
        
        # Test loading
        print("\nTesting loading into features...")
        test_df = pd.DataFrame({
            'HomeTeam': ['Arsenal', 'Man City'],
            'AwayTeam': ['Chelsea', 'Liverpool'],
            'Referee': ['M Oliver', 'A Taylor']
        })
        
        enhanced = add_stats_features(test_df)
        print(f"\n✅ Feature columns: {list(enhanced.columns)}")
    else:
        print("\n❌ Statistics generation failed")
