#!/usr/bin/env python3
"""
Simple Fixture Downloader for football-data.co.uk
Downloads the fixtures.csv file and saves to outputs/upcoming_fixtures.csv
No API, no scraping - just a simple CSV download
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuration
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
OUTPUT_DIR = Path("")
OUTPUT_FILE = OUTPUT_DIR / "upcoming_fixtures.csv"

def download_upcoming_fixtures():
    """
    Download upcoming fixtures CSV from football-data.co.uk
    
    Returns:
        Path to saved CSV file, or None if failed
    """
    print("="*60)
    print("DOWNLOADING UPCOMING FIXTURES")
    print("="*60)
    print(f"\nSource: {FIXTURES_URL}")
    
    try:
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download CSV
        print("Downloading...")
        response = requests.get(FIXTURES_URL, timeout=30)
        response.raise_for_status()
        
        # Save raw CSV
        OUTPUT_FILE.write_bytes(response.content)
        
        # Load and validate
        df = pd.read_csv(OUTPUT_FILE)
        
        # Check required columns exist
        required_cols = ['Date', 'HomeTeam', 'AwayTeam']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"⚠️ Warning: Missing columns: {missing}")
        
        # Add League column if missing (will be 'Div' in the CSV)
        if 'League' not in df.columns and 'Div' in df.columns:
            df = df.rename(columns={'Div': 'League'})
            df.to_csv(OUTPUT_FILE, index=False)
        
        # Filter to future matches only
        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            today = datetime.now().date()
            future_df = df[df['Date'].dt.date >= today]
            
            if len(future_df) < len(df):
                future_df.to_csv(OUTPUT_FILE, index=False)
                print(f"Filtered to {len(future_df)} future matches (removed {len(df) - len(future_df)} past)")
        except Exception as e:
            print(f"⚠️ Date filtering skipped: {e}")
        
        print(f"\n✅ SUCCESS!")
        print(f"   Saved: {OUTPUT_FILE}")
        print(f"   Matches: {len(df)}")
        
        if 'League' in df.columns or 'Div' in df.columns:
            league_col = 'League' if 'League' in df.columns else 'Div'
            print(f"   Leagues: {df[league_col].unique().tolist()}")
        
        print("="*60)
        
        return OUTPUT_FILE
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Download failed: {e}")
        print("\n💡 Manual fallback:")
        print("   1. Go to: https://www.football-data.co.uk/matches.php")
        print("   2. Download the fixtures CSV")
        print(f"   3. Save as: {OUTPUT_FILE}")
        return None
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = download_upcoming_fixtures()
    
    if result:
        print(f"\n🎯 Ready to use: {result}")
        print("   Run: python RUN_WEEKLY.py")
    else:
        print("\n⚠️ Download failed - use manual fallback")
    
    input("\nPress Enter to close...")