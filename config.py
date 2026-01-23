# config.py - ENHANCED VERSION with API-Football Integration
from pathlib import Path
import os
from datetime import date, datetime

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent

# Use shared data folder at parent level (same as dc_laptop)
DATA_DIR = BASE_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# NEW: Dated output directories
def get_dated_output_dir():
    """Creates a dated folder for outputs (e.g., outputs/2025-10-24)"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    dated_dir = BASE_DIR / "outputs" / date_str
    dated_dir.mkdir(parents=True, exist_ok=True)
    return dated_dir

# Main output directory - NOW DATED!
OUTPUT_DIR = get_dated_output_dir()
MODEL_ARTIFACTS_DIR = BASE_DIR / "models"
MODELS_DIR = MODEL_ARTIFACTS_DIR  # Alias for compatibility

# =============================================================================
# API-FOOTBALL CONFIGURATION (NEW!)
# =============================================================================

API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "0f17fdba78d15a625710f7244a1cc770")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

# League ID Mapping (football-data.co.uk code -> API-Football ID)
API_LEAGUE_MAP = {
    # ENGLAND
    'E0': 39,   # Premier League
    'E1': 40,   # Championship
    'E2': 41,   # League One
    'E3': 42,   # League Two
    'EC': 48,   # EFL Cup (League Cup)
    'FAC': 45,  # FA Cup

    # GERMANY
    'D1': 78,   # Bundesliga
    'D2': 79,   # 2. Bundesliga
    'DFB': 81,  # DFB Pokal

    # SPAIN
    'SP1': 140, # La Liga
    'SP2': 141, # La Liga 2
    'CDR': 143, # Copa del Rey

    # ITALY
    'I1': 135,  # Serie A
    'I2': 136,  # Serie B
    'CIT': 137, # Coppa Italia

    # FRANCE
    'F1': 61,   # Ligue 1
    'F2': 62,   # Ligue 2
    'CDF': 66,  # Coupe de France

    # NETHERLANDS
    'N1': 88,   # Eredivisie
    'KNVB': 90, # KNVB Beker (Dutch Cup)

    # BELGIUM
    'B1': 144,  # Jupiler Pro League
    'BEC': 147, # Belgian Cup

    # PORTUGAL
    'P1': 94,   # Primeira Liga
    'TCP': 96,  # Taça de Portugal

    # SCOTLAND
    'SC0': 179, # Scottish Premiership
    'SC1': 180, # Scottish Championship
    'SFC': 70,  # Scottish FA Cup

    # TURKEY
    'T1': 203,  # Süper Lig
    'TFC': 206, # Turkish Cup

    # ADDITIONAL LEAGUES
    'G1': 197,  # Greece Super League
    'A1': 218,  # Austria Bundesliga
    'SWZ': 207, # Switzerland Super League
    'POL': 106, # Poland Ekstraklasa
    'DEN': 119, # Denmark Superliga
    'NOR': 103, # Norway Eliteserien
    'SWE': 113, # Sweden Allsvenskan
    'CZE': 345, # Czech First League
    'CRO': 210, # Croatia HNL

    # EUROPEAN COMPETITIONS
    'UCL': 2,   # Champions League
    'UEL': 3,   # Europa League
    'UECL': 848,# Conference League
}

# Data source selection
USE_API_FOOTBALL = os.environ.get("USE_API_FOOTBALL", "1") == "1"
FALLBACK_TO_CSV = True  # Use football-data.co.uk if API fails

# =============================================================================
# ORIGINAL CONFIGURATION (PRESERVED)
# =============================================================================

# Legacy football-data.org token (different from API-Football!)
FOOTBALL_DATA_ORG_TOKEN = os.environ.get("FOOTBALL_DATA_ORG_TOKEN", "").strip()

# Ensure base directories exist
for p in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODEL_ARTIFACTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# --- Key File Paths ---
FEATURES_PARQUET = DATA_DIR / "processed" / "features.parquet"
HISTORICAL_PARQUET = PROCESSED_DIR / "historical_matches.parquet"
WEEKLY_OUTPUT_CSV = OUTPUT_DIR / "weekly_bets.csv"
BLEND_WEIGHTS_JSON = MODEL_ARTIFACTS_DIR / "blend_weights.json"

# API-Football Database (shared with dc_laptop)
API_FOOTBALL_DB = DATA_DIR / "football_api.db"

# --- Dates ---
CURRENT_YEAR = date.today().year
CURRENT_MONTH = date.today().month
ACTIVE_SEASON_START = CURRENT_YEAR if CURRENT_MONTH >= 7 else CURRENT_YEAR - 1

SEASON_START_YEAR = int(os.environ.get("FOOTY_SEASON_START_YEAR", 2023))
SEASONS = [f"{str(y)[-2:]}{str(y+1)[-2:]}" for y in range(SEASON_START_YEAR, ACTIVE_SEASON_START + 1)]

def get_current_season() -> int:
    """Get current season start year"""
    now = datetime.now()
    return now.year if now.month >= 7 else now.year - 1

# --- Coverage ---
LEAGUE_CODES = [
    # European Competitions
    "UCL", "UEL", "UECL",
    # England (leagues + cups)
    "E0", "E1", "E2", "E3", "EC", "FAC",
    # Germany (leagues + cup)
    "D1", "D2", "DFB",
    # Spain (leagues + cup)
    "SP1", "SP2", "CDR",
    # Italy (leagues + cup)
    "I1", "I2", "CIT",
    # France (leagues + cup)
    "F1", "F2", "CDF",
    # Netherlands (league + cup)
    "N1", "KNVB",
    # Belgium (league + cup)
    "B1", "BEC",
    # Portugal (league + cup)
    "P1", "TCP",
    # Scotland (leagues + cup)
    "SC0", "SC1", "SFC",
    # Turkey (league + cup)
    "T1", "TFC",
    # Additional leagues
    "G1", "A1", "SWZ", "POL", "DEN", "NOR", "SWE", "CZE", "CRO",
]

# Priority leagues (better data quality, higher volume)
PRIORITY_LEAGUES = ["E0", "E1", "D1", "SP1", "I1", "F1"]

# Domestic cups (knockout tournaments - different prediction dynamics)
DOMESTIC_CUPS = ["EC", "FAC", "DFB", "CDR", "CIT", "CDF", "KNVB", "BEC", "TCP", "SFC", "TFC"]

# European competitions (Champions League, Europa League, Conference League)
EUROPEAN_CUPS = ["UCL", "UEL", "UECL"]

# All cup/knockout competitions
ALL_CUPS = DOMESTIC_CUPS + EUROPEAN_CUPS

# --- Data sources ---
FOOTBALL_DATA_CSV_BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Optional football-data.org API
FOOTBALL_DATA_ORG_BASE = "https://api.football-data.org/v4"

# --- Randomness / Reproducibility ---
RANDOM_SEED = 42
GLOBAL_SEED = RANDOM_SEED

# --- Modeling ---
PRIMARY_TARGETS = ["FTR", "OU25"]
USE_ELO = True
USE_ROLLING_FORM = True  
USE_MARKET_FEATURES = True

# NEW: Enhanced feature flags
USE_XG_FEATURES = os.environ.get("USE_XG_FEATURES", "1") == "1"
USE_ADVANCED_STATS = os.environ.get("USE_ADVANCED_STATS", "1") == "1"
USE_PLAYER_DATA = os.environ.get("USE_PLAYER_DATA", "0") == "1"  # Premium tier
USE_ODDS_COMPARISON = os.environ.get("USE_ODDS_COMPARISON", "1") == "1"

TRAIN_SEASONS_BACK = int(os.environ.get("FOOTY_TRAIN_SEASONS_BACK", 8))

# Tuning configuration
OPTUNA_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "25"))
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "300"))
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"

# Rolling form windows
FORM_WINDOWS = [3, 5, 10, 20]
EWM_SPAN = 10

def season_code(year_start: int) -> str:
    return f"{str(year_start)[-2:]}{str(year_start + 1)[-2:]}"

def log_header(msg: str) -> None:
    bar = "=" * max(20, len(msg) + 4)
    print(f"\n{bar}")
    print(f"  {msg}")
    print(f"{bar}")

def log_step(step_num: int, total_steps: int, step_name: str) -> None:
    """Log a pipeline step with progress"""
    pct = int((step_num / total_steps) * 100)
    print(f"\n{'='*60}")
    print(f"STEP {step_num}/{total_steps} ({pct}%): {step_name}")
    print('='*60)

# Email configuration for Outlook
EMAIL_SMTP_SERVER = os.environ.get("EMAIL_SMTP_SERVER", "smtp-mail.outlook.com")
EMAIL_SMTP_PORT = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT", "")

# Show configuration on import
print(f"[DATA] Output directory: {OUTPUT_DIR}")
if USE_API_FOOTBALL:
    print(f"[DATA] Data source: API-Football (enhanced stats)")
else:
    print(f"[DATA] Data source: football-data.co.uk (basic)")
