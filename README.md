# Football Prediction System - Ensemble Model

High-accuracy football match prediction system using ensemble machine learning models with API-Football integration.

## Features

- **Ensemble Models**: RandomForest, XGBoost, LightGBM, CatBoost, Dixon-Coles
- **Multiple Markets**: 1X2 (Match Result), BTTS, Over/Under (1.5, 2.5, 3.5, 4.5)
- **API-Football Integration**: Rich match statistics including xG, injuries, detailed statistics
- **Advanced Feature Engineering**: 200+ features including Elo ratings, rolling form, xG metrics
- **Hyperparameter Optimization**: Optuna-based tuning with configurable trial counts
- **Probability Calibration**: Dirichlet and Temperature scaling methods
- **Accumulator Builder**: Generates safe, mixed, and aggressive accumulator strategies
- **O/U Analysis**: Specialized Over/Under prediction analysis
- **Weighted Top 50**: Confidence-weighted best picks

## Architecture

### Shared Data Structure

This system uses a shared data folder with the `dc_laptop` quick prediction system:

```
Chris Code/
├── data/                          ← SHARED DATA
│   ├── football_api.db           ← API-Football database
│   ├── processed/
│   │   ├── features.parquet
│   │   └── historical_matches.parquet
│   └── raw_api/
│
├── all_models/                    ← This repository
│   ├── run_weekly.py             ← Main entry point
│   ├── config.py                 ← Configuration
│   ├── models.py                 ← Model training
│   └── outputs/                  ← Predictions
│
└── dc_laptop/                     ← Quick DC-only system
    └── runner.py                 ← Downloads/updates shared DB
```

### Data Flow

1. **Initial Setup**: Run `dc_laptop/runner.py` to download API-Football data to shared database
2. **Weekly Predictions**: Run `all_models/run_weekly.py` to train ensemble and generate predictions
3. **Database**: Both systems read from `../data/football_api.db` (no duplicate downloads)

## Installation

### Prerequisites

- Python 3.8+
- API-Football key (set in environment variable or config.py)

### Setup

```bash
# Clone repository
git clone https://github.com/CBurns1885/all-models.git
cd all-models

# Install dependencies
pip install -r requirements.txt

# Set API key (optional - already configured)
export API_FOOTBALL_KEY="your_key_here"
```

## Usage

### Quick Start

```bash
python run_weekly.py
```

The script will:
1. Download upcoming fixtures (via API-Football or fallback)
2. Build historical database from shared API-Football DB
3. Generate features (200+ per match)
4. Train ensemble models (or load if recently trained)
5. Generate predictions
6. Create analysis reports (Top 50, O/U, Accumulators)

### Training Modes

When running `run_weekly.py`, you can choose:

1. **Full tuning (50 trials)** - Best accuracy, ~2+ hours
2. **Quick tuning (25 trials)** - Good accuracy, ~1 hour *(default)*
3. **Fast mode (10 trials)** - Decent accuracy, ~30 min
4. **No tuning** - Fastest, ~15 min (for testing)

### Configuration

Edit `config.py` to customize:

```python
# Training period
TRAINING_START_YEAR = 2023

# Leagues to include
LEAGUE_CODES = ["E0", "E1", "D1", "SP1", "I1", "F1", ...]

# Feature flags
USE_XG_FEATURES = True
USE_ADVANCED_STATS = True
USE_ODDS_COMPARISON = True

# Model tuning
OPTUNA_TRIALS = 25
N_ESTIMATORS = 300
```

## Outputs

All outputs are saved to dated folders: `outputs/YYYY-MM-DD/`

### Main Files

- **weekly_bets.csv** - All predictions with probabilities
- **top50_weighted.html** - Top picks weighted by confidence
- **top50_weighted.csv** - Top picks (CSV format)

### Specialized Reports

- **ou_analysis.html** - Over/Under analysis with confidence levels
- **ou_analysis.csv** - O/U predictions (CSV format)
- **accumulators_safe.html** - Conservative 4-fold accumulators
- **accumulators_mixed.html** - Balanced 5-fold accumulators
- **accumulators_aggressive.html** - High-risk 6-fold accumulators

## System Components

### Core Modules

- **run_weekly.py** - Main pipeline orchestrator (11 steps)
- **config.py** - Central configuration
- **data_ingest.py** - Historical data builder (uses shared DB)
- **api_football_adapter.py** - SQLite database adapter
- **features.py** - Feature engineering (200+ features)
- **models.py** - Model training and ensemble
- **predict.py** - Prediction generation

### Model Types

- **model_binary.py** - BTTS predictions
- **model_multiclass.py** - 1X2 (Home/Draw/Away) predictions
- **model_ordinal.py** - Over/Under predictions (CORAL method)

### Utilities

- **tuning.py** - Optuna hyperparameter optimization
- **calibration.py** - Probability calibration (Dirichlet, Temperature)
- **progress_utils.py** - Progress tracking and timing
- **weighted_top50.py** - Confidence-weighted bet selection
- **ou_analyzer.py** - Over/Under market analysis
- **acc_builder.py** - Accumulator strategy builder
- **accuracy_tracker.py** - Historical accuracy tracking

### Dixon-Coles Integration

- **dc_predict.py** - Dixon-Coles Poisson model
- **Supports**: 1X2, BTTS, O/U markets
- **Features**: Time decay, home advantage

## API-Football Integration

### Database Schema

The shared `football_api.db` contains:

- **fixtures** - Match results with xG, statistics, referee, venue
- **injuries** - Player injury tracking
- **league_configs** - League metadata

### Features from API

- Expected Goals (xG) for home/away teams
- Detailed match statistics (shots, possession, cards, etc.)
- Injury impact tracking
- Referee statistics
- Venue information

### Rate Limiting

- 7,500 calls/day limit on free tier
- System uses incremental updates (only downloads new fixtures)
- First run: ~2,400 calls (full season download)
- Daily updates: <100 calls

## Feature Engineering

### Rolling Form Features

- Windows: 3, 5, 10, 20 matches
- Exponential weighted moving averages (EWM)
- Home/away split statistics

### Elo Ratings

- Team strength ratings
- League-specific calibration
- Time decay

### Expected Goals (xG)

- Rolling xG averages
- xG differential
- xG over/under performance

### Market Features

- Odds-implied probabilities (when available)
- Historical bookmaker accuracy
- Market efficiency indicators

## Model Training

### Algorithms

1. **RandomForest** - Baseline ensemble, robust to overfitting
2. **XGBoost** - Gradient boosting, high accuracy
3. **LightGBM** - Fast gradient boosting, efficient
4. **CatBoost** - Handles categorical features natively
5. **Dixon-Coles** - Poisson-based football-specific model

### Ensemble Strategy

- **Stacking**: Meta-model combines individual predictions
- **Calibration**: Dirichlet/Temperature scaling for probability adjustment
- **League-specific**: Separate calibration per league type

### Cross-Validation

- Time-series aware splits
- Rolling window validation
- Out-of-sample testing

## Accuracy Tracking

The system automatically logs predictions and tracks accuracy:

```bash
# After matches complete, update results
python update_results.py
```

This updates `accuracy_database.db` with:
- Weekly prediction accuracy
- Market-specific performance
- League-specific performance
- Model contribution analysis

## Workflow Recommendations

### Daily Quick Predictions
```bash
cd dc_laptop
python runner.py
```
Fast Dixon-Coles predictions (~5-10 minutes)

### Weekly Full Ensemble
```bash
cd all_models
python run_weekly.py
```
Complete ensemble with all models (~30-60 minutes)

### Sunday Setup
1. Run full ensemble for the week ahead
2. Review top50_weighted.html for best individual bets
3. Check ou_analysis.html for O/U opportunities
4. Review accumulator strategies

### Post-Match Analysis
1. Run update_results.py to record outcomes
2. Review accuracy reports
3. Adjust calibration if needed

## Troubleshooting

### "FileNotFoundError: football_api.db"
- Run `dc_laptop/runner.py` first to download shared data
- Database will be created at `../data/football_api.db`

### "No fixtures matched"
- Check system date/time
- Verify database has recent fixtures
- Run `dc_laptop/runner.py` to update

### "API Rate Limit Exceeded"
- Normal on first full download
- Progress saved automatically
- Resume next day with same command

### "No trained models found"
- Ensure Step 3 (TRAIN/LOAD MODELS) completed successfully
- Check models/ directory for .pkl files
- Try force retraining: set `FORCE_RETRAIN=1` in config.py

## Configuration Environment Variables

```bash
# API-Football
export API_FOOTBALL_KEY="your_key_here"

# Model settings
export DISABLE_XGB="0"              # Enable XGBoost
export OPTUNA_TRIALS="25"           # Tuning iterations
export N_ESTIMATORS="400"           # Trees per model
export USE_API_FOOTBALL="1"         # Use API-Football data
export USE_XG_FEATURES="1"          # Use expected goals

# Email notifications (optional)
export EMAIL_SMTP_SERVER="smtp-mail.outlook.com"
export EMAIL_SMTP_PORT="587"
export EMAIL_SENDER="your_email@outlook.com"
export EMAIL_PASSWORD="your_password"
export EMAIL_RECIPIENT="recipient@email.com"
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues or questions:
- GitHub Issues: https://github.com/CBurns1885/all-models/issues
- Related Project: [betting-system](https://github.com/CBurns1885/betting-system)

## Related Projects

- **dc_laptop** - Quick Dixon-Coles prediction system (shares same database)
- **betting-system** - Original single-file prediction system

## Changelog

### v1.0.0 (2025-01-01)
- Initial release
- API-Football integration
- Shared database architecture
- Full ensemble model support
- Accumulator builder
- O/U analysis
- Weighted Top 50 reports
