# feature_rules.py
"""
Single source of truth for which columns may be used as model features.

models.py, models_goal.py, models_counts.py and features.get_feature_columns()
all import from here so exclusion lists can never drift out of sync.

CRITICAL LEAK NOTE: features.parquet contains the CURRENT match's raw stats
twice — once under the pre-pivot names (HC, AY, HS, ...) and once under the
post-pivot side names (Home_Corners, Away_CardsY, Home_Shots, ...). Both are
outcomes of the match being predicted, not information available beforehand.
models.py historically excluded only the pre-pivot names, so the classifiers
were trained with e.g. the match's actual corner and card counts as features —
which is why cards/corners backtests showed impossible (100%) accuracies.
Only rolling/EWM aggregates of these stats (computed from PRIOR matches) are
legitimate features.
"""

# Identifier / metadata columns — never features
ID_COLS = {
    "League", "Date", "HomeTeam", "AwayTeam", "Season", "Referee", "referee",
    "fixture_id", "Home_ID", "Away_ID", "League_ID", "venue_name", "Time",
    "Season_x", "Season_y",  # merge-collision artifacts
    # league_type ("League"/"Cup") is a raw string that predict.py's future
    # frame cannot reproduce — it is metadata, not a per-team feature, and
    # is_cup/is_european already encode the same information numerically.
    # Leaving it in would make the fitted preprocessor demand a column that
    # never exists at prediction time, and predict_proba would then skip
    # every model with "preprocessor incompatible".
    "league_type",
    # Availability/annotation columns written for the picks page and the
    # injury adjustment — prediction-time only, no historical equivalent.
    "Home_KeyPlayerOut", "Away_KeyPlayerOut",
    "Home_KeyOutNames", "Away_KeyOutNames",
    "Home_MissingAttackShare", "Away_MissingAttackShare",
    "home_injuries", "away_injuries",
    "home_injury_players", "away_injury_players",
    "home_injury_ids", "away_injury_ids",
    "home_injury_types", "away_injury_types",
    "home_formation", "away_formation", "lineups_confirmed",
    "home_team_id", "away_team_id",
}

# Full-time / half-time results — outcomes, never features
RESULT_COLS = {
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HomeGoals", "AwayGoals", "OU25",
}

# Current-match raw stats, pre-pivot names (from the source CSV/DB row)
RAW_MATCH_STATS = {
    "HS", "AS", "HST", "AST", "HC", "AC",
    "HY", "AY", "HR", "AR", "HF", "AF",
}

# Current-match raw stats, post-pivot side names. _pivot_back() joins the
# per-side long frame back as Home_*/Away_* and the raw (non-rolling) stat
# columns come along with the rolling ones. These are the same leak as
# RAW_MATCH_STATS under different names.
_SIDE_STAT_BASES = [
    "Shots", "ShotsT", "Corners", "CardsY", "CardsR", "Fouls",
    "xG", "Possession", "ShotsInBox", "BigChances", "PassAcc",
    "GoalsFor", "GoalsAgainst", "Win", "Draw", "Loss",
    "CleanSheet", "FailedToScore", "BTTS",
    # Player-derived aggregates (player_impact.PLAYER_ONLY_FEATURES). These
    # are THIS match's values — the team's rating, key passes and duels from
    # the game being predicted — so they leak the result exactly like shots
    # or cards would. Only the rolling/EWM versions are legitimate.
    "PlayerRating", "PlayerRatingTop", "KeyPasses", "Tackles",
    "Interceptions", "DuelWinPct", "DribbleSuccess",
]
SIDE_RAW_MATCH_STATS = {f"{side}_{base}" for side in ("Home", "Away")
                        for base in _SIDE_STAT_BASES}

# Also the DB-sourced per-match detail columns (adapter naming)
DB_RAW_MATCH_STATS = {
    f"{side}_{base}" for side in ("Home", "Away")
    for base in (
        "ShotsOnGoal", "ShotsOffGoal", "BlockedShots", "Shots_Inside_Box",
        "ShotsOutsideBox", "Offsides", "GKSaves", "TotalPasses",
        "Pass_Accuracy", "xG_Stats",
    )
}

# Columns known to be blank/degenerate in the API-Football data path
BLANK_COLS = {
    "B365H", "B365D", "B365A", "PSCH", "PSCD", "PSCA",
    "B365_Impl_H", "B365_Impl_D", "B365_Impl_A", "B365_Overround",
    "Home_BigChances", "Away_BigChances",
}


def feature_exclusions() -> set:
    """Every column name that must not be used as a model feature
    (targets, i.e. y_* columns, are excluded by prefix separately)."""
    return (ID_COLS | RESULT_COLS | RAW_MATCH_STATS
            | SIDE_RAW_MATCH_STATS | DB_RAW_MATCH_STATS | BLANK_COLS)


def is_feature_col(col: str) -> bool:
    """True if the column may be used as a model feature."""
    if col.startswith("y_"):
        return False
    return col not in feature_exclusions()
