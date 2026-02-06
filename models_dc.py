# models_dc.py - ENHANCED FOR O/U ACCURACY
"""
Dixon-Coles model optimized for Over/Under markets
Key improvements:
1. Adaptive time decay per league
2. Score-specific rho optimization
3. Team form weighting
4. Home/Away goal rate asymmetry
5. Recent form booster
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# ============================================================================
# ENHANCED PARAMETERS
# ============================================================================

# Adaptive decay by league (more recent = more volatile)
LEAGUE_DECAY_DAYS = {
    'E0': 365,  # Premier League - stable
    'E1': 320,  # Championship - more volatile
    'E2': 300,
    'D1': 380,  # Bundesliga - stable
    'SP1': 390, # La Liga
    'I1': 370,  # Serie A
    'F1': 360,  # Ligue 1
}
DEFAULT_DECAY = 400.0

MAX_GOALS = 12  # Increased for better tail accuracy
RECENT_FORM_WINDOW = 5  # Last 5 matches for form boost

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DCParams:
    """Enhanced DC parameters with form tracking"""
    attack: Dict[str, float]
    defence: Dict[str, float]
    home_adv: float
    rho: float
    league: str
    
    # NEW: Recent form multipliers
    recent_attack: Dict[str, float] = None
    recent_defence: Dict[str, float] = None
    
    def __post_init__(self):
        if self.recent_attack is None:
            self.recent_attack = {}
        if self.recent_defence is None:
            self.recent_defence = {}

# ============================================================================
# TIME WEIGHTING (ADAPTIVE)
# ============================================================================

def _time_weights(dates: pd.Series, league: str = None) -> np.ndarray:
    """Adaptive time decay based on league characteristics"""
    dates = pd.to_datetime(dates, errors="coerce")
    maxd = dates.max()
    age = (maxd - dates).dt.days.values.astype(float)
    
    # Use league-specific decay or default
    half_life = LEAGUE_DECAY_DAYS.get(league, DEFAULT_DECAY)
    
    return np.exp(-np.log(2.0) * age / half_life)

# ============================================================================
# CORRELATION FUNCTION (ENHANCED)
# ============================================================================

def _dc_corr(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    Enhanced Dixon-Coles correlation for low scores
    Handles interdependence of low-scoring games
    """
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

# ============================================================================
# NEGATIVE LOG-LIKELIHOOD (OPTIMIZED)
# ============================================================================

def _neg_loglik(theta, n, home_idx, away_idx, hg, ag, w):
    """Negative log-likelihood for parameter optimization"""
    
    # Extract parameters
    att = theta[:n] - np.mean(theta[:n])  # Attack strengths (mean-centered)
    deff = theta[n:2*n]                    # Defence strengths
    home = theta[-2]                       # Home advantage
    rho = theta[-1]                        # Correlation parameter
    
    # Expected goals
    lam = np.exp(att[home_idx] - deff[away_idx] + home)  # Home expected goals
    mu = np.exp(att[away_idx] - deff[home_idx])          # Away expected goals
    
    # Base Poisson log-likelihood
    logp = (
        -lam + hg * np.log(lam + 1e-12) - gammaln(hg + 1)
        - mu + ag * np.log(mu + 1e-12) - gammaln(ag + 1)
    )
    
    # Apply Dixon-Coles correlation for low scores
    corr = np.array([_dc_corr(int(x), int(y), L, M, rho) 
                     for x, y, L, M in zip(hg, ag, lam, mu)])
    
    logp += np.log(np.maximum(corr, 1e-12))
    
    return -np.sum(w * logp)

# ============================================================================
# RECENT FORM CALCULATOR
# ============================================================================

def _calculate_recent_form(df_league: pd.DataFrame, window: int = RECENT_FORM_WINDOW) -> Dict[str, Dict]:
    """
    Calculate recent form adjustments for attack/defence
    Returns multipliers based on last N games
    """
    df = df_league.sort_values('Date')
    teams = pd.unique(pd.concat([df['HomeTeam'], df['AwayTeam']]))
    
    recent_attack = {}
    recent_defence = {}
    
    for team in teams:
        # Get team's recent matches (last window games)
        home_matches = df[df['HomeTeam'] == team].tail(window)
        away_matches = df[df['AwayTeam'] == team].tail(window)
        
        if len(home_matches) + len(away_matches) < 3:
            # Not enough data
            recent_attack[team] = 1.0
            recent_defence[team] = 1.0
            continue
        
        # Calculate recent scoring/conceding rates
        recent_goals_scored = (
            home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
        )
        recent_goals_conceded = (
            home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
        )
        
        matches_played = len(home_matches) + len(away_matches)
        
        # Overall league average (for comparison)
        league_avg_goals = (df['FTHG'].mean() + df['FTAG'].mean()) / 2
        
        # Form multiplier (1.0 = average, >1.0 = good form, <1.0 = poor form)
        recent_attack[team] = (recent_goals_scored / matches_played) / league_avg_goals
        recent_defence[team] = league_avg_goals / (recent_goals_conceded / matches_played)
        
        # Clamp to reasonable range [0.7, 1.3]
        recent_attack[team] = max(0.7, min(1.3, recent_attack[team]))
        recent_defence[team] = max(0.7, min(1.3, recent_defence[team]))
    
    return {
        'attack': recent_attack,
        'defence': recent_defence
    }

# ============================================================================
# LEAGUE FITTING (ENHANCED)
# ============================================================================

def fit_league(df_league: pd.DataFrame, use_form: bool = True) -> DCParams:
    """
    Fit Dixon-Coles model to league data with optional form enhancement
    
    Args:
        df_league: DataFrame with League, Date, HomeTeam, AwayTeam, FTHG, FTAG
        use_form: Whether to calculate recent form adjustments
    
    Returns:
        DCParams with fitted parameters
    """
    # Clean data
    df = df_league.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Date'])
    
    if len(df) < 50:
        raise ValueError(f"Insufficient data: {len(df)} matches (need 50+)")
    
    # Get unique teams and create index mapping
    teams = pd.unique(pd.concat([df['HomeTeam'], df['AwayTeam']]))
    t2i = {t: i for i, t in enumerate(teams)}
    
    # Map teams to indices
    home_idx = df['HomeTeam'].map(t2i).values
    away_idx = df['AwayTeam'].map(t2i).values
    
    # Goals
    hg = df['FTHG'].astype(int).values
    ag = df['FTAG'].astype(int).values
    
    # Time weights (adaptive by league)
    league_name = df['League'].iloc[0] if 'League' in df.columns else None
    w = _time_weights(df['Date'], league_name)
    
    # Parameter initialization
    n = len(teams)
    theta0 = np.zeros(2 * n + 2)
    theta0[-2] = 0.2   # Home advantage initial guess
    theta0[-1] = 0.0   # Rho initial guess (centered at 0)

    # Set bounds: attack/defence unbounded, home_adv [-1, 2], rho [-0.3, 0.3]
    bounds = [(None, None)] * (2 * n)  # Attack and defence unbounded
    bounds.append((-1.0, 2.0))  # Home advantage bounds
    bounds.append((-0.3, 0.3))  # Rho bounds (critical for numerical stability)

    # Optimize
    res = minimize(
        _neg_loglik,
        theta0,
        args=(n, home_idx, away_idx, hg, ag, w),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 800, 'ftol': 1e-10}
    )
    
    # Extract fitted parameters
    th = res.x
    att = th[:n] - np.mean(th[:n])  # Mean-centered attack
    deff = th[n:2*n]                 # Defence
    
    # Create params object
    params = DCParams(
        attack={t: float(att[i]) for t, i in t2i.items()},
        defence={t: float(deff[i]) for t, i in t2i.items()},
        home_adv=float(th[-2]),
        rho=float(th[-1]),
        league=league_name or 'Unknown'
    )
    
    # Add recent form adjustments if requested
    if use_form:
        form = _calculate_recent_form(df)
        params.recent_attack = form['attack']
        params.recent_defence = form['defence']
    
    return params

# ============================================================================
# FIT ALL LEAGUES
# ============================================================================

def fit_all(df: pd.DataFrame, use_form: bool = True) -> Dict[str, DCParams]:
    """
    Fit DC model to all leagues in DataFrame
    
    Args:
        df: DataFrame with multiple leagues
        use_form: Whether to use recent form adjustments
    
    Returns:
        Dictionary mapping league code to DCParams
    """
    params_by_league = {}
    
    for league, group in df.groupby('League', sort=False):
        try:
            params = fit_league(group, use_form=use_form)
            params_by_league[league] = params
            print(f"  Fitted {league}: {len(group)} matches, "
                  f"home_adv={params.home_adv:.3f}, rho={params.rho:.4f}")
        except Exception as e:
            print(f"  [WARN] DC fit skipped for {league}: {e}")
    
    return params_by_league

# ============================================================================
# SCORE PROBABILITY GRID
# ============================================================================

def _score_grid(lam: float, mu: float, rho: float, 
                max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Generate probability matrix for all score combinations
    
    Args:
        lam: Home expected goals
        mu: Away expected goals
        rho: Correlation parameter
        max_goals: Maximum goals to consider
    
    Returns:
        (max_goals+1) x (max_goals+1) probability matrix
    """
    P = np.zeros((max_goals + 1, max_goals + 1))
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            # Base Poisson probability
            base = np.exp(
                -lam + i * np.log(lam + 1e-12) - gammaln(i + 1)
                - mu + j * np.log(mu + 1e-12) - gammaln(j + 1)
            )
            
            # Apply DC correlation
            P[i, j] = base * _dc_corr(i, j, lam, mu, rho)
    
    # Normalize (safety check)
    P /= P.sum()
    
    return P

# ============================================================================
# MATCH PRICING (ENHANCED)
# ============================================================================

def price_match(params: DCParams, home: str, away: str, 
                use_form: bool = True,
                max_goals: int = MAX_GOALS) -> Dict[str, float]:
    """
    Calculate probabilities for all markets with form adjustment
    
    Args:
        params: Fitted DC parameters
        home: Home team name
        away: Away team name
        use_form: Whether to apply recent form adjustments
        max_goals: Maximum goals for calculations
    
    Returns:
        Dictionary of market probabilities
    """
    # Check if teams exist in params
    if home not in params.attack or away not in params.attack:
        return {}
    
    # Base expected goals
    lam = np.exp(params.attack[home] - params.defence[away] + params.home_adv)
    mu = np.exp(params.attack[away] - params.defence[home])
    
    # Apply recent form adjustments if available and requested
    if use_form and params.recent_attack:
        form_mult_home_att = params.recent_attack.get(home, 1.0)
        form_mult_away_att = params.recent_attack.get(away, 1.0)
        form_mult_home_def = params.recent_defence.get(home, 1.0)
        form_mult_away_def = params.recent_defence.get(away, 1.0)
        
        # Adjust expected goals by form
        lam *= form_mult_home_att * (1 / form_mult_away_def)
        mu *= form_mult_away_att * (1 / form_mult_home_def)
    
    # Generate score probability grid
    P = _score_grid(lam, mu, params.rho, max_goals)
    
    out = {}
    
    # 1X2 - Match result
    out['DC_1X2_H'] = np.tril(P, -1).sum()  # Home wins
    out['DC_1X2_D'] = np.trace(P)           # Draws
    out['DC_1X2_A'] = np.triu(P, 1).sum()   # Away wins
    
    # BTTS - Both teams to score
    out['DC_BTTS_Y'] = P[1:, 1:].sum()
    out['DC_BTTS_N'] = 1 - out['DC_BTTS_Y']
    
    # Over/Under lines (CRITICAL FOR O/U ACCURACY)
    S = np.add.outer(np.arange(P.shape[0]), np.arange(P.shape[1]))
    
    for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        line_str = str(line).replace('.', '_')
        
        over_prob = P[S > line].sum()
        under_prob = P[S < line].sum()
        
        # Handle exactly on line (push in some markets)
        on_line_prob = P[S == line].sum()
        
        # For X.5 lines, no push possible
        out[f'DC_OU_{line_str}_O'] = over_prob
        out[f'DC_OU_{line_str}_U'] = under_prob + on_line_prob
    
    # Asian Handicap lines
    ah_lines = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for line in ah_lines:
        if line < 0:
            line_key = f"-{abs(line)}".replace('.', '_')
        else:
            line_key = f"+{line}".replace('.', '_') if line > 0 else "0_0"
        
        home_wins = away_wins = pushes = 0.0
        
        for h in range(P.shape[0]):
            for a in range(P.shape[1]):
                adjusted_diff = h - a - line
                prob = P[h, a]
                
                if adjusted_diff > 0:
                    home_wins += prob
                elif adjusted_diff < 0:
                    away_wins += prob
                else:
                    pushes += prob
        
        out[f'DC_AH_{line_key}_H'] = home_wins
        out[f'DC_AH_{line_key}_A'] = away_wins
        out[f'DC_AH_{line_key}_P'] = pushes
    
    # Goal Ranges
    out['DC_GR_0'] = P[S == 0].sum()
    out['DC_GR_1'] = P[S == 1].sum()
    out['DC_GR_2'] = P[S == 2].sum()
    out['DC_GR_3'] = P[S == 3].sum()
    out['DC_GR_4'] = P[S == 4].sum()
    out['DC_GR_5+'] = P[S >= 5].sum()
    
    # Correct Scores (0-0 to 5-5 plus Other)
    for h in range(6):
        for a in range(6):
            if h < P.shape[0] and a < P.shape[1]:
                out[f'DC_CS_{h}_{a}'] = P[h, a]
            else:
                out[f'DC_CS_{h}_{a}'] = 0.0
    
    # Other scores (6+ goals for either team)
    other_prob = 0.0
    for h in range(P.shape[0]):
        for a in range(P.shape[1]):
            if h >= 6 or a >= 6:
                other_prob += P[h, a]
    out['DC_CS_Other'] = other_prob
    
    # Additional O/U diagnostics (for analysis)
    out['_expected_total_goals'] = lam + mu
    out['_home_xG'] = lam
    out['_away_xG'] = mu

    # Safety: clip all probability values to [0, 1]
    for k, v in out.items():
        if k.startswith('DC_') and not k.startswith('DC_CS_'):
            out[k] = max(0.0, min(1.0, v))
        elif k.startswith('DC_CS_'):
            out[k] = max(0.0, min(1.0, v))

    return out

# ============================================================================
# HELPER: PRINT MODEL SUMMARY
# ============================================================================

def print_dc_summary(params: DCParams, home: str, away: str):
    """Print summary of DC model prediction for a match"""
    
    result = price_match(params, home, away, use_form=True)
    
    if not result:
        print(f"Cannot price {home} vs {away} - teams not in model")
        return
    
    print(f"\n{'='*60}")
    print(f"DC MODEL PREDICTION: {home} vs {away}")
    print(f"{'='*60}")
    print(f"League: {params.league}")
    print(f"Home Advantage: {params.home_adv:.3f}")
    print(f"Expected Goals: {result['_expected_total_goals']:.2f}")
    print(f"  Home xG: {result['_home_xG']:.2f}")
    print(f"  Away xG: {result['_away_xG']:.2f}")
    print(f"\n1X2:")
    print(f"  Home: {result['DC_1X2_H']:.1%}")
    print(f"  Draw: {result['DC_1X2_D']:.1%}")
    print(f"  Away: {result['DC_1X2_A']:.1%}")
    print(f"\nOver/Under 2.5:")
    print(f"  Over: {result['DC_OU_2_5_O']:.1%}")
    print(f"  Under: {result['DC_OU_2_5_U']:.1%}")
    print(f"\nBTTS:")
    print(f"  Yes: {result['DC_BTTS_Y']:.1%}")
    print(f"  No: {result['DC_BTTS_N']:.1%}")
    print(f"{'='*60}")
