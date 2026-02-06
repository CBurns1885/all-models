# calibration.py
"""
Calibration methods for probability predictions.

Includes:
- TemperatureScaler: Simple temperature scaling for binary/multiclass
- DirichletCalibrator: Vector calibration using multinomial logistic regression
- IsotonicOrdinalCalibrator: Ordinal calibration enforcing class ordering
- BetaCalibrator: Beta calibration for binary probabilities
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from typing import Optional

class TemperatureScaler:
    def __init__(self):
        self.T = 1.0
    def fit(self, logits: np.ndarray, y: np.ndarray):
        from scipy.optimize import minimize_scalar
        def nll(T):
            P = softmax(logits / T)
            eps=1e-12
            return -np.mean(np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0)))
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.T = result.x
        return self
    def transform(self, logits: np.ndarray):
        return softmax(logits / self.T)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

class DirichletCalibrator:
    """
    Vector (Dirichlet) calibration using multinomial logistic regression
    on the probability simplex logits (log odds). Simple & effective.
    """
    def __init__(self, C=1.0, max_iter=2000):
        self.C = C; self.max_iter = max_iter
        self.lr = None
        self.K = None
    def fit(self, P: np.ndarray, y: np.ndarray):
        self.K = P.shape[1]
        X = np.log(np.clip(P, 1e-12, 1-1e-12))
        self.lr = LogisticRegression(C=self.C, max_iter=self.max_iter, n_jobs=-1)
        self.lr.fit(X, y)
        return self
    def transform(self, P: np.ndarray):
        X = np.log(np.clip(P, 1e-12, 1-1e-12))
        # predict_proba returns calibrated vector
        return self.lr.predict_proba(X)


class IsotonicOrdinalCalibrator:
    """
    Ordinal calibration using isotonic regression to enforce class ordering.

    For ordinal targets (e.g., 0,1,2,3,4,5+ goals), this ensures that
    cumulative probabilities P(Y <= k) are monotonically non-decreasing.

    This is theoretically sound for ordered classes and prevents
    probability inversions that can occur with standard calibration.
    """

    def __init__(self, method: str = "cumulative"):
        """
        Args:
            method: "cumulative" (default) or "adjacent"
                - cumulative: Calibrate P(Y <= k) cumulatively
                - adjacent: Calibrate each class probability separately
        """
        self.method = method
        self.isotonic_models = {}
        self.K = None
        self.is_fitted = False

    def fit(self, P: np.ndarray, y: np.ndarray):
        """
        Fit isotonic regression models.

        Args:
            P: Probability matrix (n_samples, n_classes)
            y: True labels (integers 0 to K-1)
        """
        self.K = P.shape[1]

        if self.method == "cumulative":
            # Fit isotonic regression on cumulative probabilities
            # For each threshold k, predict P(Y <= k)
            for k in range(self.K - 1):
                # Create binary target: 1 if y <= k, else 0
                y_binary = (y <= k).astype(int)

                # Cumulative probability up to class k
                P_cumulative = P[:, :k+1].sum(axis=1)

                # Fit isotonic regression
                iso = IsotonicRegression(
                    y_min=0.0,
                    y_max=1.0,
                    out_of_bounds='clip',
                    increasing=True  # Enforce monotonicity
                )
                iso.fit(P_cumulative, y_binary)
                self.isotonic_models[k] = iso

        elif self.method == "adjacent":
            # Fit isotonic regression on each class probability
            for k in range(self.K):
                y_binary = (y == k).astype(int)
                iso = IsotonicRegression(
                    y_min=0.0,
                    y_max=1.0,
                    out_of_bounds='clip'
                )
                iso.fit(P[:, k], y_binary)
                self.isotonic_models[k] = iso

        self.is_fitted = True
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.

        Args:
            P: Probability matrix (n_samples, n_classes)

        Returns:
            Calibrated probability matrix
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted")

        n_samples = P.shape[0]

        if self.method == "cumulative":
            # Convert cumulative calibrated probs back to class probs
            P_cal = np.zeros((n_samples, self.K))

            # Get calibrated cumulative probabilities
            cum_probs = np.zeros((n_samples, self.K))

            for k in range(self.K - 1):
                P_cumulative = P[:, :k+1].sum(axis=1)
                cum_probs[:, k] = self.isotonic_models[k].predict(P_cumulative)

            # Last cumulative is always 1.0
            cum_probs[:, -1] = 1.0

            # Ensure monotonicity
            for i in range(1, self.K):
                cum_probs[:, i] = np.maximum(cum_probs[:, i], cum_probs[:, i-1])

            # Convert cumulative to class probabilities
            P_cal[:, 0] = cum_probs[:, 0]
            for k in range(1, self.K):
                P_cal[:, k] = cum_probs[:, k] - cum_probs[:, k-1]

            # Ensure non-negative and normalize
            P_cal = np.maximum(P_cal, 0.0)
            row_sums = P_cal.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            P_cal = P_cal / row_sums

        elif self.method == "adjacent":
            # Apply isotonic calibration to each class
            P_cal = np.zeros((n_samples, self.K))
            for k in range(self.K):
                P_cal[:, k] = self.isotonic_models[k].predict(P[:, k])

            # Normalize to sum to 1
            row_sums = P_cal.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            P_cal = P_cal / row_sums

        return P_cal


class BetaCalibrator:
    """
    Beta calibration for binary classification.

    Transforms probabilities using a beta function to correct
    for systematic over/under-confidence. More flexible than
    Platt scaling for non-sigmoidal distortions.

    Reference: Kull et al. "Beta calibration: a well-founded and
    easily implemented improvement on logistic calibration for
    binary classifiers" (AISTATS 2017)
    """

    def __init__(self, param_bounds: tuple = (0.01, 100.0)):
        self.a = 1.0
        self.b = 1.0
        self.c = 0.0
        self.param_bounds = param_bounds
        self.is_fitted = False

    def fit(self, P: np.ndarray, y: np.ndarray):
        """
        Fit beta calibration parameters.

        Args:
            P: Probability predictions (n_samples,) or (n_samples, 2)
            y: True binary labels
        """
        if P.ndim == 2:
            p = P[:, 1]  # Positive class probability
        else:
            p = P

        # Clip probabilities to avoid log(0)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)

        # Transform to log-odds space
        log_odds = np.log(p / (1 - p))

        # Fit logistic regression on log-odds
        lr = LogisticRegression(C=1e10, max_iter=1000, solver='lbfgs')
        lr.fit(log_odds.reshape(-1, 1), y)

        self.a = lr.coef_[0, 0]
        self.c = lr.intercept_[0]
        self.b = 1.0  # Simplified version

        self.is_fitted = True
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        """
        Apply beta calibration.

        Args:
            P: Probability predictions

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted")

        if P.ndim == 2:
            p = P[:, 1]
        else:
            p = P

        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)

        # Apply calibration: logit(p_cal) = a * logit(p) + c
        log_odds = np.log(p / (1 - p))
        log_odds_cal = self.a * log_odds + self.c

        # Convert back to probability
        p_cal = 1.0 / (1.0 + np.exp(-log_odds_cal))
        p_cal = np.clip(p_cal, eps, 1 - eps)

        if P.ndim == 2:
            return np.column_stack([1 - p_cal, p_cal])
        return p_cal


class PlattScaler:
    """
    Platt scaling (sigmoid calibration) for binary classification.

    A simple and effective calibration method that fits a logistic
    regression on the predicted probabilities (or scores).
    """

    def __init__(self):
        self.lr = None
        self.is_fitted = False

    def fit(self, P: np.ndarray, y: np.ndarray):
        """Fit Platt scaling."""
        if P.ndim == 2:
            p = P[:, 1]
        else:
            p = P

        # Use log-odds as features
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        log_odds = np.log(p / (1 - p))

        self.lr = LogisticRegression(C=1e10, max_iter=1000, solver='lbfgs')
        self.lr.fit(log_odds.reshape(-1, 1), y)
        self.is_fitted = True
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted")

        if P.ndim == 2:
            p = P[:, 1]
        else:
            p = P

        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        log_odds = np.log(p / (1 - p))

        p_cal = self.lr.predict_proba(log_odds.reshape(-1, 1))[:, 1]

        if P.ndim == 2:
            return np.column_stack([1 - p_cal, p_cal])
        return p_cal


def get_calibrator_for_market(target_col: str, n_classes: int):
    """
    Get the appropriate calibrator for a market type.

    Args:
        target_col: Target column name
        n_classes: Number of classes

    Returns:
        Appropriate calibrator instance
    """
    # Import here to avoid circular dependency
    try:
        from market_config import should_use_isotonic, get_market_type, MarketType

        if should_use_isotonic(target_col):
            return IsotonicOrdinalCalibrator(method="cumulative")

        market_type = get_market_type(target_col)

        if market_type == MarketType.ORDINAL:
            return IsotonicOrdinalCalibrator(method="cumulative")
        elif n_classes == 2:
            return BetaCalibrator()
        else:
            return DirichletCalibrator()

    except ImportError:
        # Fallback if market_config not available
        if n_classes == 2:
            return TemperatureScaler()
        else:
            return DirichletCalibrator()
