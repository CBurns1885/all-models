# model_ordinal_FULL.py
# FULL VERSION: High-performance ordinal predictions
# Goal Range, CS - Maximum accuracy

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier

class OrdinalMarketModelFull:
    """
    Full-power ordinal model for high-performance hardware.

    Key features:
    - More trees (200 for accuracy)
    - Deeper depth (10 for complexity)
    - Single robust model (RF)
    - Advanced smoothing
    """
    
    def __init__(self, target_name: str, ordered_classes: List[str], random_state: int = 42):
        self.target_name = target_name
        self.ordered_classes = ordered_classes
        self.n_classes = len(ordered_classes)
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
    def _build_model(self):
        """Build single full-power model."""
        import os
        n_est = int(os.environ.get("N_ESTIMATORS", "300"))
        return RandomForestClassifier(
            n_estimators=n_est,  # Respects environment variable
            max_depth=10,        # Deeper for complexity
            min_samples_leaf=20,
            max_features='sqrt',
            n_jobs=-1,           # Multi-threaded
            random_state=self.random_state
        )
    
    def _fast_smoothing(self, probs: np.ndarray) -> np.ndarray:
        """Fast adjacent smoothing."""
        n_samples, n_classes = probs.shape
        smoothed = probs.copy()
        
        # Simple: add 5% from neighbors
        for i in range(n_samples):
            for j in range(1, n_classes-1):
                smoothed[i, j] += 0.05 * (probs[i, j-1] + probs[i, j+1])
            
            # Edge cases
            smoothed[i, 0] += 0.05 * probs[i, 1]
            smoothed[i, -1] += 0.05 * probs[i, -2]
            
            # Normalize
            smoothed[i] /= smoothed[i].sum()
        
        return smoothed
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Ultra-fast training."""
        self.model = self._build_model()
        
        try:
            self.model.fit(X, y)
            print(f"  [OK] RF (ordinal-full, {self.n_classes} classes)")
        except Exception as e:
            print(f"  [X] RF: {e}")
            raise
        
        self.is_fitted = True
    
    def predict_proba(self, X: np.ndarray, smoothing: bool = True) -> np.ndarray:
        """Ultra-fast prediction."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        pred = self.model.predict_proba(X)
        
        if smoothing:
            pred = self._fast_smoothing(pred)
        
        return pred


def is_ordinal_market(target_col: str) -> bool:
    """Check if ordinal market."""
    return target_col in ['y_GOAL_RANGE', 'y_CS', 'y_HomeCardsY_BAND',
                          'y_AwayCardsY_BAND', 'y_HomeCorners_BAND', 'y_AwayCorners_BAND']


# Alias for backward compatibility
OrdinalMarketModel = OrdinalMarketModelFull
