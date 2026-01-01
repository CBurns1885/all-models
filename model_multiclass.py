# model_multiclass_FULL.py
# FULL VERSION: High-performance multiclass predictions
# 1X2, HT, HTFT - Maximum accuracy

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class MulticlassMarketModelFull:
    """
    Full-power multiclass model for high-performance hardware.

    Key features:
    - More trees (300 for accuracy)
    - Deeper depth (12 for complexity)
    - 2 robust models
    - Multi-threaded
    """
    
    def __init__(self, target_name: str, n_classes: int, random_state: int = 42):
        self.target_name = target_name
        self.n_classes = n_classes
        self.random_state = random_state
        self.models = {}
        self.is_fitted = False
        
    def _build_models(self) -> Dict:
        """Build full-power model set."""
        import os
        n_est = int(os.environ.get("N_ESTIMATORS", "300"))
        models = {}

        # Full-power Random Forest
        models['rf'] = RandomForestClassifier(
            n_estimators=n_est,  # Respects environment variable
            max_depth=12,        # Deeper for complexity
            min_samples_leaf=15,
            max_features='sqrt',
            n_jobs=-1,           # Multi-threaded
            random_state=self.random_state
        )

        # Logistic Regression
        models['lr'] = LogisticRegression(
            solver='lbfgs',
            C=1.0,
            max_iter=500,
            n_jobs=-1,         # Multi-threaded
            random_state=self.random_state
        )
        
        return models
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fast training."""
        base_models = self._build_models()
        
        for name, model in base_models.items():
            try:
                model.fit(X, y)
                self.models[name] = model
                print(f"  [OK] {name.upper()} (multiclass-full, {self.n_classes} classes)")
            except Exception as e:
                print(f"  [X] {name.upper()}: {e}")
        
        if not self.models:
            raise RuntimeError(f"No models trained")
        
        self.is_fitted = True
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Fast prediction."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        preds = [m.predict_proba(X) for m in self.models.values()]
        return np.mean(preds, axis=0)


def is_multiclass_market(target_col: str) -> bool:
    """Check if multiclass market."""
    return any(target_col == p for p in ['y_1X2', 'y_HT', 'y_HTFT', 'y_GOAL_RANGE'])


# Alias for backward compatibility
MulticlassMarketModel = MulticlassMarketModelFull
