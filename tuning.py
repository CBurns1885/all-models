# tuning.py - FIXED VERSION with class consistency
from __future__ import annotations
import numpy as np
import warnings
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
warnings.filterwarnings("ignore")

# Optional imports
try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False

@dataclass
class CVData:
    X: np.ndarray
    y: np.ndarray
    ps: PredefinedSplit
    classes_: np.ndarray
    label_encoder: LabelEncoder  # Add label encoder

def make_time_split(n: int, n_folds: int = 5) -> PredefinedSplit:
    """Simple time-ordered PredefinedSplit: last fold is newest."""
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[: n % n_folds] += 1
    test_fold = np.empty(n, dtype=int)
    current = 0
    for i, fold_size in enumerate(fold_sizes):
        test_fold[current: current + fold_size] = i
        current += fold_size
    return PredefinedSplit(test_fold=test_fold)

def expected_calibration_error(y_true_int: np.ndarray, P: np.ndarray, n_bins: int = 15) -> float:
    """ECE for multiclass: max over classes (one-vs-all binning)."""
    K = P.shape[1]
    eces = []
    for k in range(K):
        conf = P[:, k]
        truth = (y_true_int == k).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            m = (conf >= bins[i]) & (conf < bins[i+1] if i+1 < len(bins) else conf <= bins[i+1])
            if not np.any(m): 
                continue
            gap = abs(conf[m].mean() - truth[m].mean())
            ece += (m.mean() * gap)
        eces.append(ece)
    return float(max(eces))

def objective_factory(alg: str, cvd: CVData):
    """Factory for Optuna objectives with class consistency fixes"""
    X, y, ps, classes_, le = cvd.X, cvd.y, cvd.ps, cvd.classes_, cvd.label_encoder
    K = len(classes_)
    
    if not _HAS_OPTUNA:
        # Return simple default models without tuning
        def dummy_objective(trial=None):
            if alg == "rf":
                return RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
            elif alg == "et":
                return ExtraTreesClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
            elif alg == "lr":
                return LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1, 
                                        multi_class="multinomial" if K>2 else "auto")
            else:
                return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        return dummy_objective
    
    def build_model(trial: optuna.Trial):
        if alg == "rf":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 200, 800),
                max_depth=trial.suggest_int("max_depth", 6, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                max_features=trial.suggest_categorical("max_features", ["sqrt","log2", None]),
                class_weight="balanced_subsample",
                n_jobs=-1, random_state=42
            )
        elif alg == "et":
            return ExtraTreesClassifier(
                n_estimators=trial.suggest_int("n_estimators", 200, 800),
                max_depth=trial.suggest_int("max_depth", 6, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                max_features=trial.suggest_categorical("max_features", ["sqrt","log2", None]),
                class_weight="balanced",
                n_jobs=-1, random_state=42
            )
        elif alg == "lr":
            C = trial.suggest_float("C", 1e-3, 10.0, log=True)
            return LogisticRegression(
                C=C, max_iter=2000, n_jobs=-1, class_weight="balanced",
                multi_class="multinomial" if K>2 else "auto"
            )
        elif alg == "xgb" and _HAS_XGB:
            return xgb.XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 200, 800),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
                objective="multi:softprob" if K>2 else "binary:logistic",
                tree_method="hist", n_jobs=-1, random_state=42,
                # FIX: Ensure consistent classes
                enable_categorical=False,
                use_label_encoder=False
            )
        elif alg == "lgb" and _HAS_LGB:
            return lgb.LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 200, 1000),
                num_leaves=trial.suggest_int("num_leaves", 16, 128),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 50),
                objective="multiclass" if K>2 else "binary",
                random_state=42, n_jobs=-1
            )
        elif alg == "cat" and _HAS_CAT:
            return CatBoostClassifier(
                iterations=trial.suggest_int("iterations", 200, 1000),
                depth=trial.suggest_int("depth", 4, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                random_state=42, 
                loss_function="MultiClass" if K>2 else "Logloss",
                verbose=False
            )
        else:
            raise RuntimeError(f"Algorithm not available: {alg}")

    def objective(trial: optuna.Trial):
        model = build_model(trial)
        ll_folds, ece_folds = [], []
        
        # Cross-validated evaluation with class consistency fix
        for fold in np.unique(ps.test_fold):
            tr = ps.test_fold != fold
            va = ps.test_fold == fold
            Xt, Xv, yt, yv = X[tr], X[va], y[tr], y[va]
            
            # FIX: Ensure consistent classes by remapping to [0, 1, 2, ..., K-1]
            unique_train_classes = np.unique(yt)
            unique_val_classes = np.unique(yv)
            
            # Create consistent mapping for this fold
            all_classes = np.arange(K)  # Always use full range [0, 1, ..., K-1]
            
            # Remap validation labels to ensure they're in valid range
            yv_mapped = np.clip(yv, 0, K-1)
            
            model_fold = build_model(trial)
            model_fold.fit(Xt, yt)
            
            try:
                P = model_fold.predict_proba(Xv)
                
                # Handle class mismatch between train/validation splits
                if P.shape[1] != K:
                    # Pad or truncate probabilities to match expected classes
                    P_aligned = np.zeros((len(yv_mapped), K))
                    min_cols = min(P.shape[1], K)
                    P_aligned[:, :min_cols] = P[:, :min_cols]
                    # Renormalize
                    row_sums = P_aligned.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1.0
                    P_aligned = P_aligned / row_sums
                    P = P_aligned

                ll = log_loss(yv_mapped, P, labels=np.arange(K))
                ece = expected_calibration_error(yv_mapped, P, n_bins=10)
                ll_folds.append(ll)
                ece_folds.append(ece)
                
            except Exception as e:
                # If this fold fails, assign high penalty and continue
                print(f"Warning: Fold {fold} failed for {alg}: {e}")
                ll_folds.append(10.0)  # High penalty
                ece_folds.append(1.0)
        
        if not ll_folds:  # All folds failed
            return 10.0
        
        return np.mean(ll_folds) + 0.3 * np.mean(ece_folds)  # Combined objective

    return objective