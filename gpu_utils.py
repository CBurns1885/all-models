# gpu_utils.py
"""
GPU capability probing for the boosting libraries.

USE_GPU=1 only expresses intent. Whether GPU training actually works depends
on the installed wheel: the standard pip `lightgbm` wheel has no GPU support
and raises at fit time, and `catboost` with task_type="GPU" fails without
CUDA. Passing the GPU parameters blindly is dangerous here because the
training loop catches per-model failures and substitutes uniform
probabilities to keep the stacked feature width consistent — so a bad GPU
config would not crash the run, it would quietly drop that base model from
the ensemble for every fold and still produce confident-looking output.

This module probes each library ONCE with a tiny fit and caches the answer,
so GPU parameters are only ever passed when they demonstrably work.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

_PROBE_CACHE: Dict[str, bool] = {}

# Params per library when the GPU is confirmed working
_GPU_PARAMS = {
    "xgb": {"device": "cuda"},
    "lgb": {"device": "gpu"},
    "cat": {"task_type": "GPU", "devices": "0"},
}


def _probe(lib: str) -> bool:
    """Try a 2-iteration fit on the GPU. True only if it completes."""
    X = np.random.RandomState(0).rand(64, 4)
    y = (np.arange(64) % 2).astype(int)
    try:
        if lib == "xgb":
            import xgboost as xgb
            xgb.XGBClassifier(n_estimators=2, tree_method="hist",
                              verbosity=0, **_GPU_PARAMS["xgb"]).fit(X, y)
        elif lib == "lgb":
            import lightgbm as lgb
            lgb.LGBMClassifier(n_estimators=2, verbose=-1,
                               **_GPU_PARAMS["lgb"]).fit(X, y)
        elif lib == "cat":
            from catboost import CatBoostClassifier
            CatBoostClassifier(iterations=2, verbose=False,
                               **_GPU_PARAMS["cat"]).fit(X, y)
        else:
            return False
        return True
    except Exception as e:
        print(f"[GPU] {lib}: GPU unavailable ({type(e).__name__}: {str(e)[:120]}) "
              f"-- falling back to CPU for {lib}")
        return False


def gpu_params(lib: str) -> dict:
    """GPU kwargs for `lib` ('xgb'|'lgb'|'cat'), or {} when unavailable.

    Returns {} unless USE_GPU is set AND a probe fit actually succeeded.
    """
    try:
        from config import USE_GPU
    except Exception:
        return {}
    if not USE_GPU:
        return {}
    if lib not in _PROBE_CACHE:
        _PROBE_CACHE[lib] = _probe(lib)
        if _PROBE_CACHE[lib]:
            print(f"[GPU] {lib}: GPU acceleration active")
    return dict(_GPU_PARAMS[lib]) if _PROBE_CACHE[lib] else {}
