# calibration.py
import numpy as np
from sklearn.linear_model import LogisticRegression

class TemperatureScaler:
    def __init__(self):
        self.T = 1.0
    def fit(self, logits: np.ndarray, y: np.ndarray):
        # 1D search over temperature
        def nll(T):
            P = softmax(logits / T)
            eps=1e-12
            return -np.mean(np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0)))
        T_vals = np.linspace(0.5, 5.0, 30)
        self.T = T_vals[np.argmin([nll(T) for T in T_vals])]
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
        self.lr = LogisticRegression(C=self.C, max_iter=self.max_iter, multi_class="multinomial", n_jobs=-1)
        self.lr.fit(X, y)
        return self
    def transform(self, P: np.ndarray):
        X = np.log(np.clip(P, 1e-12, 1-1e-12))
        # predict_proba returns calibrated vector
        return self.lr.predict_proba(X)
