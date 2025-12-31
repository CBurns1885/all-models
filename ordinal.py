# ordinal.py
import numpy as np
from sklearn.linear_model import LogisticRegression

class CORALOrdinal:
    """
    Simple CORAL: K ordered classes -> K-1 binary heads with shared weights.
    Predicts cumulative probabilities, then converts to class probabilities.
    """
    def __init__(self, C=1.0, max_iter=2000):
        self.C = C
        self.max_iter = max_iter
        self.clfs = []
        self.thresholds_ = None
        self.classes_ = None

    def fit(self, X, y_ord):
        self.classes_ = np.unique(y_ord)
        K = len(self.classes_)
        self.clfs = []
        for k in range(K-1):
            yk = (y_ord > k).astype(int)
            clf = LogisticRegression(C=self.C, max_iter=self.max_iter, n_jobs=-1)
            clf.fit(X, yk)
            self.clfs.append(clf)
        return self

    def predict_proba(self, X):
        K = len(self.classes_)
        if K == 2:
            p1 = self.clfs[0].predict_proba(X)[:,1]
            P = np.vstack([1-p1, p1]).T
            return P
        cum = []
        for clf in self.clfs:
            cum.append(clf.predict_proba(X)[:,1])
        cum = np.clip(np.vstack(cum).T, 1e-9, 1-1e-9)  # shape (n, K-1)
        # Convert cumulative to class probs
        n = X.shape[0]
        P = np.zeros((n, K))
        P[:,0] = 1 - cum[:,0]
        for k in range(1, K-1):
            P[:,k] = cum[:,k-1] - cum[:,k]
        P[:,K-1] = cum[:,K-2]
        # normalize for safety
        s = P.sum(axis=1, keepdims=True); s[s==0]=1.0
        return P / s
