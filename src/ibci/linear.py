"""Linear lag decoders (ridge on a causal window of binned features) and their recalibration variants.

A decoder maps z-scored features Z (T, C) to kinematics Y (T, K) through ``n_lags`` causal lags:
    y_t = sum_{l=0}^{L-1} W[l] z_{t-l} + b,  W: (L, C, K)
``W[0]`` multiplies the current bin. Bins before the start of a segment count as zero, which matches
``preprocess.add_history`` (oldest lag first there; we store current-lag first here).
"""
import numpy as np


def lagged(Z: np.ndarray, n_lags: int) -> np.ndarray:
    """(T, C) -> (T, L*C) with block l holding z_{t-l} (zeros before t=0)."""
    T, C = Z.shape
    out = np.zeros((T, n_lags, C), dtype=np.float32)
    for l in range(n_lags):
        out[l:, l] = Z[: T - l]
    return out.reshape(T, n_lags * C)


def _center(X, Y):
    xm, ym = X.mean(0), Y.mean(0)
    return X - xm, Y - ym, xm, ym


def ridge(X: np.ndarray, Y: np.ndarray, alpha: float):
    """Ridge with an unpenalised intercept. Returns (W (D, K), b (K,))."""
    Xc, Yc, xm, ym = _center(X, Y)
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1], dtype=X.dtype), Xc.T @ Yc)
    return W, ym - xm @ W


def ridge_to_prior(X, Y, W0, b0, alpha):
    """Ridge shrunk toward a previous decoder: argmin ||Y - XW - b||^2 + alpha ||W - W0||^2."""
    Xc, Yc, xm, ym = _center(X, Y)
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1], dtype=X.dtype), Xc.T @ Yc + alpha * W0)
    return W, ym - xm @ W


def ridge_cv_alpha(X, Y, groups, alphas, prior=None, n_folds=5):
    """Pick alpha by K-fold CV over contiguous groups (e.g. trial ids). ``prior=(W0, b0)`` uses ridge_to_prior."""
    ug = np.unique(groups)
    if len(ug) < n_folds:
        n_folds = max(2, len(ug))
    folds = np.array_split(ug, n_folds)
    err = np.zeros(len(alphas))
    for f in folds:
        te = np.isin(groups, f)
        for a_i, a in enumerate(alphas):
            W, b = ridge(X[~te], Y[~te], a) if prior is None else ridge_to_prior(X[~te], Y[~te], prior[0], prior[1], a)
            err[a_i] += ((X[te] @ W + b - Y[te]) ** 2).sum()
    return alphas[int(np.argmin(err))]


class LagDecoder:
    def __init__(self, W: np.ndarray, b: np.ndarray, n_lags: int):
        self.n_lags = n_lags
        self.W = W.astype(np.float32)  # (L*C, K)
        self.b = b.astype(np.float32)

    @classmethod
    def fit(cls, Z, Y, n_lags=8, alpha=0.1):
        W, b = ridge(lagged(Z, n_lags), Y, alpha)
        return cls(W, b, n_lags)

    def predict(self, Z):
        return lagged(Z, self.n_lags) @ self.W + self.b

    @property
    def W3(self):
        """Weights as (L, C, K)."""
        return self.W.reshape(self.n_lags, -1, self.W.shape[1])
