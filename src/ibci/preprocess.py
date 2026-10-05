"""Shared preprocessing: trial-based train/test splits, lag history, normalization.

The split follows the LINK paper (Temmar et al. 2025): train on the first 300 trials of a session and test on
the remaining trials, so cross-day numbers are comparable with theirs.
"""
import numpy as np

N_TRAIN_TRIALS = 300


def split_bins(trial_start: np.ndarray, n_bins: int, n_train_trials: int = N_TRAIN_TRIALS):
    """Return (train_slice, test_slice) over bins: train = first ``n_train_trials`` trials, test = the rest."""
    if len(trial_start) <= n_train_trials:
        raise ValueError(f"session has only {len(trial_start)} trials")
    cut = int(trial_start[n_train_trials])
    return slice(0, cut), slice(cut, n_bins)


def add_history(x: np.ndarray, n_lags: int) -> np.ndarray:
    """(T, C) -> (T, n_lags, C), oldest lag first and current bin last; bins before t=0 are zero."""
    T, C = x.shape
    out = np.zeros((T, n_lags, C), dtype=x.dtype)
    for k in range(n_lags):  # k = how many bins back
        out[k:, n_lags - 1 - k] = x[: T - k]
    return out


class ZScore:
    """Per-channel z-scoring with stats from a reference block. Channels with ~zero variance are left centred."""

    def fit(self, x: np.ndarray):
        self.mean = x.mean(0)
        self.std = x.std(0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std
