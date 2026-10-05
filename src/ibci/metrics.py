"""Decoding metrics, averaged over output dimensions (degrees of freedom) as in the LINK paper."""
import numpy as np


def r2_per_dof(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def corr_per_dof(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = pred - pred.mean(0)
    t = true - true.mean(0)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    return (p * t).sum(0) / np.maximum(den, 1e-12)


def summarize(pred: np.ndarray, true: np.ndarray) -> dict:
    r2 = r2_per_dof(pred, true)
    cc = corr_per_dof(pred, true)
    return {"r2": float(r2.mean()), "corr": float(cc.mean()), "mse": float(((pred - true) ** 2).mean()),
            **{f"r2_{i}": float(v) for i, v in enumerate(r2)}}
