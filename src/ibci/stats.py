"""Statistics that respect the pair structure of cross-day analyses.

Pairs that share a training session are not independent, so confidence intervals are obtained with a cluster
bootstrap over training sessions (resample sessions with replacement, keep all their pairs).
"""
import numpy as np
import pandas as pd


def cluster_bootstrap(df: pd.DataFrame, value, cluster="train", stat=np.median, n_boot=2000, seed=0, ci=95):
    """Point estimate and percentile CI of ``stat`` of column/callable ``value`` with cluster resampling."""
    rng = np.random.default_rng(seed)
    vals = df[value].to_numpy() if isinstance(value, str) else np.asarray(value(df))
    groups = df[cluster].to_numpy()
    ug = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ug}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        boots[b] = stat(vals[np.concatenate([idx_by[g] for g in pick])])
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(stat(vals)), float(lo), float(hi)


def paired_gain(df, a, b, cluster="train", n_boot=2000, seed=0):
    """Median of (a - b) per pair with cluster-bootstrap CI and the share of clusters whose mean gain is > 0."""
    d = df.assign(_g=df[a] - df[b])
    est, lo, hi = cluster_bootstrap(d, "_g", cluster, np.median, n_boot, seed)
    per_cluster = d.groupby(cluster)["_g"].mean()
    return {"median_gain": est, "ci_lo": lo, "ci_hi": hi, "frac_clusters_positive": float((per_cluster > 0).mean()),
            "n_pairs": int(len(d)), "n_clusters": int(per_cluster.size)}


def by_gap_table(df, cols, gap="gap_target", cluster="train", n_boot=1000):
    """Median [95% CI] per gap for each column, as a tidy DataFrame."""
    rows = []
    for g, d in df.groupby(gap):
        row = {gap: g, "n_pairs": len(d)}
        for c in cols:
            est, lo, hi = cluster_bootstrap(d, c, cluster, np.median, n_boot)
            row[c] = f"{est:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)
