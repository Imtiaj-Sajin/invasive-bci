"""Summarize the latent-alignment ceiling (results/latent_ceiling/<subject>.csv, from scripts/latent_ceiling.py).

All rungs are measured on the same pairs and test trials, with the tuned penalty. With all labelled trials of the later
day ('nall', the data the reference decoder is trained on):
  rotation ceiling   L4_nall  best rotation of the dominant 16-dim subspace, fitted with labels
  per-channel        L3_nall  re-weighting of each channel, fitted with labels
  label-free         L4u      Procrustes rotation without labels;  L2  renormalization only
Share of the loss recovered by a rung R:  (R - L2) / (own - L2), computed from the medians over pairs of each column
within one gap band, with a cluster bootstrap over training sessions (2,000 resamples). Paired per-session comparisons
(Wilcoxon over training sessions, all gaps): ceiling vs label-free rotation (does a good rotation exist that the
label-free method misses?) and re-weighting vs ceiling (is the change better described per channel or as a rotation?).

Usage: python scripts/ceiling_summary.py [--out results/latent_ceiling/summary.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.participants import humans, individuals, valid_pairs  # noqa: E402

BANDS = {"short": (1, 7), "mid": (30, 30), "long": (120, 480), "all": (1, 480)}


def _share(l2, own, x):
    lo = np.median(l2)
    den = np.median(own) - lo
    return float((np.median(x) - lo) / den) if den > 0.02 else np.nan


def share(d, col):
    return _share(d.L2.to_numpy(), d.own.to_numpy(), d[col].to_numpy())


def boot_share(d, col, n_boot=2000, seed=0):
    """Cluster bootstrap over training sessions (resampling row indices; same draws as resampling the groups)."""
    rng = np.random.default_rng(seed)
    groups = d.train.unique()
    rows = {g: np.flatnonzero((d.train == g).to_numpy()) for g in groups}
    l2, own, x = d.L2.to_numpy(), d.own.to_numpy(), d[col].to_numpy()
    est = _share(l2, own, x)
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[g] for g in rng.choice(groups, len(groups))])
        bs.append(_share(l2[idx], own[idx], x[idx]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return [round(est, 3), round(float(lo), 3), round(float(hi), 3)]


def paired(d, a, b):
    s = d.groupby("train")[[a, b]].mean().dropna()
    diff = s[a] - s[b]
    p = float(stats.wilcoxon(diff).pvalue) if len(s) >= 6 and (diff != 0).any() else None
    return {"n_sessions": int(len(s)), "median_diff": round(float(diff.median()), 4),
            "frac_positive": round(float((diff > 0).mean()), 2), "P": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/latent_ceiling/summary.json")
    args = ap.parse_args()
    out = {}
    for k in individuals():
        p = f"results/latent_ceiling/{k}.csv"
        if not os.path.exists(p):
            continue
        d = valid_pairs(pd.read_csv(p))
        r = {"n_pairs": int(len(d)), "n_train": int(d.train.nunique()),
             "trials_available_median": float(d.n_trials_available.median()),
             "retention_by_gap": {int(g): {c: round(float((q[c] / q.own).median()), 3)
                                           for c in ("L2", "L4u", "L4s", "L3_n300", "L4_n300", "L3_nall", "L4_nall")}
                                  for g, q in d.groupby("gap_target")}}
        for band, (g0, g1) in BANDS.items():
            q = d[(d.gap_target >= g0) & (d.gap_target <= g1)]
            if q.train.nunique() >= 6:
                r[f"share_{band}"] = {c: boot_share(q, c) for c in ("L4u", "L4_nall", "L3_nall")}
        r["ceiling_vs_labelfree"] = paired(d, "L4_nall", "L4u")
        r["reweight_vs_ceiling"] = paired(d, "L3_nall", "L4_nall")
        out[k] = r
        sl = r.get("share_long", {})
        print(f"{k:4s} pairs {len(d):4d}  long-gap share recovered: label-free rot {sl.get('L4u')}, "
              f"best rot {sl.get('L4_nall')}, re-weight {sl.get('L3_nall')}; "
              f"re-weight - best rot {r['reweight_vs_ceiling']['median_diff']:+.3f} (P={r['reweight_vs_ceiling']['P']})")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
