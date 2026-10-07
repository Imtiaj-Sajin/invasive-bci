"""Permutation null for electrode silence and recovery (200 shuffles), replacing the single shuffle.

For each array, each electrode's session order is shuffled independently, and the silence and recovery statistics of
scripts/failure_robustness.py are recomputed. Repeating this gives a null distribution of the pooled recovery
fraction (and of the number of silences) for the main definition, the fixed-microvolt threshold, deep silence and
silences of at least 30 days. We report the observed value, the null mean and 95% interval, a permutation P value and
the share of arrays whose observed recovery fraction exceeds their own null median (array as the unit).

Usage: python scripts/failure_null.py [--n-perm 200] [--out results/failure_robustness]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from failure_robustness import electrode_series, fixed_uv_rate, revival_counts  # noqa: E402

VARIANTS = {"base": dict(col="r-4.5"), "fixed_uv": dict(col="r_fixed"),
            "deep": dict(col="r-4.5", silent_max=0.5), "long_silence": dict(col="r-4.5", min_silent_days=30)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default="results/failure_robustness")
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(args.out, "electrode_thresholds.csv.gz"))
    rng = np.random.default_rng(1)
    obs = {k: np.zeros(2) for k in VARIANTS}
    null = {k: np.zeros((args.n_perm, 2)) for k in VARIANTS}
    arr_above = {k: [] for k in VARIANTS}
    for (p, a), g in df.groupby(["participant", "array"]):
        g = fixed_uv_rate(g)
        for k, v in VARIANTS.items():
            ser = electrode_series(g, v["col"])
            kw = {kk: vv for kk, vv in v.items() if kk != "col"}
            s, r = revival_counts(ser, None, **kw)
            obs[k] += (s, r)
            arr_null = np.zeros((args.n_perm, 2))
            for i in range(args.n_perm):
                shuf = {e: (rng.permutation(x), d) for e, (x, d) in ser.items()}
                arr_null[i] = revival_counts(shuf, None, **kw)
            null[k] += arr_null
            fr = arr_null[:, 1] / np.maximum(arr_null[:, 0], 1)
            if s >= 3:
                arr_above[k].append(float(r / s > np.median(fr)))
        print(p, a, flush=True)
    out = {}
    for k in VARIANTS:
        f_obs = obs[k][1] / obs[k][0]
        f_null = null[k][:, 1] / null[k][:, 0]
        out[k] = {"silenced_obs": int(obs[k][0]), "revived_obs": int(obs[k][1]), "fraction_obs": float(f_obs),
                  "fraction_null_mean": float(f_null.mean()),
                  "fraction_null_ci": [float(np.percentile(f_null, 2.5)), float(np.percentile(f_null, 97.5))],
                  "P_upper": float((np.sum(f_null >= f_obs) + 1) / (len(f_null) + 1)),
                  "silenced_null_mean": float(null[k][:, 0].mean()),
                  "arrays_above_null_median": f"{int(sum(arr_above[k]))}/{len(arr_above[k])}"}
    json.dump(out, open(os.path.join(args.out, "null_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
