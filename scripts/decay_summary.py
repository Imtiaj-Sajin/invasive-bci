"""Per-subject decay summary: overnight retention and time to 50% retention, with cluster-bootstrap 95% CIs.

Retention = R2 of the renormalized frozen decoder divided by own-day R2 (paired by test session). The median retention
at each target gap is interpolated in log(days) to find where it first falls below 0.5. CIs: resampling training
sessions with replacement (2,000 draws).

Usage: python scripts/decay_summary.py [--out results/decay_summary.csv]
"""
import argparse
import os

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.participants import humans, valid_pairs  # noqa: E402

SUBJ = [("monkey N", "results/anatomy/ladder.csv"), ("monkey C", "results/replication/C_ladder.csv"),
        ("monkey M", "results/replication/M_ladder.csv")]
SUBJ += [(f"human {p}", f"results/replication_bg/{p}_ladder.csv") for p in humans()]
GAPS = [1, 7, 30, 120, 480]


def t_half(med):
    lg = np.log(GAPS)
    for k in range(1, len(GAPS)):
        if med[k] < 0.5 <= med[k - 1]:
            return float(np.exp(lg[k - 1] + (med[k - 1] - 0.5) / (med[k - 1] - med[k]) * (lg[k] - lg[k - 1])))
    return np.inf if med[-1] >= 0.5 else (GAPS[0] if med[0] < 0.5 else np.nan)


def curve(d):
    return np.array([np.median(d[d.gap_target == g].ret) if (d.gap_target == g).any() else np.nan for g in GAPS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/decay_summary.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for name, path in SUBJ:
        d = valid_pairs(pd.read_csv(path))
        d = d[d.gap_target.isin(GAPS)].assign(ret=lambda x: x.L2 / x.own)
        med = curve(d)
        groups = d.train.unique()
        by = {g: d[d.train == g] for g in groups}
        bt_half, bt_1 = [], []
        for _ in range(2000):
            b = pd.concat([by[g] for g in rng.choice(groups, len(groups))])
            m = curve(b)
            bt_half.append(t_half(m))
            bt_1.append(m[0])
        bt_half = np.array(bt_half, dtype=float)
        rows.append(dict(subject=name, n_pairs=len(d), n_train_sessions=len(groups),
                         overnight=round(float(med[0]), 3),
                         overnight_ci=f"[{np.nanpercentile(bt_1, 2.5):.2f}, {np.nanpercentile(bt_1, 97.5):.2f}]",
                         t_half_days=round(t_half(med), 1),
                         t_half_ci=f"[{np.nanpercentile(bt_half, 2.5):.1f}, {np.nanpercentile(bt_half, 97.5):.1f}]"))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
