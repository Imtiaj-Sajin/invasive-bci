"""The 1% regularization rule across subjects: gains over alpha = 0.1 on the same day and at each gap.

For each subject's trade-off table (scripts/reg_tradeoff.py), the rule picks the largest alpha whose median same-day R2
is within 1% of the best median, using same-day data only. Per gap we report median R2 for alpha = 0.1 and for the
chosen alpha, the median paired gain with a cluster-bootstrap 95% CI over training sessions, and the share of training
sessions whose mean gain is positive. Human BrainGate participants are included when results/reg_tradeoff_<P>/ exists.

Usage: python scripts/rule_summary.py [--out results/reg_tradeoff/rule_across_subjects.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.participants import humans  # noqa: E402
from ibci.stats import cluster_bootstrap  # noqa: E402

HUMANS = humans()   # included participants (pre-specified reference-decoder rule, ibci.participants)


def subjects():
    out = [("N (LINK)", "results/reg_tradeoff/tradeoff.csv"), ("C", "results/reg_tradeoff_C/tradeoff.csv"),
           ("M", "results/reg_tradeoff_M/tradeoff.csv")]
    out += [(f"{p} (human)", f"results/reg_tradeoff_{p}/tradeoff.csv") for p in HUMANS
            if os.path.exists(f"results/reg_tradeoff_{p}/tradeoff.csv")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/reg_tradeoff/rule_across_subjects.csv")
    args = ap.parse_args()
    rows = []
    for name, path in subjects():
        d = pd.read_csv(path)
        same = d[d.gap_target == 0].groupby("alpha").r2.median()
        a_rule = max(a for a, v in same.items() if v >= 0.99 * same.max())
        w = d.pivot_table(index=["train", "gap_target"], columns="alpha", values="r2").reset_index()
        w["gain"] = w[a_rule] - w[0.1]
        for g, q in w.groupby("gap_target"):
            est, lo, hi = cluster_bootstrap(q, "gain", n_boot=2000)
            rows.append(dict(subject=name, alpha_rule=a_rule, gap=int(g), n=len(q), base=round(q[0.1].median(), 3),
                             rule=round(q[a_rule].median(), 3), gain=round(est, 3), ci=f"[{lo:.3f}, {hi:.3f}]",
                             frac_pos=round(float((q.groupby("train").gain.mean() > 0).mean()), 2)))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
