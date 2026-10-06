"""Decoder decay across subjects: retention (rung R2 / own-day R2) vs days, monkeys N (LINK), C and M (000688).

A  renormalized frozen decoder (no labels)      B  + per-channel gains (96 labelled parameters)
Medians with cluster-bootstrap 95% CIs over training sessions.

Usage: python scripts/make_decay_figure.py [--out results/figures]
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci import plotting as P  # noqa: E402
from ibci.stats import cluster_bootstrap  # noqa: E402

SUBJECTS = [("monkey N (LINK, SBP)", "results/anatomy/ladder.csv"),
            ("monkey C (000688, units)", "results/replication/C_ladder.csv"),
            ("monkey M (000688, units)", "results/replication/M_ladder.csv")]
GAPS = [1, 7, 30, 120, 480]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    P.setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, (col, title) in zip(axes, [("L2", "A  No labels: daily renormalization"),
                                       ("L3_n300", "B  + 96 per-channel gains (labelled)")]):
        labels = []
        for (name, path), c in zip(SUBJECTS, P.CAT):
            d = pd.read_csv(path)
            d = d[d.gap_target.isin(GAPS)].assign(ret=lambda x: x[col] / x["own"])
            est = [cluster_bootstrap(d[d.gap_target == g], "ret", n_boot=1000) for g in GAPS]
            m = [e[0] for e in est]
            ax.plot(GAPS, m, color=c, marker="o", ms=3.5, label=name)
            ax.fill_between(GAPS, [e[1] for e in est], [e[2] for e in est], color=c, alpha=0.15, lw=0)
            labels.append((m[-1], name))
        ax.axhline(1, color=P.INK2, ls=":", lw=0.8)
        ax.axhline(0, color=P.INK2, lw=0.6)
        P.day_axis(ax, GAPS)
        ax.set_xlabel("days between training and test session")
        ax.set_title(title)
        if ax is axes[0]:
            ax.set_ylabel("R² relative to own-day decoder")
            ax.legend(loc="lower left", fontsize=7)
    fig.subplots_adjust(wspace=0.15)
    P.save(fig, os.path.join(args.out, "fig_decay_xsubject"))


if __name__ == "__main__":
    main()
