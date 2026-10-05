"""Figures for the drift-anatomy / simulator study. Each subcommand reads a results folder and writes PNG + PDF.

  python scripts/make_figures.py anatomy   --ladder results/anatomy/ladder.csv --out results/figures
  python scripts/make_figures.py efficiency --ladder results/data_efficiency/ladder.csv --out results/figures
  python scripts/make_figures.py health    --table results/channel_health/channel_day.csv --out results/figures
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci import plotting as P  # noqa: E402

RUNG_LABELS = {
    "L1": "recentre (no labels)", "L2": "renormalize (no labels)", "L4u": "Procrustes align (no labels)",
    "L3": "+ channel gains (96 p)", "L4": "+ latent rotation (k=16)", "L5": "+ full input remap",
    "L6p": "recalibrate (ridge to prior)",
}


def q(x, p):
    return np.nanpercentile(x, p)


def fig_anatomy(ladder_csv, out):
    df = pd.read_csv(ladder_csv)
    n = max(int(c.split("_n")[1]) for c in df.columns if c.startswith("L3_n"))
    rungs = ["L1", "L2", "L3", "L5", "L6p"]
    col = {r: (f"{r}_n{n}" if r in ("L3", "L4", "L5", "L6p") else r) for r in rungs}
    gaps = sorted(df.gap_target.unique())
    colors = P.ordinal(len(rungs))
    P.setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    own_med = [df[df.gap_target == g]["own"].median() for g in gaps]
    ax.plot(gaps, own_med, ls="--", color=P.INK2, lw=1.2)
    labels = [(own_med[-1], "own-day decoder")]
    for r, c in zip(rungs, colors):
        v = [df[df.gap_target == g][col[r]] for g in gaps]
        ax.plot(gaps, [x.median() for x in v], color=c, marker="o", ms=3.5)
        ax.fill_between(gaps, [q(x, 25) for x in v], [q(x, 75) for x in v], color=c, alpha=0.12, lw=0)
        labels.append((v[-1].median(), RUNG_LABELS[r]))
    P.day_axis(ax, [g for g in (1, 3, 7, 30, 120, 480) if gaps[0] <= g <= gaps[-1]])
    ax.set_ylim(bottom=max(-0.3, ax.get_ylim()[0]))
    P.end_labels(ax, gaps[-1] * 1.12, labels)
    ax.set_xlabel("days between training and test session")
    ax.set_ylabel("decoding R² (median, IQR)")
    ax.set_title("A  What restores a drifting decoder")

    ax = axes[1]
    labels = []
    for r, c in zip(["L3", "L5", "L6p"], colors[2:]):
        frac = [((d[col[r]] - d["L2"]) / (d["own"] - d["L2"])).clip(-0.5, 1.5) for g in gaps
                for d in [df[df.gap_target == g]]]
        ax.plot(gaps, [f.median() for f in frac], color=c, marker="o", ms=3.5)
        labels.append((frac[-1].median(), RUNG_LABELS[r]))
    ax.axhline(1, color=P.INK2, lw=0.8, ls=":")
    P.day_axis(ax, [g for g in (1, 3, 7, 30, 120, 480) if gaps[0] <= g <= gaps[-1]])
    P.end_labels(ax, gaps[-1] * 1.12, labels)
    ax.set_xlabel("days between training and test session")
    ax.set_ylabel("share of drift loss recovered")
    ax.set_title("B  Loss recovered beyond renormalization")
    fig.subplots_adjust(wspace=0.9)
    P.save(fig, os.path.join(out, "fig_anatomy"))


def fig_efficiency(ladder_csv, out):
    df = pd.read_csv(ladder_csv)
    ns = sorted({int(c.split("_n")[1]) for c in df.columns if c.startswith("L6_n")})
    gaps = sorted(df.gap_target.unique())
    methods = [("L3", "channel gains (96 p)"), ("L5", "full input remap"), ("L6p", "ridge to prior"),
               ("L6", "retrain from scratch")]
    P.setup()
    fig, axes = plt.subplots(1, len(gaps), figsize=(1.9 * len(gaps) + 1.2, 2.6), sharey=True)
    for ax, g in zip(np.atleast_1d(axes), gaps):
        d = df[df.gap_target == g]
        for (m, lab), c in zip(methods, P.CAT):
            ax.plot(ns, [d[f"{m}_n{n}"].median() for n in ns], color=c, marker="o", ms=3)
            if g == gaps[-1]:
                ax.text(ns[-1] * 1.15, d[f"{m}_n{ns[-1]}"].median(), lab, fontsize=7, va="center")
        ax.axhline(d["L2"].median(), color=P.NEUTRAL, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns], fontsize=7)
        ax.minorticks_off()
        ax.set_title(f"{g} day{'s' if g > 1 else ''} later", fontweight="normal")
        ax.set_xlabel("labelled trials")
    np.atleast_1d(axes)[0].set_ylabel("decoding R² (median)")
    np.atleast_1d(axes)[0].text(ns[0], df["L2"].median(), "no labels (renorm)", fontsize=7, color=P.INK2, va="bottom")
    fig.suptitle("How much calibration data does each correction need?", x=0.02, ha="left", fontweight="bold", fontsize=10)
    P.save(fig, os.path.join(out, "fig_efficiency"))


def fig_health(table_csv, out, alive_hz=2.0):
    df = pd.read_csv(table_csv)
    daily = df.groupby(["day", "ch"])[["tc_rate", "sbp_mean", "imp", "tune_r2"]].mean().reset_index()
    tc = daily.pivot(index="day", columns="ch", values="tc_rate")
    P.setup()
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 6.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    ax = axes[0]
    order = np.argsort(np.nanmean(np.log(tc.values[:10] + 0.1), 0))
    im = ax.imshow(np.log10(tc.values[:, order].T + 0.1), aspect="auto", cmap="Blues", vmin=-1, vmax=1.5,
                   extent=[tc.index.min(), tc.index.max(), tc.shape[1], 0], interpolation="nearest")
    ax.grid(False)
    ax.set_ylabel("channel (sorted by\ninitial activity)")
    ax.set_title("A  Threshold-crossing rate per channel over 3.5 years")
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label("log10 rate (Hz)")
    ax = axes[1]
    n_act = (tc > alive_hz).sum(axis=1)
    dlog = np.log(tc + 0.1).diff()
    glob = dlog.median(axis=1).abs() > 1                     # session-wide jumps (all channels at once)
    ax.plot(tc.index[~glob], n_act[~glob], ".", color=P.CAT[0], ms=3)
    ax.plot(tc.index[glob], n_act[glob], "x", color=P.INK2, ms=5, mew=1)
    ax.set_ylabel(f"channels > {alive_hz:g} Hz")
    ax.set_title("B  Active channels (× = session-wide event, all channels jump together)", fontweight="normal")
    ax = axes[2]
    imp = daily.pivot(index="day", columns="ch", values="imp")
    med = imp.median(axis=1)
    m = imp.notna().any(axis=1) & (med > 50e3)              # medians < 50 kOhm are failed measurements
    ax.plot(imp.index[m], med[m] / 1e3, ".", color=P.CAT[1], ms=3)
    ax.vlines(imp.index[m], imp[m].quantile(0.25, axis=1) / 1e3, imp[m].quantile(0.75, axis=1) / 1e3,
              color=P.CAT[1], alpha=0.25, lw=1)
    ax.set_ylabel("impedance (kΩ)")
    ax.set_xlabel("days since first session")
    ax.set_title("C  Electrode impedance (median, IQR; failed measurements removed)", fontweight="normal")
    P.save(fig, os.path.join(out, "fig_health"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["anatomy", "efficiency", "health"])
    ap.add_argument("--ladder")
    ap.add_argument("--table")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.which == "anatomy":
        fig_anatomy(args.ladder, args.out)
    elif args.which == "efficiency":
        fig_efficiency(args.ladder, args.out)
    else:
        fig_health(args.table, args.out)


if __name__ == "__main__":
    main()
