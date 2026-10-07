"""Manuscript v2 figures (Nature Communications style: 180 mm wide, Arial 5-7 pt).

  fig1_overview     data across six decoder-drift individuals, 20 human arrays and closed-loop monitoring; the ladder
  fig2_decay        decay of a renormalized fixed decoder in six individuals; strict 1-day vs 2-day; tuned penalty
  fig3_reweight     re-weighting each channel restores most accuracy in people (linear and network decoders),
                    matched-output control in monkeys, and how many labeled trials it needs
  fig4_labelfree    label-free corrections relative to renormalization; monitoring offline and in closed loop
  fig5_recipes      ridge penalty trade-off and the 1% rule; recalibration shrunk toward the old decoder
  fig6_electrodes   yield and impedance; electrode silence versus a shuffled null; edge effect per array
  fig7_simulator    calibrated simulator: fit, held-out validation against two baselines, implant age

Colours: one per individual (validated for colour-vision deficiency with the dataviz palette checker);
monkeys use circles and humans squares as a second encoding.

Usage: python scripts/make_v2_figures.py [--only fig2 ...] [--out results/figures/v2]
"""
import argparse
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from ibci import plotting as P  # noqa: E402
from ibci.stats import cluster_bootstrap  # noqa: E402

GAPS = [1, 7, 30, 120, 480]
IND = [("N", "Monkey N", "results/anatomy/ladder.csv", "#2a78d6", "o"),
       ("C", "Monkey C", "results/replication/C_ladder.csv", "#eb6834", "o"),
       ("M", "Monkey M", "results/replication/M_ladder.csv", "#1baf7a", "o"),
       ("T6", "Human T6", "results/replication_bg/T6_ladder.csv", "#4a3aa7", "s"),
       ("T5", "Human T5", "results/replication_bg/T5_ladder.csv", "#eda100", "s"),
       ("T9", "Human T9", "results/replication_bg/T9_ladder.csv", "#e87ba4", "s")]
IND = [i for i in IND if os.path.exists(i[2])]
COL = {k: c for k, _, _, c, _ in IND}
MK = {k: m for k, _, _, _, m in IND}
NAME = {k: n for k, n, _, _, _ in IND}
RET = "retention (R² / fresh R²)"


def boot_curve(d, num, den="own", gaps=GAPS, n_boot=800):
    d = d.assign(_r=d[num] / d[den]) if den else d.assign(_r=d[num])
    return np.array([cluster_bootstrap(d[d.gap_target == g], "_r", n_boot=n_boot) if (d.gap_target == g).sum() >= 3
                     else (np.nan, np.nan, np.nan) for g in gaps])


def line_ci(ax, x, est, color, label=None, marker="o", ls="-"):
    ax.plot(x, est[:, 0], color=color, marker=marker, ms=2.4, label=label, ls=ls, lw=1)
    ax.fill_between(x, est[:, 1], est[:, 2], color=color, alpha=0.13, lw=0)


def ref_lines(ax):
    ax.axhline(1, color=P.INK2, ls=":", lw=0.6)
    ax.axhline(0, color=P.INK2, lw=0.4)


def ladders():
    return {k: pd.read_csv(p)[lambda x: x.gap_target.isin(GAPS)] for k, _, p, _, _ in IND}


# ------------------------------------------------------------------------------------------------ Figure 1
def fig1(out):
    sess = json.load(open("results/figures/overview_sessions.json"))
    for p in ("T5", "T9"):
        root = {"T5": "D:/ibci-data/braingate/decoding/T5", "T9": "G:/ibci-data/braingate/decoding/T9"}[p]
        days = sorted(int(re.search(r"_day_(\d+)_", f).group(1)) for f in glob.glob(os.path.join(root, "*_decoding.mat")))
        if days:
            sess[f"human {p} decoding"] = days
    json.dump(sess, open("results/figures/overview_sessions.json", "w"))
    groups = [("Decoder drift", ["monkey N (LINK)", "monkey C (000688)", "monkey M (000688)", "human T6 decoding",
                                 "human T5 decoding", "human T9 decoding"]),
              ("Electrode statistics", sorted([k for k in sess if k.startswith("BrainGate")],
                                              key=lambda k: -(sess[k][-1] - sess[k][0]))),
              ("Closed-loop monitoring", ["MINDFUL T11", "MINDFUL T5"])]
    colors = {"Decoder drift": P.CAT[0], "Electrode statistics": "#1baf7a", "Closed-loop monitoring": "#eb6834"}
    P.setup_nature()
    fig = plt.figure(figsize=(P.DOUBLE_COL, 3.4))
    ax = fig.add_axes([0.085, 0.1, 0.5, 0.82])
    rows, y = [], 0
    for gname, keys in groups:
        for k in keys:
            if k not in sess:
                continue
            days = sess[k]
            yrs = [(d - days[0]) / 365.25 for d in days]
            ax.plot([yrs[0], yrs[-1]], [y, y], color=colors[gname], lw=0.6, alpha=0.5)
            ax.plot(yrs, [y] * len(yrs), "|", color=colors[gname], ms=3, mew=0.5)
            lab = k.replace(" (yield)", "").replace("BrainGate ", "").replace(" decoding", "").replace("MINDFUL ", "")
            rows.append((y, f"{lab}  (n={len(days)})"))
            y -= 1
        y -= 0.8
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=4.6)
    ax.set_xlabel("years since first session")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(y + 0.3, 1)
    ax.legend([Line2D([], [], color=c, lw=1.2) for c in colors.values()], list(colors), loc="lower right",
              title="analysis", title_fontsize=5.5)
    ax.set_title("17 individuals, up to 7.3 years per array", loc="left")
    P.panel(ax, "a", x=-0.14)
    bx = fig.add_axes([0.66, 0.1, 0.32, 0.82])
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    bx.axis("off")
    bx.text(0, 0.97, "Decoder trained on day i, tested on later day j", fontsize=6, va="top")
    rungs = [("No correction", "none", P.NEUTRAL), ("Rescale each channel", "no labels", P.BLUE_ORDINAL[2]),
             ("Align neural activity", "no labels", P.BLUE_ORDINAL[3]),
             ("Re-weight each channel", "labels, 1 per channel", P.BLUE_ORDINAL[5]),
             ("Refit near old decoder", "labels, few trials", P.BLUE_ORDINAL[7]),
             ("Reference decoder (day j)", "reference", P.INK2)]
    for k, (name, req, c) in enumerate(rungs):
        yy = 0.82 - k * 0.13
        bx.add_patch(FancyBboxPatch((0.02, yy - 0.045), 0.62, 0.09, boxstyle="round,pad=0.008,rounding_size=0.015",
                                    fc=c, ec="none", alpha=0.9 if k else 0.5))
        bx.text(0.04, yy, name, fontsize=5.5, va="center", color="white" if 1 <= k <= 4 else P.INK)
        bx.text(0.68, yy, req, fontsize=5, va="center", color=P.INK2)
    bx.annotate("", xy=(0.97, 0.13), xytext=(0.97, 0.84), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=P.INK2))
    bx.text(0.99, 0.48, "more correction", rotation=90, fontsize=5, va="center", ha="left", color=P.INK2)
    P.panel(bx, "b", x=-0.05)
    P.save(fig, os.path.join(out, "fig1_overview"))


# ------------------------------------------------------------------------------------------------ Figure 2
def fig2(out):
    P.setup_nature()
    L = ladders()
    fig, axes = plt.subplots(1, 4, figsize=(P.DOUBLE_COL, 2.0), gridspec_kw={"width_ratios": [1.45, 0.8, 0.8, 1.45]})
    ax = axes[0]
    for k in L:
        line_ci(ax, GAPS, boot_curve(L[k], "L2"), COL[k], NAME[k], MK[k], "-" if MK[k] == "o" else "--")
    ref_lines(ax)
    P.day_axis(ax, GAPS)
    ax.set_ylim(-0.7, 1.1)
    ax.set_xlabel("days since decoder training")
    ax.set_ylabel(RET)
    ax.set_title("Renormalized fixed decoder", loc="left")
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=4.8)
    P.panel(ax, "a", x=-0.26)

    ax = axes[1]
    for i, k in enumerate(L):
        one = L[k][L[k].gap_target == 1].assign(r=lambda x: x.L2 / x.own)
        for dd, off, filled in ((1, -0.13, True), (2, 0.13, False)):
            q = one[one.days == dd]
            if len(q) >= 3:
                est, lo, hi = cluster_bootstrap(q, "r", n_boot=1500)
                ax.plot([lo, hi], [i + off] * 2, color=COL[k], lw=0.8)
                ax.plot(est, i + off, MK[k], color=COL[k], ms=3, mfc=COL[k] if filled else "white", mew=0.8)
                ax.text(1.03, i + off, f"{len(q)}", fontsize=4.2, va="center", color=P.INK2,
                        transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(L)))
    ax.set_yticklabels([NAME[k] for k in L])
    ax.invert_yaxis()
    ax.set_xlim(-0.1, 1.05)
    ax.axvline(1, color=P.INK2, ls=":", lw=0.6)
    ax.set_xlabel("retention")
    ax.set_title("After 1 day (filled), 2 (open)", loc="left", fontsize=6)
    ax.grid(axis="y", visible=False)
    P.panel(ax, "b", x=-0.62)

    ax = axes[2]
    summ = json.load(open("results/v2_summary.json"))
    for i, k in enumerate(L):
        d0 = summ.get(k, {}).get("default", {}).get("t_half_days")
        d1 = summ.get(k, {}).get("tuned", {}).get("t_half_days")
        if d0 is None:
            continue
        ax.plot(d0, i, MK[k], color=COL[k], ms=3, mfc="white", mew=0.8)
        if d1 is not None and np.isfinite(d1):
            ax.annotate("", xy=(d1, i), xytext=(d0, i), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=COL[k]))
            ax.plot(d1, i, MK[k], color=COL[k], ms=3)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 400)
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    ax.minorticks_off()
    ax.set_yticks(range(len(L)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("days to half retention")
    ax.set_title("Default (open) → tuned", loc="left", fontsize=6)
    ax.grid(axis="y", visible=False)
    P.panel(ax, "c", x=-0.1)

    ax = axes[3]
    for k in L:
        tp = f"results/decay_alpha/{k}_alpha10000.csv"
        if os.path.exists(tp):
            line_ci(ax, GAPS, boot_curve(pd.read_csv(tp), "L2"), COL[k], NAME[k], MK[k], "-" if MK[k] == "o" else "--")
    ref_lines(ax)
    P.day_axis(ax, GAPS)
    ax.set_ylim(-0.7, 1.1)
    ax.set_xlabel("days since decoder training")
    ax.set_title("Same, with the tuned ridge penalty", loc="left")
    P.panel(ax, "d", x=-0.12)
    fig.subplots_adjust(wspace=0.5, left=0.065, right=0.99, bottom=0.2, top=0.86)
    P.save(fig, os.path.join(out, "fig2_decay"))


# ------------------------------------------------------------------------------------------------ Figure 3
def tuned_ladders():
    """Correction ladders with the tuned ridge penalty (alpha = 1e4) for day-i and reference decoders."""
    out = {}
    for k, _, _, _, _ in IND:
        p = f"results/decay_alpha/{k}_alpha10000.csv"
        if os.path.exists(p):
            out[k] = pd.read_csv(p)[lambda x: x.gap_target.isin(GAPS)]
    return out


def fig4c(out):
    """Corrections work best when channels keep their identity (figure 4 of the manuscript)."""
    P.setup_nature()
    L = tuned_ladders()
    fig, axes = plt.subplots(1, 4, figsize=(P.DOUBLE_COL, 2.05), gridspec_kw={"width_ratios": [1.3, 1.1, 0.9, 1.0]})
    ax = axes[0]
    for k in L:
        line_ci(ax, GAPS, boot_curve(L[k], "L3_n300"), COL[k], NAME[k], MK[k], "-" if MK[k] == "o" else "--")
    ref_lines(ax)
    P.day_axis(ax, GAPS)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("days since decoder training")
    ax.set_ylabel(RET)
    ax.set_title("After re-weighting each channel", loc="left")
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=4.6)
    P.panel(ax, "a", x=-0.25)

    ax = axes[1]   # correct versus scrambled channel assignment
    for i, k in enumerate(L):
        f = f"results/reweight_controls/{k}.csv"
        if not os.path.exists(f):
            continue
        q = pd.read_csv(f)
        med = {c: cluster_bootstrap(q.assign(_r=q[c] / q.own), "_r", n_boot=800) for c in
               ("renorm", "rew", "rew_perm", "rew_nonneg", "rew_perm_nn")}
        for off, (tc, pc) in ((-0.15, ("rew", "rew_perm")), (0.15, ("rew_nonneg", "rew_perm_nn"))):
            y = i + off
            ax.plot([med[pc][0], med[tc][0]], [y, y], color=COL[k], lw=0.6, alpha=0.6)
            ax.plot(med[tc][0], y, MK[k], color=COL[k], ms=3 if off < 0 else 2.6, mfc=COL[k] if off < 0 else "white",
                    mew=0.8)
            ax.plot(med[pc][0], y, "x", color=COL[k], ms=3, mew=0.8)
        ax.plot(med["renorm"][0], i, "|", color=P.INK2, ms=6, mew=0.8)
    ax.set_yticks(range(len(L)))
    ax.set_yticklabels([NAME[k] for k in L])
    ax.invert_yaxis()
    ax.axvline(0, color=P.INK2, lw=0.4)
    ax.set_xlim(-0.35, 1.05)
    ax.set_xlabel(RET)
    ax.set_title("Correct vs scrambled channels", loc="left", fontsize=6)
    ax.legend([Line2D([], [], marker="o", color=P.INK2, ls="", ms=3), Line2D([], [], marker="o", color=P.INK2, ls="",
                                                                              ms=3, mfc="white"),
               Line2D([], [], marker="x", color=P.INK2, ls="", ms=3), Line2D([], [], marker="|", color=P.INK2, ls="",
                                                                             ms=6)],
              ["signed weights", "non-negative", "scrambled", "no labels"], loc="lower left", frameon=False,
              fontsize=4.2, handletextpad=0.2)
    ax.grid(axis="y", visible=False)
    P.panel(ax, "b", x=-0.45)

    ax = axes[2]   # fitted weights track electrode firing-rate changes (humans)
    hum = [k for k in ("T6", "T5", "T9") if os.path.exists(f"results/gain_mechanism/{k}.csv")]
    for i, k in enumerate(hum):
        q = pd.read_csv(f"results/gain_mechanism/{k}.csv")
        s_ = q.groupby("train").rho_rate.mean().dropna()
        nl = q.groupby("train").rho_rate_null.mean().dropna()
        jit = np.random.default_rng(i).uniform(-0.12, 0.12, len(s_))
        ax.plot(i + jit - 0.18, s_, MK[k], color=COL[k], ms=1.8, alpha=0.7)
        ax.plot(i + np.random.default_rng(10 + i).uniform(-0.12, 0.12, len(nl)) + 0.18, nl, ".", color=P.NEUTRAL,
                ms=2, alpha=0.7)
        ax.plot([i - 0.32, i - 0.04], [s_.median()] * 2, color=P.INK, lw=1)
        ax.plot([i + 0.04, i + 0.32], [nl.median()] * 2, color=P.INK2, lw=1)
    ax.axhline(0, color=P.INK2, lw=0.5)
    ax.set_xticks(range(len(hum)))
    ax.set_xticklabels(hum)
    ax.set_ylabel("Spearman ρ, weight vs\nfiring-rate change")
    ax.set_title("Weights track electrodes", loc="left", fontsize=6)
    ax.legend([Line2D([], [], marker="s", color=P.INK2, ls="", ms=2.5), Line2D([], [], marker=".", color=P.NEUTRAL,
                                                                                ls="", ms=3)],
              ["observed", "channels shuffled"], loc="upper left", frameon=False, fontsize=4.2)
    ax.grid(axis="x", visible=False)
    P.panel(ax, "c", x=-0.42)

    ax = axes[3]   # share of loss recovered: linear vs network
    for i, k in enumerate(L):
        d = L[k].assign(rec=lambda x: (x.L3_n300 - x.L2) / (x.own - x.L2))
        est, lo, hi = cluster_bootstrap(d, "rec", n_boot=1200)
        ax.plot([lo, hi], [i - 0.13] * 2, color=COL[k], lw=0.8)
        ax.plot(est, i - 0.13, MK[k], color=COL[k], ms=3)
        nn = f"results/nn_ladder/{k}.csv"
        if os.path.exists(nn):
            q = pd.read_csv(nn).assign(rec=lambda x: (x.L3 - x.L2) / (x.own - x.L2))
            est, lo, hi = cluster_bootstrap(q, "rec", n_boot=1200)
            ax.plot([lo, hi], [i + 0.13] * 2, color=COL[k], lw=0.8)
            ax.plot(est, i + 0.13, MK[k], color=COL[k], ms=3, mfc="white", mew=0.8)
    ax.set_yticks(range(len(L)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(-0.1, 1.15)
    ax.set_xlabel("share of loss recovered")
    ax.set_title("Linear (filled), network (open)", loc="left", fontsize=6)
    ax.grid(axis="y", visible=False)
    P.panel(ax, "d", x=-0.1)
    fig.subplots_adjust(wspace=0.5, left=0.065, right=0.99, bottom=0.2, top=0.86)
    P.save(fig, os.path.join(out, "fig4_channels"))


# ------------------------------------------------------------------------------------------------ Figure 4
def fig3(out):
    P.setup_nature()
    fig, axes = plt.subplots(1, 3, figsize=(P.DOUBLE_COL, 2.1), gridspec_kw={"width_ratios": [1.6, 0.9, 0.9]})
    ax = axes[0]
    methods = [("mean_only", "align", "update means only"), ("mean_plus_align", "align", "means + rotation"),
               ("renorm_plus_align", "align", "rescale + rotation"), ("coral", "lfp", "full covariance (CORAL)"),
               ("fa_stab", "lfp", "factor-analysis stabilizer")]
    xs = np.arange(len(methods))
    for j, k in enumerate(IND):
        k = k[0]
        for m_i, (col, src, _) in enumerate(methods):
            f = f"results/{'align_variants' if src == 'align' else 'label_free_plus'}/{k}.csv"
            if not os.path.exists(f):
                continue
            q = pd.read_csv(f)
            s = q.groupby("train")[[col, "renorm"]].mean()
            diff = (s[col] - s["renorm"]).rename("d").reset_index()
            est, lo, hi = cluster_bootstrap(diff.assign(train=diff.train), "d", n_boot=800)
            x = m_i + (j - (len(IND) - 1) / 2) * 0.11
            ylo, yhi = -0.3, 0.1
            if est < ylo:   # off-scale: arrow at the bottom with the value
                ax.annotate(f"{est:.1f}", xy=(x, ylo + 0.01), xytext=(x, ylo + 0.06 + 0.035 * (j % 2)), fontsize=3.8, ha="center",
                            color=COL[k], arrowprops=dict(arrowstyle="-|>", lw=0.5, color=COL[k]))
                continue
            ax.plot([x, x], [max(lo, ylo), min(hi, yhi)], color=COL[k], lw=0.7)
            ax.plot(x, est, MK[k], color=COL[k], ms=2.6, label=NAME[k] if m_i == 3 else None)
    ax.axhline(0, color=P.INK2, lw=0.6)
    ax.set_ylim(-0.3, 0.1)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[2].replace(" + ", "\n+ ").replace(" (CORAL)", "\n(CORAL)")
                        .replace("factor-analysis ", "factor-analysis\n") for m in methods], fontsize=4.8)
    ax.set_ylabel("R² change vs rescaling each channel")
    ax.set_title("Label-free corrections (all gaps)", loc="left")
    ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=4.2)
    ax.grid(axis="x", visible=False)
    P.panel(ax, "a", x=-0.12)

    ax = axes[1]
    t = json.load(open("results/mindful_test/summary.json"))
    labs = ["neural", "output", "combined"]
    raw = [abs(t[k]["raw_spearman_with_r2"]) for k in ("kl_neural", "kl_output", "kl_sum")]
    par = [abs(t[k]["partial_given_logdays"]) for k in ("kl_neural", "kl_output", "kl_sum")]
    x = np.arange(3)
    ax.bar(x - 0.18, raw, 0.34, color=P.BLUE_ORDINAL[2], label="raw")
    ax.bar(x + 0.18, par, 0.34, color=P.BLUE_ORDINAL[7], label="given elapsed days")
    ax.axhline(abs(t["spearman_r2_logdays"]), color=P.INK2, ls="--", lw=0.7)
    ax.text(2.4, abs(t["spearman_r2_logdays"]) + 0.015, "elapsed days", fontsize=4.6, ha="right", color=P.INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("|Spearman ρ| with decoder R²")
    ax.set_title("Divergence, offline (monkey N)", loc="left", fontsize=6)
    ax.legend(loc="upper left", frameon=False, fontsize=4.6, bbox_to_anchor=(0, 0.93))
    ax.grid(axis="x", visible=False)
    P.panel(ax, "b", x=-0.3)

    ax = axes[2]
    s = json.load(open("results/mindful_reanalysis/summary.json"))
    x = np.arange(len(s))
    keys = [("days", "days", P.NEUTRAL), ("days+kl_neural", "+ neural", P.BLUE_ORDINAL[4]),
            ("days+kl_output", "+ output", "#eb6834")]
    for i, (kk, lab, c) in enumerate(keys):
        ax.bar(x + (i - 1) * 0.26, [r["lodo_mae"][kk] for r in s], 0.24, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([r["participant"] for r in s])
    ax.set_ylabel("prediction error of angular error (°)")
    ax.set_title("Divergence, closed loop", loc="left", fontsize=6)
    ax.legend(loc="upper left", frameon=False, fontsize=4.6)
    ax.grid(axis="x", visible=False)
    P.panel(ax, "c", x=-0.3)
    fig.subplots_adjust(wspace=0.42, left=0.06, right=0.99, bottom=0.2, top=0.86)
    P.save(fig, os.path.join(out, "fig3_labelfree"))


# ------------------------------------------------------------------------------------------------ Figure 5
def fig5(out):
    P.setup_nature()
    fig, axes = plt.subplots(1, 4, figsize=(P.DOUBLE_COL, 2.0), gridspec_kw={"width_ratios": [1.1, 1.2, 1.0, 0.9]})
    ax = axes[0]
    d = pd.read_csv("results/reg_tradeoff/tradeoff.csv")
    med = d.groupby(["gap_target", "alpha"]).r2.median().unstack(0)
    cols = P.ordinal(len(med.columns))
    for c, g in zip(cols, med.columns):
        ax.plot(med.index, med[g], "o-", color=c, ms=2, lw=0.9, label="same day" if g == 0 else f"{g} d")
    ax.axvline(0.1, color=P.INK2, ls=":", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("ridge penalty α")
    ax.set_ylabel("decoding R² (median)")
    ax.set_title("Monkey N", loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=4.2, ncol=2)
    P.panel(ax, "a", x=-0.3)
    ax = axes[1]
    t = pd.read_csv("results/reg_tradeoff/rule_across_subjects.csv")
    names = {"N (LINK)": "N", "C": "C", "M": "M", "T6 (human)": "T6", "T5 (human)": "T5", "T9 (human)": "T9"}
    keys = [k for k in names if (t.subject == k).any() and names[k] in COL]
    offs = np.linspace(-0.12, 0.12, len(keys))
    for o, key in zip(offs, keys):
        k = names[key]
        s = t[t.subject == key]
        xs = np.log10([max(g, 0.3) for g in s.gap]) + o
        lo = [float(v.strip("[]").split(",")[0]) for v in s.ci]
        hi = [float(v.strip("[]").split(",")[1]) for v in s.ci]
        for x, l_, h_ in zip(xs, lo, hi):
            ax.plot([x, x], [l_, h_], color=COL[k], lw=0.7)
        ax.plot(xs, s.gain, MK[k], color=COL[k], ms=2.4, label=NAME[k])
    ax.axhline(0, color=P.INK2, lw=0.5)
    ax.set_xticks(np.log10([0.3, 1, 7, 30, 120, 480]))
    ax.set_xticklabels(["same\nday", "1", "7", "30", "120", "480"])
    ax.set_xlabel("days since decoder training")
    ax.set_ylabel("R² gain over α = 0.1")
    ax.set_title("Tuned penalty (1% rule)", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=4.4, ncol=2)
    P.panel(ax, "b", x=-0.25)
    ax = axes[2]
    ns = [10, 20, 50, 100, 300]
    e = pd.concat([pd.read_csv(f"results/gain_efficiency/{k}.csv") for k in ("T6", "T5", "T9")
                   if os.path.exists(f"results/gain_efficiency/{k}.csv")])
    e = e.assign(train=e.subject + "_" + e.train.astype(str))
    for m, name, c in (("L6", "retrained on day j", P.NEUTRAL), ("L6p", "refit near old decoder", P.INK2),
                       ("L3", "re-weight channels", P.BLUE_ORDINAL[5])):
        est = np.array([cluster_bootstrap(e.dropna(subset=[f"{m}_n{n}"]).assign(_r=lambda x, c_=f"{m}_n{n}": x[c_] / x.own),
                                          "_r", n_boot=800) for n in ns])
        line_ci(ax, ns, est, c, name)
    ax.axhline(float((e.L2 / e.own).median()), color=P.NEUTRAL, ls=":", lw=0.7)
    ax.text(12, float((e.L2 / e.own).median()) + 0.03, "no labels", fontsize=4.4, color=P.INK2)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("labeled trials on day j")
    ax.set_ylabel(RET)
    ax.set_title("Three humans, all gaps", loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=4.4)
    P.panel(ax, "c", x=-0.3)
    ax = axes[3]
    p = pd.read_csv("results/recal_policy/policies.csv")
    rates = p.groupby("n").apply(lambda q: pd.Series({c: (q[c] < q.renorm - 0.01).mean() * 100 for c in ("cv", "history")}))
    x = np.arange(len(rates))
    ax.bar(x - 0.18, rates.cv, 0.34, color=P.NEUTRAL, label="strength by cross-validation")
    ax.bar(x + 0.18, rates.history, 0.34, color=P.CAT[0], label="fixed strong shrinkage")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in rates.index])
    ax.set_xlabel("labeled trials on day j")
    ax.set_ylabel("worse than no recalibration (%)")
    ax.set_title("Harmful recalibrations", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=4.4)
    ax.grid(axis="x", visible=False)
    P.panel(ax, "d", x=-0.32)
    fig.subplots_adjust(wspace=0.5, left=0.06, right=0.99, bottom=0.22, top=0.86)
    P.save(fig, os.path.join(out, "fig5_recipes"))


# ------------------------------------------------------------------------------------------------ Figure 6
def fig6(out):
    P.setup_nature()
    bg = pd.read_csv("results/braingate_failure/electrode_session.csv")
    ln = pd.read_csv("results/channel_health/channel_day.csv")
    ln_daily = ln.groupby(["day", "ch"]).agg(rate=("tc_rate", "mean"), imp=("imp", "mean")).reset_index()
    lay = ln.drop_duplicates("ch").set_index("ch")[["array_name", "row", "col"]]
    H, M = "#4a3aa7", "#2a78d6"
    fig, axes = plt.subplots(1, 4, figsize=(P.DOUBLE_COL, 2.2), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0, 1.1]})
    ax = axes[0]
    for _, g in bg.groupby(["participant", "array"]):
        y = g.groupby("day").rate.apply(lambda r: (r >= 2).mean() * 100)
        if len(y) >= 5:
            ax.plot(y.index / 365.25, y.rolling(5, min_periods=1, center=True).median(), color=H, lw=0.5, alpha=0.55)
    for _, g in ln_daily.merge(lay, left_on="ch", right_index=True).groupby("array_name"):
        y = g.groupby("day").rate.apply(lambda r: (r >= 2).mean() * 100)
        ax.plot((y.index + 30) / 365.25, y.rolling(5, min_periods=1, center=True).median(), color=M, lw=1)
    ax.set_xlabel("years since implant")
    ax.set_ylabel("electrodes with spiking (%)")
    ax.set_title("Array yield", loc="left")
    ax.legend([Line2D([], [], color=H, lw=1), Line2D([], [], color=M, lw=1)], ["humans, 20 arrays", "monkey N"],
              loc="upper right", frameon=False)
    P.panel(ax, "a", x=-0.25)
    ax = axes[1]
    for _, g in bg.groupby(["participant", "array"]):
        m = g[(g.impedance > 0) & (g.impedance < 4000)].groupby("day").impedance.median()
        if len(m) >= 5:
            ax.plot(m.index / 365.25, m / m.iloc[:3].median(), color=H, lw=0.5, alpha=0.55)
    m = ln_daily.groupby("day").imp.median().dropna()
    m = m[m > 50e3]
    ax.plot((m.index + 30) / 365.25, m / m.iloc[:3].median(), color=M, lw=1)
    ax.axhline(1, color=P.NEUTRAL, lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.3, 1, 3])
    ax.set_yticklabels(["0.1", "0.3", "1", "3"])
    ax.minorticks_off()
    ax.set_xlabel("years since implant")
    ax.set_ylabel("impedance relative to start")
    ax.set_title("Impedance", loc="left")
    P.panel(ax, "b", x=-0.25)
    ax = axes[2]
    r = json.load(open("results/failure_robustness/summary.json"))["revival"]
    groups = [("base", "shuffle_null", "≥3\nsessions"), ("fixed_uv", "shuffle_fixed_uv", "fixed\nµV"),
              ("deep", "shuffle_deep", "<0.5\nHz"), ("long_silence", "shuffle_long", "≥30\ndays")]
    x = np.arange(len(groups))
    nullf = "results/failure_robustness/null_summary.json"
    ns_ = json.load(open(nullf)) if os.path.exists(nullf) else {}
    obs = [r[a]["fraction"] * 100 for a, _, _ in groups]
    nul = [ns_[a]["fraction_null_mean"] * 100 if a in ns_ else r[b]["fraction"] * 100 for a, b, _ in groups]
    err = [[(ns_[a]["fraction_null_mean"] - ns_[a]["fraction_null_ci"][0]) * 100 if a in ns_ else 0 for a, _, _ in groups],
           [(ns_[a]["fraction_null_ci"][1] - ns_[a]["fraction_null_mean"]) * 100 if a in ns_ else 0 for a, _, _ in groups]]
    ax.bar(x - 0.18, obs, 0.34, color=H, label="observed")
    ax.bar(x + 0.18, nul, 0.34, color=P.NEUTRAL, label="shuffled (200×, 95% range)")
    ax.errorbar(x + 0.18, nul, yerr=err, fmt="none", ecolor=P.INK2, elinewidth=0.6, capsize=1.2)
    for xi, (a, b, _) in zip(x, groups):
        ax.text(xi - 0.18, obs[xi] + 1.5, f"{r[a]['silenced']}", fontsize=3.8, ha="center", va="bottom",
                rotation=90, color=P.INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([g[2] for g in groups], fontsize=4.6)
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("silence definition")
    ax.set_ylabel("silenced electrodes that recover (%)")
    ax.set_title("Silence then recovery", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=4.2)
    ax.grid(axis="x", visible=False)
    P.panel(ax, "c", x=-0.3)
    ax = axes[3]
    ea = pd.read_csv("results/failure_robustness/edge_arrays.csv").sort_values("edge_minus_interior")
    lab = (ea.participant + " " + ea.array.replace({"single": ""}).str.slice(0, 3)).tolist()
    y = np.arange(len(ea))
    ax.plot(ea.edge_minus_interior, y, "s", color=H, ms=2.6, label="edge − interior")
    ax.plot(ea.edge_coef_adj_initial_rate, y, "s", color=H, ms=2.6, mfc="white", mew=0.7,
            label="adjusted for initial rate")
    ax.axvline(0, color=P.INK2, lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=4.4)
    ax.set_xlabel("difference in decline\n(log rate per year)")
    e = json.load(open("results/failure_robustness/summary.json"))["edge"]["edge_minus_interior"]
    ax.set_title(f"Edge vs interior ({e['negative']}/{e['n_arrays']} faster)", loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=4.2)
    ax.grid(axis="y", visible=False)
    P.panel(ax, "d", x=-0.42)
    fig.subplots_adjust(wspace=0.62, left=0.06, right=0.99, bottom=0.22, top=0.87)
    P.save(fig, os.path.join(out, "fig6_electrodes"))


# ------------------------------------------------------------------------------------------------ Figure 7
def early_curve_baseline():
    """Median absolute error when the early-period (train day < 700) real medians predict the late-period ones."""
    v = pd.read_csv("results/sim_validation/validation.csv")
    v = v[v.kind == "sim"]
    a = pd.read_csv("results/anatomy/ladder.csv")
    e = pd.read_csv("results/data_efficiency/ladder.csv")
    errs = []
    for _, r in v.iterrows():
        src = e if re.search(r"_n(10|20|50|100)$", r.rung) else a
        if r.rung not in src:
            continue
        q = src[(src.train_day < 700) & (src.gap_target == r.gap)]
        if len(q) < 3:
            continue
        errs.append(dict(group=r.group, err=abs(float((q[r.rung] / q.own).median()) - r.real)))
    return pd.DataFrame(errs).groupby("group").err.median() if errs else None


def fig7(out):
    P.setup_nature()
    fig, axes = plt.subplots(1, 3, figsize=(P.DOUBLE_COL, 2.0), gridspec_kw={"width_ratios": [1.2, 1.2, 1.1]})
    e = json.load(open("results/sim_calib_split/calibration.json"))
    late = json.load(open("results/sim_calib_late/calibration.json"))
    gaps = [g for g in ["1", "7", "30", "120", "480"] if g in e["sim_fresh_seed"]]
    gx = [int(g) for g in gaps]
    ax = axes[0]
    for key, name, c in (("L2", "renormalized", P.CAT[0]), ("L3_n300", "re-weighted", P.BLUE_ORDINAL[7])):
        ax.plot(gx, [e["targets"][g][key] for g in gaps], "o", color=c, ms=3, label=f"real, {name}")
        ax.plot(gx, [e["sim_fresh_seed"][g][key] for g in gaps], "-", color=c, lw=1, label=f"simulated, {name}")
    ref_lines(ax)
    P.day_axis(ax, gx)
    ax.set_xlabel("days since decoder training")
    ax.set_ylabel(RET)
    ax.set_title("Calibration, monkey N (< 700 days)", loc="left", pad=12)
    ax.legend(loc="lower left", frameon=False, fontsize=4.6)
    P.panel(ax, "a", x=-0.22)
    ax = axes[1]
    v = pd.read_csv("results/sim_validation/validation.csv")
    s = v.groupby(["group", "kind"]).abs_err.median().unstack()
    base = early_curve_baseline()
    groups = [("targeted", "fitted rungs"), ("untargeted", "unfitted rungs"), ("efficiency", "unfitted data\nbudgets")]
    x = np.arange(len(groups))
    ax.bar(x - 0.27, [s.loc[g, "sim"] for g, _ in groups], 0.25, color=P.CAT[0], label="calibrated simulator")
    if base is not None:
        ax.bar(x, [base.get(g, np.nan) for g, _ in groups], 0.25, color=P.BLUE_ORDINAL[2], label="early-years curve")
    ax.bar(x + 0.27, [s.loc[g, "nodrift"] for g, _ in groups], 0.25, color=P.NEUTRAL, label="no drift")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in groups], fontsize=4.8)
    ax.set_ylabel("median absolute error (retention)")
    ax.set_ylim(0, 0.5)
    ax.set_title("Held-out years (≥ 700 days)", loc="left", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=4.4)
    ax.grid(axis="x", visible=False)
    P.panel(ax, "b", x=-0.22)
    ax = axes[2]
    keys = [("s_mix0", "session\nmixing"), ("s_mix", "slow\nmixing"), ("rho0", "initial\nturnover ×10"),
            ("tau_rho", "turnover\ntime (yr)")]
    ev = [e["params"]["s_mix0"], e["params"]["s_mix"], e["params"]["rho0"] * 10, e["params"]["tau_rho"] / 365.25]
    lv = [late["params"]["s_mix0"], late["params"]["s_mix"], late["params"]["rho0"] * 10, late["params"]["tau_rho"] / 365.25]
    x = np.arange(len(keys))
    ax.bar(x - 0.18, ev, 0.34, color=P.BLUE_ORDINAL[2], label="early (< 700 d)")
    ax.bar(x + 0.18, lv, 0.34, color=P.BLUE_ORDINAL[7], label="late (≥ 700 d)")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in keys], fontsize=4.8)
    ax.set_ylabel("fitted value")
    ax.set_title("Parameters by implant age", loc="left", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=4.6)
    ax.set_ylim(0, 1.75)
    ax.grid(axis="x", visible=False)
    P.panel(ax, "c", x=-0.2)
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.99, bottom=0.22, top=0.86)
    P.save(fig, os.path.join(out, "fig7_simulator"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="results/figures/v2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    figs = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4c, "fig5": fig5, "fig6": fig6, "fig7": fig7}
    for k, f in figs.items():
        if args.only is None or k in args.only:
            f(args.out)
            print(k, "saved", flush=True)


if __name__ == "__main__":
    main()
