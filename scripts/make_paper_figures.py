"""Main manuscript figures 2-7 in Nature Communications style (180 mm wide, Arial 5-7 pt).

  fig2_decay        conserved overnight drop, subject-specific decay, effect of re-learning channel gains
  fig3_labelfree    label-free alignment vs renormalization; neural vs decoder-output drift metrics
  fig4_regularize   regularizing for the future
  fig5_recalibrate  few-trial recalibration and how often it backfires
  fig6_failure      electrode failure across species
  fig7_simulator    calibrated simulator: fit, held-out validation, implant age

Usage: python scripts/make_paper_figures.py [--only fig2 fig3 ...] [--out results/figures/paper]
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from ibci import plotting as P  # noqa: E402
from ibci.stats import cluster_bootstrap  # noqa: E402

GAPS = [1, 7, 30, 120, 480]
SUBJ = [("Monkey N", "results/anatomy/ladder.csv", P.CAT[0]),
        ("Monkey C", "results/replication/C_ladder.csv", P.CAT[1]),
        ("Monkey M", "results/replication/M_ladder.csv", P.CAT[2]),
        ("Human T6", "results/replication_bg/T6_ladder.csv", P.CAT[3])]
# further BrainGate participants join automatically once their ladders exist
SUBJ += [(f"Human {p}", f"results/replication_bg/{p}_ladder.csv", P.CAT[c]) for p, c in (("T5", 4), ("T9", 6))
         if os.path.exists(f"results/replication_bg/{p}_ladder.csv")]


def boot_curve(d, col, gaps=GAPS, ref="own"):
    d = d.assign(_r=d[col] / d[ref]) if ref else d.assign(_r=d[col])
    est = [cluster_bootstrap(d[d.gap_target == g], "_r", n_boot=1000) if (d.gap_target == g).sum() >= 3
           else (np.nan, np.nan, np.nan) for g in gaps]
    return np.array(est)


def line_ci(ax, x, est, color, label=None, marker="o"):
    ax.plot(x, est[:, 0], color=color, marker=marker, ms=2.5, label=label)
    ax.fill_between(x, est[:, 1], est[:, 2], color=color, alpha=0.15, lw=0)


# ------------------------------------------------------------------------------------------------ Figure 2
def fig2(out):
    P.setup_nature()
    fig, axes = plt.subplots(1, 4, figsize=(P.DOUBLE_COL, 1.9), gridspec_kw={"width_ratios": [1.5, 0.7, 0.7, 1.5]})
    data = {n: pd.read_csv(p) for n, p, _ in SUBJ}
    ax = axes[0]
    for n, _, c in SUBJ:
        line_ci(ax, GAPS, boot_curve(data[n], "L2"), c, n)
    ax.axhline(1, color=P.INK2, ls=":", lw=0.6)
    ax.axhline(0, color=P.INK2, lw=0.4)
    P.day_axis(ax, GAPS)
    ax.set_xlabel("days since decoder training")
    ax.set_ylabel("R² relative to fresh decoder")
    ax.set_title("Renormalized fixed decoder", loc="left")
    ax.legend(loc="lower left", frameon=False)
    P.panel(ax, "a", x=-0.28)

    summ = pd.read_csv("results/decay_summary.csv")
    for ax, col, cicol, title, lab in ((axes[1], "overnight", "overnight_ci", "Kept after one day", "b"),
                                       (axes[2], "t_half_days", "t_half_ci", "Days to lose half", "c")):
        for k, (n, _, c) in enumerate(SUBJ):
            r = summ[summ.subject == n.lower()].iloc[0] if (summ.subject == n.lower()).any() else \
                summ[summ.subject.str.lower() == n.lower()].iloc[0]
            lo, hi = [float(v) for v in r[cicol].strip("[]").split(",")]
            ax.plot([lo, hi], [k, k], color=c, lw=1)
            ax.plot(r[col], k, "o", color=c, ms=3)
        ax.set_yticks(range(len(SUBJ)))
        ax.set_yticklabels([n for n, _, _ in SUBJ] if lab == "b" else [])
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        P.panel(ax, lab, x=-0.55 if lab == "b" else -0.12)
    axes[1].set_xlim(0.3, 1.0)
    axes[1].set_xlabel("retention")
    axes[2].set_xscale("log")
    axes[2].set_xticks([3, 10, 30, 100, 300])
    axes[2].set_xticklabels(["3", "10", "30", "100", "300"])
    axes[2].minorticks_off()
    axes[2].set_xlabel("days")

    ax = axes[3]
    for n, _, c in SUBJ:
        line_ci(ax, GAPS, boot_curve(data[n], "L3_n300"), c, n)
    ax.axhline(1, color=P.INK2, ls=":", lw=0.6)
    ax.axhline(0, color=P.INK2, lw=0.4)
    P.day_axis(ax, GAPS)
    ax.set_ylim(axes[0].get_ylim())
    ax.set_xlabel("days since decoder training")
    ax.set_title("After re-learning 96 channel gains", loc="left")
    P.panel(ax, "d", x=-0.12)
    fig.subplots_adjust(wspace=0.45, left=0.07, right=0.99, bottom=0.2, top=0.86)
    P.save(fig, os.path.join(out, "fig2_decay"))


# ------------------------------------------------------------------------------------------------ Figure 3
def fig3(out):
    P.setup_nature()
    fig, axes = plt.subplots(1, 4, figsize=(P.DOUBLE_COL, 1.9))
    rungs = [("L1", "Mean only", P.NEUTRAL), ("L4u", "Subspace alignment", P.CAT[6]), ("L2", "Full renormalization", P.CAT[0])]
    for ax, (n, path), lab in ((axes[0], ("Monkey N", "results/anatomy/ladder.csv"), "a"),
                               (axes[1], ("Human T6", "results/replication_bg/T6_ladder.csv"), "b")):
        d = pd.read_csv(path)
        for col, name, c in rungs:
            line_ci(ax, GAPS, boot_curve(d, col, ref=None), c, name)
        ax.axhline(0, color=P.INK2, lw=0.4)
        P.day_axis(ax, GAPS)
        ax.set_xlabel("days since decoder training")
        ax.set_title(n, loc="left")
        P.panel(ax, lab, x=-0.25 if lab == "a" else -0.15)
    axes[0].set_ylabel("decoding R² (median)")
    axes[0].legend(loc="lower left", frameon=False)

    ax = axes[2]
    m = json.load(open("results/mindful_test/summary.json"))
    feats = [("kl_neural", "neural drift"), ("kl_output", "output drift"), ("kl_sum", "combined")]
    raw = [abs(m[f]["raw_spearman_with_r2"]) for f, _ in feats]
    par = [abs(m[f]["partial_given_logdays"]) for f, _ in feats]
    x = np.arange(len(feats))
    ax.bar(x - 0.18, raw, 0.34, color=P.BLUE_ORDINAL[2], label="raw")
    ax.bar(x + 0.18, par, 0.34, color=P.BLUE_ORDINAL[7], label="after controlling for days")
    ax.axhline(abs(m["spearman_r2_logdays"]), color=P.INK2, ls="--", lw=0.6)
    ax.text(2.45, abs(m["spearman_r2_logdays"]) + 0.02, "days alone", fontsize=5, ha="right", color=P.INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f for _, f in feats])
    ax.set_ylabel("|Spearman ρ| with decoder R²")
    ax.set_title("Offline, monkey N", loc="left")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="x", visible=False)
    P.panel(ax, "c", x=-0.25)

    ax = axes[3]
    r = json.load(open("results/mindful_reanalysis/summary.json"))
    models = [("days", "days"), ("days+kl_neural", "+ neural drift"), ("days+kl_output", "+ output drift")]
    cols = [P.NEUTRAL, P.BLUE_ORDINAL[3], P.CAT[1]]
    for k, part in enumerate(r):
        for j, ((mk, _), c) in enumerate(zip(models, cols)):
            ax.bar(k + (j - 1) * 0.26, part["lodo_mae"][mk], 0.24, color=c, label=models[j][1] if k == 0 else None)
    ax.set_xticks(range(len(r)))
    ax.set_xticklabels([f"{p['participant']}" for p in r])
    ax.set_ylabel("prediction error of angular error (°)")
    ax.set_title("Closed loop, human data", loc="left")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="x", visible=False)
    P.panel(ax, "d", x=-0.25)
    fig.subplots_adjust(wspace=0.5, left=0.07, right=0.99, bottom=0.2, top=0.86)
    P.save(fig, os.path.join(out, "fig3_labelfree"))


# ------------------------------------------------------------------------------------------------ Figure 4
def fig4(out):
    P.setup_nature()
    fig, axes = plt.subplots(1, 3, figsize=(P.DOUBLE_COL, 2.3), gridspec_kw={"width_ratios": [1.2, 1.2, 1.3]})
    for ax, (path, title, lab) in zip(axes[:2], [("results/reg_tradeoff/tradeoff.csv", "Monkey N", "a"),
                                               ("results/reg_tradeoff_T6/tradeoff.csv", "Human T6", "b")]):
        d = pd.read_csv(path)
        tab = d.groupby(["gap_target", "alpha"]).r2.median().unstack()
        for g, c in zip(tab.index, P.ordinal(len(tab.index))):
            ax.plot(tab.columns, tab.loc[g], color=c, marker="o", ms=2,
                    label="same day" if g == 0 else f"{g} d later")
        ax.axvline(0.1, color=P.NEUTRAL, ls=":", lw=0.6)
        ax.axvline(1e4, color=P.NEUTRAL, ls="--", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("ridge regularization α")
        ax.set_title(title, loc="left")
        P.panel(ax, lab, x=-0.2)
    axes[0].set_ylabel("decoding R² (median)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.33, 0.0), ncol=6, frameon=False, fontsize=5,
               title="test session", title_fontsize=5)
    ax = axes[2]
    t = pd.read_csv("results/reg_tradeoff/rule_across_subjects.csv")
    names = {"N (LINK)": ("Monkey N", P.CAT[0]), "C": ("Monkey C", P.CAT[1]), "M": ("Monkey M", P.CAT[2]),
             "T6 (human)": ("Human T6", P.CAT[3]), "T5 (human)": ("Human T5", P.CAT[4]),
             "T9 (human)": ("Human T9", P.CAT[6])}
    names = {k: v for k, v in names.items() if (t.subject == k).any()}
    offs = {k: o for k, o in zip(names, np.linspace(-0.09, 0.09, len(names)) if len(names) > 1 else [0.0])}
    for key, (lab, c) in names.items():
        s = t[t.subject == key]
        xs = np.log10([max(g, 0.5) for g in s.gap]) + offs[key]
        lo = [float(v.strip("[]").split(",")[0]) for v in s.ci]
        hi = [float(v.strip("[]").split(",")[1]) for v in s.ci]
        for x, l, h in zip(xs, lo, hi):
            ax.plot([x, x], [l, h], color=c, lw=0.8)
        ax.plot(xs, s.gain, "o", color=c, ms=2.5, label=lab)
    ax.axhline(0, color=P.INK2, lw=0.5)
    ax.set_xticks(np.log10([0.5, 1, 7, 30, 120, 480]))
    ax.set_xticklabels(["same\nday", "1", "7", "30", "120", "480"])
    ax.set_xlabel("days since decoder training")
    ax.set_ylabel("R² gain over α = 0.1")
    ax.set_title("Gain from the 1% rule (median, 95% CI)", loc="left")
    ax.legend(loc="upper left", frameon=False)
    P.panel(ax, "c", x=-0.2)
    fig.subplots_adjust(wspace=0.35, left=0.07, right=0.99, bottom=0.33, top=0.88)
    P.save(fig, os.path.join(out, "fig4_regularize"))


# ------------------------------------------------------------------------------------------------ Figure 5
def fig5(out):
    P.setup_nature()
    fig, axes = plt.subplots(1, 3, figsize=(P.DOUBLE_COL, 1.95))
    d = pd.read_csv("results/data_efficiency/ladder.csv")
    ns = [10, 20, 50, 100, 300]
    methods = [("L6p", "ridge shrunk to old decoder", P.CAT[0]), ("L6", "retrain from scratch", P.CAT[1]),
               ("L3", "re-learn channel gains", P.CAT[2])]
    for ax, g, lab in ((axes[0], 7, "a"), (axes[1], 120, "b")):
        e = d[d.gap_target == g]
        for m, name, c in methods:
            est = np.array([cluster_bootstrap(e.assign(_r=e[f"{m}_n{n}"] / e.own), "_r", n_boot=800) for n in ns])
            line_ci(ax, ns, est, c, name)
        ax.axhline(float(np.median(e.L2 / e.own)), color=P.NEUTRAL, ls=":", lw=0.7)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.minorticks_off()
        ax.set_xlabel("labelled trials on the new day")
        ax.set_title(f"{g} days after training (monkey N)", loc="left")
        P.panel(ax, lab, x=-0.22)
    axes[0].set_ylabel("R² relative to fresh decoder")
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].text(140, float(np.median(d[d.gap_target == 7].L2 / d[d.gap_target == 7].own)) - 0.07, "no labels",
                 fontsize=5, color=P.INK2)
    ax = axes[2]
    p = pd.read_csv("results/recal_policy/policies.csv")
    rates = p.groupby("n").apply(lambda q: pd.Series({c: (q[c] < q.renorm - 0.01).mean() * 100 for c in ("cv", "history")}))
    x = np.arange(len(rates))
    ax.bar(x - 0.18, rates.cv, 0.34, color=P.NEUTRAL, label="strength by cross-validation")
    ax.bar(x + 0.18, rates.history, 0.34, color=P.CAT[0], label="strength from past sessions")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in rates.index])
    ax.set_xlabel("labelled trials on the new day")
    ax.set_ylabel("recalibrations worse than none (%)")
    ax.set_title("How often recalibration backfires", loc="left")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="x", visible=False)
    P.panel(ax, "c", x=-0.22)
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.99, bottom=0.21, top=0.86)
    P.save(fig, os.path.join(out, "fig5_recalibrate"))


# ------------------------------------------------------------------------------------------------ Figure 6
def fig6(out):
    import make_failure_figure as mf
    P.setup_nature()
    bg = pd.read_csv("results/braingate_failure/electrode_session.csv")
    pa = pd.read_csv("results/braingate_failure/per_array.csv")
    ln = pd.read_csv("results/channel_health/channel_day.csv")
    ln_daily = ln.groupby(["day", "ch"]).agg(rate=("tc_rate", "mean"), imp=("imp", "mean")).reset_index()
    lay = ln.drop_duplicates("ch").set_index("ch")[["array_name", "row", "col"]]
    H, M = P.CAT[0], P.CAT[1]
    fig, axes = plt.subplots(1, 4, figsize=(P.DOUBLE_COL, 2.3), gridspec_kw={"width_ratios": [1.1, 1.1, 0.9, 1.1]})
    ax = axes[0]
    for (pp, a), g in bg.groupby(["participant", "array"]):
        y = g.groupby("day").rate.apply(lambda r: (r >= 2).mean() * 100)
        if len(y) >= 5:
            ax.plot(y.index / 365.25, y.rolling(5, min_periods=1, center=True).median(), color=H, lw=0.5, alpha=0.6)
    for arr, g in ln_daily.merge(lay, left_on="ch", right_index=True).groupby("array_name"):
        y = g.groupby("day").rate.apply(lambda r: (r >= 2).mean() * 100)
        ax.plot((y.index + 30) / 365.25, y.rolling(5, min_periods=1, center=True).median(), color=M, lw=1)
    ax.set_xlabel("years since implant")
    ax.set_ylabel("electrodes with spiking (%)")
    ax.set_title("Array yield", loc="left")
    ax.legend([Line2D([], [], color=H, lw=1), Line2D([], [], color=M, lw=1)], ["humans, 20 arrays", "monkey N"],
              loc="upper right", frameon=False)
    P.panel(ax, "a", x=-0.25)
    ax = axes[1]
    for (pp, a), g in bg.groupby(["participant", "array"]):
        m = g[(g.impedance > 0) & (g.impedance < 4000)].groupby("day").impedance.median()
        if len(m) >= 5:
            ax.plot(m.index / 365.25, m / m.iloc[:3].median(), color=H, lw=0.5, alpha=0.6)
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
    q = pa[pa.silenced >= 5].copy()
    q["share"] = q.revived / q.silenced
    q["label"] = q.participant + " " + q.array.replace({"single": ""}).str.slice(0, 3)
    q = q.sort_values("share")
    ax.barh(range(len(q)), q.share * 100, color=H, height=0.65)
    ax.barh(len(q), 23 / 31 * 100, color=M, height=0.65)
    ax.set_yticks(list(range(len(q))) + [len(q)])
    ax.set_yticklabels(list(q.label) + ["monkey N"], fontsize=4.5)
    ax.axvline(584 / 730 * 100, color=P.INK2, ls="--", lw=0.6)
    ax.set_xlim(0, 105)
    ax.set_xlabel("silenced electrodes that recover (%)")
    ax.set_title("Transient silencing", loc="left")
    ax.grid(axis="y", visible=False)
    P.panel(ax, "c", x=-0.42)
    ax = axes[3]
    rows = []
    for (pp, a), g in bg.groupby(["participant", "array"]):
        if pa[(pa.participant == pp) & (pa.array == a)].initially_active.squeeze() < 10:
            continue
        xy = g.drop_duplicates("electrode_id").set_index("electrode_id")
        sl, edge = mf.array_slopes(g, edge_fn=lambda cols: (xy.loc[cols].x.isin([0, 9]) | xy.loc[cols].y.isin([0, 9])).values)
        if edge.sum() > 3 and (~edge).sum() > 3:
            rows.append((f"{pp} {a.replace('single', '')[:3]}", *mf.boot_diff(sl, edge), H))
    for arr in ("Medial", "Lateral"):
        g = ln_daily[ln_daily.ch.isin(lay.index[lay.array_name == arr])]
        sl, edge = mf.array_slopes(g, rate_col="rate", ch_col="ch",
                                   edge_fn=lambda cols: (lay.loc[cols].row.isin([0, 7]) | lay.loc[cols].col.isin([0, 7])).values)
        rows.append((f"monkey N {arr[:3].lower()}", *mf.boot_diff(sl, edge), M))
    rows.sort(key=lambda r: r[1])
    for k, (lab, dd, lo, hi, c) in enumerate(rows):
        ax.plot([lo, hi], [k, k], color=c, lw=0.8)
        ax.plot(dd, k, "o", color=c, ms=2.5)
    ax.axvline(0, color=P.INK2, lw=0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=4.5)
    ax.set_xlabel("edge minus interior decline\n(log rate per year)")
    neg = sum(r[1] < 0 for r in rows)
    ax.set_title(f"Edges decline faster ({neg}/{len(rows)})", loc="left")
    ax.grid(axis="y", visible=False)
    P.panel(ax, "d", x=-0.42)
    fig.subplots_adjust(wspace=0.55, left=0.06, right=0.99, bottom=0.22, top=0.87)
    P.save(fig, os.path.join(out, "fig6_failure"))


# ------------------------------------------------------------------------------------------------ Figure 7
def fig7(out):
    P.setup_nature()
    fig, axes = plt.subplots(1, 3, figsize=(P.DOUBLE_COL, 2.0), gridspec_kw={"width_ratios": [1.2, 1.1, 1.1]})
    e = json.load(open("results/sim_calib_split/calibration.json"))
    late = json.load(open("results/sim_calib_late/calibration.json"))
    gaps = [g for g in ["1", "7", "30", "120", "480"] if g in e["sim_fresh_seed"]]
    gx = [int(g) for g in gaps]
    ax = axes[0]
    for key, name, c in (("L2", "renormalized", P.CAT[0]), ("L3_n300", "+ channel gains", P.CAT[2])):
        ax.plot(gx, [e["targets"][g][key] for g in gaps], "o", color=c, ms=3, label=f"real, {name}")
        ax.plot(gx, [e["sim_fresh_seed"][g][key] for g in gaps], "-", color=c, lw=1, label=f"simulated, {name}")
    ax.axhline(0, color=P.INK2, lw=0.4)
    P.day_axis(ax, gx)
    ax.set_xlabel("days since decoder training")
    ax.set_ylabel("R² relative to fresh decoder")
    ax.set_title("Calibration (implant days < 700)", loc="left", pad=12)
    ax.legend(loc="lower left", frameon=False, fontsize=4.8)
    P.panel(ax, "a", x=-0.22)
    ax = axes[1]
    v = pd.read_csv("results/sim_validation/validation.csv")
    s = v.groupby(["kind", "group"]).abs_err.median().unstack(0)
    groups = [("targeted", "fitted rungs"), ("untargeted", "unfitted rungs"), ("efficiency", "unfitted data budgets")]
    x = np.arange(len(groups))
    ax.bar(x - 0.18, [s.loc[g, "sim"] for g, _ in groups], 0.34, color=P.CAT[0], label="calibrated simulator")
    ax.bar(x + 0.18, [s.loc[g, "nodrift"] for g, _ in groups], 0.34, color=P.NEUTRAL, label="no-drift reference")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in groups], fontsize=5)
    ax.set_ylabel("median absolute error")
    ax.set_title("Held-out years (days ≥ 700)", loc="left", pad=12)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.97), ncol=2, frameon=False, fontsize=4.8)
    ax.grid(axis="x", visible=False)
    P.panel(ax, "b", x=-0.22)
    ax = axes[2]
    keys = [("s_mix0", "session\nmixing"), ("s_mix", "slow\nmixing"), ("rho0", "initial\nturnover ×10"),
            ("tau_rho", "turnover\ntime (yr)")]
    ev = [e["params"]["s_mix0"], e["params"]["s_mix"], e["params"]["rho0"] * 10, e["params"]["tau_rho"] / 365.25]
    lv = [late["params"]["s_mix0"], late["params"]["s_mix"], late["params"]["rho0"] * 10, late["params"]["tau_rho"] / 365.25]
    x = np.arange(len(keys))
    ax.bar(x - 0.18, ev, 0.34, color=P.BLUE_ORDINAL[2], label="early implant (< 700 d)")
    ax.bar(x + 0.18, lv, 0.34, color=P.BLUE_ORDINAL[7], label="late implant (≥ 700 d)")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in keys], fontsize=4.8)
    ax.set_ylabel("fitted value")
    ax.set_title("Drift slows as the implant ages", loc="left", pad=12)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.97), ncol=2, frameon=False, fontsize=4.8)
    ax.set_ylim(0, 1.75)
    ax.grid(axis="x", visible=False)
    P.panel(ax, "c", x=-0.2)
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.99, bottom=0.22, top=0.86)
    P.save(fig, os.path.join(out, "fig7_simulator"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="results/figures/paper")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    figs = {"fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7}
    for k, f in figs.items():
        if args.only is None or k in args.only:
            f(args.out)
            print(k, "saved", flush=True)


if __name__ == "__main__":
    main()
