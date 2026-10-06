"""Cross-species electrode failure figure: 20 human Utah arrays (BrainGate) + monkey N (LINK).

A  yield (% electrodes >= 2 Hz at -4.5 RMS) vs years since implant
B  median impedance (normalised to each array's first measurement) vs years since implant
C  share of silenced electrodes that later revive, per array
D  edge - interior difference in activity decline (log-rate slope per year), per array, with bootstrap 95% CI

Usage: python scripts/make_failure_figure.py [--out results/figures]
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci import plotting as P  # noqa: E402

HUMAN, MONKEY = P.CAT[0], P.CAT[1]


def array_slopes(g, days_col="day", rate_col="rate", ch_col="electrode_id", edge_fn=None):
    piv = g.pivot_table(index=days_col, columns=ch_col, values=rate_col).sort_index()
    piv = piv.loc[:, piv.notna().mean() > 0.8].ffill().bfill()
    yrs = piv.index.values / 365.25
    lr = np.log(piv.values + 0.1)
    slopes = np.array([stats.linregress(yrs, lr[:, c]).slope for c in range(lr.shape[1])])
    return slopes, edge_fn(piv.columns)


def boot_diff(slopes, edge, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    e, i = slopes[edge], slopes[~edge]
    d = [np.median(rng.choice(e, len(e))) - np.median(rng.choice(i, len(i))) for _ in range(n)]
    return np.median(e) - np.median(i), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    bg = pd.read_csv("results/braingate_failure/electrode_session.csv")
    pa = pd.read_csv("results/braingate_failure/per_array.csv")
    ln = pd.read_csv("results/channel_health/channel_day.csv")
    ln_daily = ln.groupby(["day", "ch"]).agg(rate=("tc_rate", "mean"), imp=("imp", "mean")).reset_index()
    lay = ln.drop_duplicates("ch").set_index("ch")[["array_name", "row", "col"]]

    P.setup()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8))

    # A: yield trajectories
    ax = axes[0, 0]
    for (p, a), g in bg.groupby(["participant", "array"]):
        y = g.groupby("day").rate.apply(lambda r: (r >= 2).mean() * 100)
        if len(y) >= 5:
            ax.plot(y.index / 365.25, y.rolling(5, min_periods=1, center=True).median(), color=HUMAN, lw=0.8, alpha=0.6)
    for arr, g in ln_daily.merge(lay, left_on="ch", right_index=True).groupby("array_name"):
        y = g.groupby("day").rate.apply(lambda r: (r >= 2).mean() * 100)
        ax.plot((y.index + 30) / 365.25, y.rolling(5, min_periods=1, center=True).median(), color=MONKEY, lw=1.4)
    ax.set_xlabel("years since implant")
    ax.set_ylabel("electrodes ≥ 2 Hz (%)")
    ax.set_title("A  Array yield")
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=HUMAN, lw=1.4), Line2D([], [], color=MONKEY, lw=1.4)],
              ["humans (20 arrays)", "monkey N (2 arrays)"], loc="upper right")

    # B: impedance trajectories (normalised)
    ax = axes[0, 1]
    for (p, a), g in bg.groupby(["participant", "array"]):
        m = g[(g.impedance > 0) & (g.impedance < 4000)].groupby("day").impedance.median()
        if len(m) >= 5:
            ax.plot(m.index / 365.25, m / m.iloc[:3].median(), color=HUMAN, lw=0.8, alpha=0.6)
    m = ln_daily.groupby("day").imp.median().dropna()
    m = m[m > 50e3]
    ax.plot((m.index + 30) / 365.25, m / m.iloc[:3].median(), color=MONKEY, lw=1.4)
    ax.axhline(1, color=P.NEUTRAL, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.3, 1, 3])
    ax.set_yticklabels(["0.1", "0.3", "1", "3"])
    ax.set_xlabel("years since implant")
    ax.set_ylabel("impedance / initial")
    ax.set_title("B  Impedance falls")

    # C: revival share per array
    ax = axes[1, 0]
    q = pa[pa.silenced >= 5].copy()
    q["share"] = q.revived / q.silenced
    q["label"] = q.participant + " " + q.array.replace({"single": ""}).str.slice(0, 3)
    q = q.sort_values("share")
    ax.barh(range(len(q)), q.share, color=HUMAN, height=0.6)
    ax.barh(len(q), 23 / 31, color=MONKEY, height=0.6)
    ax.set_yticks(list(range(len(q))) + [len(q)])
    ax.set_yticklabels(list(q.label) + ["monkey N"], fontsize=6)
    ax.axvline(584 / 730, color=P.INK2, ls="--", lw=0.8)
    ax.text(584 / 730 + 0.01, 0.2, "pooled humans 0.80", fontsize=7, color=P.INK2)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("silenced electrodes that later revive")
    ax.set_title("C  Losses are mostly transient")
    ax.grid(axis="y", visible=False)

    # D: edge - interior forest plot
    ax = axes[1, 1]
    rows = []
    for (p, a), g in bg.groupby(["participant", "array"]):
        if pa[(pa.participant == p) & (pa.array == a)].initially_active.squeeze() < 10:
            continue
        xy = g.drop_duplicates("electrode_id").set_index("electrode_id")
        sl, edge = array_slopes(g, edge_fn=lambda cols: (xy.loc[cols].x.isin([0, 9]) | xy.loc[cols].y.isin([0, 9])).values)
        if edge.sum() > 3 and (~edge).sum() > 3:
            rows.append((f"{p} {a.replace('single', '')[:3]}", *boot_diff(sl, edge), HUMAN))
    for arr in ("Medial", "Lateral"):
        g = ln_daily[ln_daily.ch.isin(lay.index[lay.array_name == arr])]
        sl, edge = array_slopes(g, rate_col="rate", ch_col="ch",
                                edge_fn=lambda cols: (lay.loc[cols].row.isin([0, 7]) | lay.loc[cols].col.isin([0, 7])).values)
        rows.append((f"monkey N {arr[:3].lower()}", *boot_diff(sl, edge), MONKEY))
    rows.sort(key=lambda r: r[1])
    for k, (lab, d, lo, hi, c) in enumerate(rows):
        ax.plot([lo, hi], [k, k], color=c, lw=1.2)
        ax.plot(d, k, "o", color=c, ms=4)
    ax.axvline(0, color=P.INK2, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6)
    ax.set_xlabel("edge − interior decline (Δ log-rate per year)")
    neg = sum(r[1] < 0 for r in rows)
    ax.set_title(f"D  Edge electrodes decline faster ({neg}/{len(rows)} arrays)")
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    P.save(fig, os.path.join(args.out, "fig_failure_xspecies"))
    print(f"edge faster in {neg}/{len(rows)} arrays (point estimate); saved")


if __name__ == "__main__":
    main()
