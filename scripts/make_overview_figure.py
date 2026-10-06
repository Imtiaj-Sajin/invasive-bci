"""Figure 1: data overview (sessions per individual over years) and the analysis design.

Usage: python scripts/make_overview_figure.py [--out results/figures]
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci import plotting as P  # noqa: E402

GROUPS = [
    ("Decoder drift", ["monkey N (LINK)", "monkey C (000688)", "monkey M (000688)", "human T6 decoding"]),
    ("Electrode failure (humans)", None),
    ("Closed-loop monitoring", ["MINDFUL T11", "MINDFUL T5"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    sess = json.load(open("results/figures/overview_sessions.json"))
    bg = sorted([k for k in sess if k.startswith("BrainGate")], key=lambda k: -(sess[k][-1] - sess[k][0]))
    P.setup_nature()
    fig = plt.figure(figsize=(P.DOUBLE_COL, 3.3))
    ax = fig.add_axes([0.08, 0.1, 0.5, 0.82])
    rows, y = [], 0
    colors = {"Decoder drift": P.CAT[0], "Electrode failure (humans)": P.CAT[2], "Closed-loop monitoring": P.CAT[1]}
    for gname, keys in GROUPS:
        keys = bg if keys is None else keys
        for k in keys:
            days = sess[k]
            first = days[0]
            yrs = [(d - first) / 365.25 for d in days]
            ax.plot([yrs[0], yrs[-1]], [y, y], color=colors[gname], lw=0.6, alpha=0.5)
            ax.plot(yrs, [y] * len(yrs), "|", color=colors[gname], ms=3, mew=0.5)
            label = (k.replace(" (yield)", "").replace("BrainGate ", "").replace(" decoding", "")
                     .replace("MINDFUL ", ""))
            rows.append((y, f"{label}  (n={len(days)})"))
            y -= 1
        y -= 0.8
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=4.8)
    ax.set_xlabel("years since first session")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(y + 0.3, 1)
    for gname, _ in GROUPS:
        pass
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=c, lw=1.2) for c in colors.values()], list(colors.keys()), loc="lower right",
              title="analysis", title_fontsize=5.5)
    P.panel(ax, "a", x=-0.13)
    ax.set_title("2,833 sessions from 17 individuals", loc="left")

    # b: analysis design (oracle ladder)
    bx = fig.add_axes([0.66, 0.1, 0.32, 0.82])
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    bx.axis("off")
    bx.text(0, 0.97, "Decoder trained on day i, tested on later day j", fontsize=6, va="top")
    rungs = [("No correction", "none", P.NEUTRAL), ("Renormalize channels", "no labels", P.BLUE_ORDINAL[2]),
             ("Align neural subspace", "no labels", P.BLUE_ORDINAL[3]),
             ("Re-learn per-channel gains", "labels, 96 params", P.BLUE_ORDINAL[5]),
             ("Ridge shrunk to old decoder", "labels, few trials", P.BLUE_ORDINAL[7]),
             ("Fresh decoder on day j", "reference", P.INK2)]
    for k, (name, req, c) in enumerate(rungs):
        yy = 0.82 - k * 0.13
        bx.add_patch(FancyBboxPatch((0.02, yy - 0.045), 0.62, 0.09, boxstyle="round,pad=0.008,rounding_size=0.015",
                                    fc=c, ec="none", alpha=0.9 if k else 0.5))
        bx.text(0.04, yy, name, fontsize=5.5, va="center", color="white" if 1 <= k <= 4 else P.INK)
        bx.text(0.68, yy, req, fontsize=5, va="center", color=P.INK2)
    bx.annotate("", xy=(0.97, 0.13), xytext=(0.97, 0.84), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=P.INK2))
    bx.text(0.99, 0.48, "more correction", rotation=90, fontsize=5, va="center", ha="left", color=P.INK2)
    P.panel(bx, "b", x=-0.05)
    P.save(fig, os.path.join(args.out, "fig1_overview"))


if __name__ == "__main__":
    main()
