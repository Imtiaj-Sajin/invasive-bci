"""Shared figure style (validated reference palette from the dataviz guidelines; light mode for print).

- Categorical identity: fixed slot order, never cycled.
- Ordered series (e.g. oracle-ladder rungs, increasing correction power): one-hue blue ramp, light -> dark,
  starting no lighter than step 250 so every mark keeps >= 2:1 contrast on the surface.
- Thin marks, recessive grid/axes, text in ink colours (never the series colour), direct labels where few series.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_ORDINAL = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
                "#0d366b"]
NEUTRAL = "#8a8984"


def ordinal(n):
    """n ordered colours from the blue ramp (light -> dark), evenly spaced."""
    if n == 1:
        return [BLUE_ORDINAL[5]]
    idx = [round(i * (len(BLUE_ORDINAL) - 1) / (n - 1)) for i in range(n)]
    return [BLUE_ORDINAL[i] for i in idx]


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "lines.linewidth": 1.6, "lines.markersize": 5, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "legend.fontsize": 8, "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, path_stem):
    """Save PNG (for quick viewing) and PDF (vector, for the manuscript)."""
    fig.savefig(path_stem + ".png")
    fig.savefig(path_stem + ".pdf")
    plt.close(fig)


def end_labels(ax, x, items, min_gap_frac=0.06, fontsize=7, color=INK):
    """Place right-edge labels at the series' last y-values, nudged apart so they never overlap.

    items: list of (y, text). min_gap_frac is the minimum spacing as a fraction of the y-axis span.
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda t: t[0])
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = max(0.0, ys[-1] - (hi - gap / 2)) if ys else 0.0
    for (y0, text), y in zip(items, ys):
        ax.text(x, y - shift, text, fontsize=fontsize, va="center", color=color)


def day_axis(ax, ticks=(1, 3, 7, 30, 120, 480)):
    """Log-scaled x axis labelled with plain day counts."""
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
