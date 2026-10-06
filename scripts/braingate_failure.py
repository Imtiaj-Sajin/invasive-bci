"""Electrode failure statistics for 20 human Utah arrays (BrainGate 20-year release, Dryad doi:10.5061/dryad.x0k6djj1h).

Reads the yield files (<root>/yield/<P>/<P>_day_<d>_yield.mat; extracted from the yield_<P>.tar.gz archives). Each file
holds, per electrode: array, grid x/y (10 x 10 Utah), impedance (kOhm), threshold-crossing rates at several thresholds
and the mean threshold-crossing waveform. We use the same definitions as for the monkeys (scripts/failure_stats_xdata.py):
an electrode is 'active' when its rate at -4.5 RMS is >= 2 Hz (also BrainGate's own yield definition).

Per array (participant x array) we report: span, sessions, yield at start/end, alive<->silent Markov switching rates,
silencing and revival counts, abruptness (kurtosis of log-rate changes between sessions <= 7 days apart, after removing
session-wide changes), impedance trend, and the edge-vs-interior difference in activity decline (edge = x or y in {0, 9}).

Usage: python scripts/braingate_failure.py [--root D:/ibci-data/braingate] [--out results/braingate_failure]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(__file__))
from channel_health import runs_below  # noqa: E402
from failure_stats_xdata import fit_switching  # noqa: E402

ALIVE_HZ = 2.0


def load_yield(path):
    d = loadmat(path, simplify_cells=True)
    el = pd.DataFrame(d["electrodes"])
    if "impedance" not in el.columns:                      # impedance is only stored for some sessions
        el["impedance"] = np.nan
    fr = pd.DataFrame(d["firing_rates"])
    fr = fr[fr["threshold_description"].astype(str).isin(["-4.5", "-4.50"])]
    rate = fr.set_index("electrode_id")["firing_rate"]
    el["rate"] = el["electrode_id"].map(rate)
    wf = d.get("spike_waveforms", {})
    if isinstance(wf, dict) and "mean_waveforms" in wf:
        amp = {int(e): float(np.min(w)) for e, w in zip(np.atleast_1d(wf["electrode_id"]), wf["mean_waveforms"])}
        el["wf_amp"] = el["electrode_id"].map(amp)
    el["day"] = int(d["post_implant_day"])
    el["participant"] = str(d["participant_id"])
    return el


def summarize_array(g):
    days = np.sort(g.day.unique()).astype(float)
    R = g.pivot_table(index="day", columns="electrode_id", values="rate").reindex(days).values
    R = np.where(np.isfinite(R), R, np.nan)
    keep = np.isfinite(R).mean(0) > 0.8                    # electrodes present in most sessions
    R = R[:, keep]
    R = pd.DataFrame(R).ffill().bfill().values
    alive = R >= ALIVE_HZ
    yield_ = alive.mean(1) * 100
    ever = alive.sum(0) >= 3
    h_off, h_on = fit_switching(days, alive[:, ever]) if ever.sum() >= 3 and len(days) >= 5 else (np.nan, np.nan)
    init = R[: min(3, len(days))].mean(0) >= ALIVE_HZ
    died = revived = 0
    for c in np.flatnonzero(init):
        below = runs_below(R[:, c], ALIVE_HZ)
        if below.any():
            died += 1
            i0 = int(np.argmax(below))
            revived += int((pd.Series(R[i0:, c] >= 2 * ALIVE_HZ).rolling(3).min() == 1).any())
    lr = np.log(R + 0.1)
    short = np.diff(days) <= 7
    dl = np.diff(lr, axis=0)[short]
    v_ch = (dl - np.median(dl, axis=1, keepdims=True))[:, init].ravel() if dl.size else np.array([])
    # edge vs interior decline (slope of log rate per year), within this array
    xy = g.drop_duplicates("electrode_id").set_index("electrode_id").loc[np.array(sorted(g.electrode_id.unique()))[keep]]
    edge = (xy.x.isin([0, 9]) | xy.y.isin([0, 9])).values
    slopes = np.array([stats.linregress(days / 365.25, lr[:, c]).slope for c in range(lr.shape[1])]) if len(days) > 2 else None
    imp = g.pivot_table(index="day", columns="electrode_id", values="impedance").reindex(days)
    imp = imp.where(imp < 4000)                            # BrainGate excludes >= 4000 kOhm
    imp_med = imp.median(axis=1)
    out = dict(sessions=int(len(days)), span_days=int(days[-1] - days[0]), first_day=int(days[0]),
               yield_first3=float(np.median(yield_[:3])), yield_last3=float(np.median(yield_[-3:])),
               h_off=h_off, h_on=h_on, initially_active=int(init.sum()), silenced=died, revived=revived,
               channel_level_kurtosis=float(stats.kurtosis(v_ch)) if v_ch.size > 20 else np.nan,
               frac_abs_logchange_gt1=float((np.abs(v_ch) > 1).mean()) if v_ch.size else np.nan,
               impedance_first=float(imp_med.iloc[:3].median()), impedance_last=float(imp_med.iloc[-3:].median()),
               impedance_spearman_vs_day=float(stats.spearmanr(days, imp_med, nan_policy="omit").correlation))
    if slopes is not None and edge.any() and (~edge).any():
        out.update(edge_slope=float(np.median(slopes[edge])), interior_slope=float(np.median(slopes[~edge])),
                   edge_vs_interior_p=float(stats.mannwhitneyu(slopes[edge], slopes[~edge], alternative="less").pvalue))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ.get("IBCI_DATA", "D:/ibci-data"), "braingate"))
    ap.add_argument("--out", default="results/braingate_failure")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.root, "**", "*_yield.mat"), recursive=True))
    print(f"{len(files)} yield files", flush=True)
    frames = []
    for i, f in enumerate(files):
        try:
            frames.append(load_yield(f))
        except Exception as e:  # report and continue
            print("skip", os.path.basename(f), e)
        if i % 200 == 0:
            print(i, os.path.basename(f), flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(args.out, "electrode_session.csv"), index=False)
    rows = []
    for (p, a), g in df.groupby(["participant", "array"]):
        if g.day.nunique() < 5:
            continue
        rows.append(dict(participant=p, array=a, **summarize_array(g)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "per_array.csv"), index=False)
    print(res.round(4).to_string())
    pooled = {
        "n_arrays": int(len(res)),
        "median_h_off": float(res.h_off.median()), "median_h_on": float(res.h_on.median()),
        "revived_over_silenced": float(res.revived.sum() / max(res.silenced.sum(), 1)),
        "median_channel_kurtosis": float(res.channel_level_kurtosis.median()),
        "arrays_edge_declines_faster_p<0.05": int((res.get("edge_vs_interior_p", pd.Series(dtype=float)) < 0.05).sum()),
        "fisher_combined_p_edge": float(stats.combine_pvalues(res.edge_vs_interior_p.dropna())[1])
        if "edge_vs_interior_p" in res else None,
        "median_edge_minus_interior_slope": float((res.edge_slope - res.interior_slope).median()) if "edge_slope" in res else None,
    }
    print(json.dumps(pooled, indent=1))
    json.dump(pooled, open(os.path.join(args.out, "pooled.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
