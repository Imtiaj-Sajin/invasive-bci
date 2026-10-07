"""Spike amplitude early vs late on each electrode of one BrainGate array (released robust mean waveforms).

For every electrode with spikes in at least 3 sessions before day 200 and 3 sessions after day 2000, the median
peak-to-peak amplitude in each period is compared (Wilcoxon signed-rank test across electrodes).

Usage: python scripts/amplitude_summary.py [--participant T5] [--array lateral] [--out results/amplitude_T5.json]
"""
import argparse
import glob
import json
import re

import numpy as np
from scipy.io import loadmat
from scipy.stats import wilcoxon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default="T5")
    ap.add_argument("--array", default="lateral")
    ap.add_argument("--root", default="D:/ibci-data/braingate/yield")
    ap.add_argument("--early", type=int, default=200)
    ap.add_argument("--late", type=int, default=2000)
    ap.add_argument("--out", default="results/amplitude_T5.json")
    args = ap.parse_args()
    files = glob.glob(f"{args.root}/{args.participant}/{args.participant}_day_*_yield.mat")
    amp = {}
    for f in sorted(files, key=lambda f: int(re.search(r"_day_(\d+)_", f).group(1))):
        m = loadmat(f, simplify_cells=True, variable_names=["post_implant_day", "electrodes", "spike_waveforms"])
        e, w = m["electrodes"], m["spike_waveforms"]
        on = set(np.asarray(e["electrode_id"])[np.asarray(e["array"]) == args.array])
        for i, v in zip(np.atleast_1d(w["electrode_id"]), np.atleast_2d(w["mean_waveforms"])):
            if i in on and np.all(np.isfinite(v)):
                amp.setdefault(int(i), []).append((int(m["post_implant_day"]), float(v.max() - v.min())))
    early, late = [], []
    for lst in amp.values():
        d, a = np.array(lst).T
        if (d < args.early).sum() >= 3 and (d > args.late).sum() >= 3:
            early.append(np.median(a[d < args.early]))
            late.append(np.median(a[d > args.late]))
    early, late = np.array(early), np.array(late)
    ratio = late / early
    out = {"participant": args.participant, "array": args.array, "n_electrodes": int(len(early)),
           "early_median_uv": float(np.median(early)), "late_median_uv": float(np.median(late)),
           "ratio_median": float(np.median(ratio)), "ratio_q25": float(np.percentile(ratio, 25)),
           "ratio_q75": float(np.percentile(ratio, 75)), "n_smaller": int((late < early).sum()),
           "wilcoxon_p": float(wilcoxon(early, late).pvalue), "early_days": args.early, "late_days": args.late,
           "n_sessions": len(files)}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
