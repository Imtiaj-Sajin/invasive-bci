"""Descriptive statistics of the BrainGate decoding sessions per participant (for the Methods and Supplementary Table 1).

Per participant: sessions, span in years, arrays and channels, median trials (movement periods) per session and median
movement-period duration. Reads only the trial, electrode and bin-size fields of the compact session files.

Usage: python scripts/decoding_session_stats.py [--out results/decoding_session_stats.json]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.io import loadmat
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ibci.paths import data_root  # noqa: E402

ROOTS = {p: data_root(p) for p in ("T6", "T5", "T9", "T7", "T10", "T8", "T11", "T2", "T3")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/decoding_session_stats.json")
    args = ap.parse_args()
    out = {}
    for p, root in ROOTS.items():
        files = glob.glob(f"{root}/braingate/decoding/{p}/{p}_day_*_decoding.mat")
        if not files:
            continue
        days, ntr, dur, arrays, chans = [], [], [], set(), 0
        for f in files:
            d = loadmat(f, simplify_cells=True, variable_names=["post_implant_day", "trials", "bin_size", "electrodes"])
            days.append(int(d["post_implant_day"]))
            tr = d["trials"]
            st = np.atleast_1d(tr["trial_start_index"]).astype(int)
            en = np.atleast_1d(tr["trial_end_index"]).astype(int)
            ntr.append(len(st))
            dur.append(np.median(en - st) * float(d["bin_size"]))
            arrays |= set(np.atleast_1d(d["electrodes"]["array"]).astype(str))
            chans = max(chans, len(np.atleast_1d(d["electrodes"]["electrode_id"])))
        out[p] = {"sessions": len(files), "first_day": min(days), "last_day": max(days),
                  "years": round((max(days) - min(days)) / 365.25, 2), "arrays": len(arrays), "channels": chans,
                  "median_trials": float(np.median(ntr)), "median_movement_s": round(float(np.median(dur)), 2)}
        print(p, out[p], flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
