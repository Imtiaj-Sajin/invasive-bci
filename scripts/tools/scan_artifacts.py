"""Scan BrainGate decoding sessions for non-physiological spike-band-power values.

For each session, the movement-period features are z-scored with the mean and s.d. of the session's training segment
(first 80% of trials, as in the analyses), and we report the largest |z| in the held-out segment and the share of bins
with any channel above |z| = 50. Physiological SBP changes stay within tens of s.d.; values of 1e3 or more indicate
recording artefacts.

Usage: python scripts/tools/scan_artifacts.py P [P ...] [--out results/artifact_scan.csv]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "..", "src"), os.path.join(os.path.dirname(__file__), "..")]
from replicate_braingate_decoding import TRAIN_FRAC, load_session  # noqa: E402

ROOTS = {"T6": "D:/ibci-data", "T5": "D:/ibci-data", "T9": "G:/ibci-data", "T7": "F:/ibci-data", "T10": "F:/ibci-data",
         "T8": "G:/ibci-data", "T11": "D:/ibci-data", "T2": "F:/ibci-data", "T3": "F:/ibci-data"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("participants", nargs="+")
    ap.add_argument("--out", default="results/artifact_scan.csv")
    args = ap.parse_args()
    rows = []
    for p in args.participants:
        for f in glob.glob(f"{ROOTS[p]}/braingate/decoding/{p}/{p}_day_*_decoding.mat"):
            day, X, Y, T = load_session(f)
            tr = T < int(round(TRAIN_FRAC * (T.max() + 1)))
            mu, sd = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-9)
            z_te, z_tr = np.abs((X[~tr] - mu) / sd), np.abs((X[tr] - mu) / sd)
            rows.append(dict(participant=p, day=day, max_z_test=float(z_te.max()), max_z_train=float(z_tr.max()),
                             frac_bins_z50=float((z_te.max(1) > 50).mean()), max_raw=float(X.max()),
                             median_raw=float(np.median(X))))
        q = pd.DataFrame([r for r in rows if r["participant"] == p])
        print(f"{p}: {len(q)} sessions; max|z| test >1e3 in {(q.max_z_test > 1e3).sum()}, >100 in "
              f"{(q.max_z_test > 100).sum()}; train max|z| >1e3 in {(q.max_z_train > 1e3).sum()}", flush=True)
    old = pd.read_csv(args.out) if os.path.exists(args.out) else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old.participant.isin(new.participant.unique())]
    pd.concat([old, new]).sort_values(["participant", "day"]).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
