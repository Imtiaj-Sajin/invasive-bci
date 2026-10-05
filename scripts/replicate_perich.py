"""Replicate the drift anatomy on DANDI 000688 (Perich/Miller lab): monkeys C (Chewie, ~3 years) and M (Mihili, ~1.5 years).

Differences from LINK, kept on purpose to test generality: another lab, a reaching task (center-out CO / random-target
RT), spike-sorted units summed per electrode instead of threshold crossings / SBP, and cursor kinematics. Same oracle
ladder (scripts/drift_anatomy.py) with the same decoder (ridge, 8 lags of z-scored features). Sessions are paired only
within the same task; train = first 80% of trials, test = the rest; up to 300 labelled trials for supervised rungs.

Usage: python scripts/replicate_perich.py [--subjects C M] [--gaps 1 7 30 120 480] [--out results/replication]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.anatomy import K_LAT, N_LAGS, ladder  # noqa: E402
from ibci.data import perich  # noqa: E402
from ibci.linear import LagDecoder  # noqa: E402
from ibci.preprocess import ZScore  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from drift_anatomy import select_pairs  # noqa: E402


class PerichSess:
    """Duck-typed equivalent of ibci.anatomy.Sess for a 000688 session with a subject-wide channel order."""

    def __init__(self, s: perich.PSession, X: np.ndarray, day0: np.datetime64, alpha=1.0):
        self.key = s.key
        self.day = int((np.datetime64(s.date) - day0).astype(int))
        self.style = s.task
        tr, te = perich.split_bins(s)
        self.x_tr, self.x_te = X[tr], X[te]
        self.y_tr, self.y_te = s.kin[tr], s.kin[te]
        n_tr = int(round(perich.TRAIN_FRAC * len(s.trial_start)))
        starts = np.r_[s.trial_start[:n_tr], tr.stop]
        self.trial_id = np.repeat(np.arange(n_tr), np.diff(starts)).astype(int)
        lead = len(self.x_tr) - len(self.trial_id)      # bins before the first trial start belong to trial 0
        self.trial_id = np.r_[np.zeros(lead, int), self.trial_id]
        self.z = ZScore().fit(self.x_tr)
        self.ztr, self.zte = self.z.transform(self.x_tr), self.z.transform(self.x_te)
        self.dec = LagDecoder.fit(self.ztr, self.y_tr, N_LAGS, alpha)
        self.V = np.linalg.svd(self.ztr, full_matrices=False)[2][:K_LAT].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["C", "M"])
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 7, 30, 120, 480])
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge alpha of the base decoder (sparser features than LINK)")
    ap.add_argument("--out", default="results/replication")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for subj in args.subjects:
        keys = perich.list_sessions(subj)
        raw = [perich.load_session(k) for k in keys]
        X, labels = perich.aligned_counts(raw)
        day0 = np.datetime64(raw[0].date)
        sessions = [PerichSess(s, x, day0, args.alpha) for s, x in zip(raw, X)]
        pairs = select_pairs(sessions, args.gaps)
        print(f"{subj}: {len(sessions)} sessions over {max(s.day for s in sessions)} days, {len(labels)} channels, {len(pairs)} pairs",
              flush=True)
        rows, t0 = [], time.time()
        for c, (i, j, g) in enumerate(pairs):
            si, sj = sessions[i], sessions[j]
            rows.append(dict(subject=subj, train=si.key, test=sj.key, style=si.style, gap_target=g,
                             days=sj.day - si.day, train_day=si.day, **ladder(si, sj, [300])))
            if (c + 1) % 20 == 0 or c + 1 == len(pairs):
                pd.DataFrame(rows).to_csv(os.path.join(args.out, f"{subj}_ladder.csv"), index=False)
                print(f"  {c + 1}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
        df = pd.DataFrame(rows)
        cols = ["L0", "L2", "L4u", "L4s", "L3_n300", "L5_n300", "L6p_n300", "own"]
        print(df.groupby("gap_target")[cols].median().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
