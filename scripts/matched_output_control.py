"""Matched-output control: decode movement *direction* in the reaching monkeys, as in the human analyses.

In the BrainGate participants the decoded variable is a 2-d unit vector (direction from cursor to target). In
monkeys C and M the main analyses decode 4 kinematic outputs. To test whether differences between species in what
the correction ladder recovers reflect the decoded variable rather than the recordings, this script rebuilds the
monkey sessions with a 2-d unit-vector target: the direction of hand velocity, on bins where speed exceeds the
session's 25th percentile (other bins dropped, as non-movement periods are dropped for the humans). The ladder
(supervised subspace rotation and full input remap skipped) is then run on the same pairs as the main analysis.

Usage: python scripts/matched_output_control.py --subject C|M [--out results/matched_output]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from decay_alpha import GAPS, LADDER, sessions_for  # noqa: E402
from ibci.anatomy import K_LAT, N_LAGS, ladder  # noqa: E402
from ibci.linear import LagDecoder  # noqa: E402
from ibci.preprocess import ZScore  # noqa: E402


class DirSess:
    """Duck-typed session with a unit-vector velocity-direction target (movement bins only)."""

    def __init__(self, s, alpha):
        self.key, self.day, self.style = s.key, s.day, s.style
        x_tr = s.ztr * s.z.std + s.z.mean
        v_tr, v_te = s.y_tr[:, 2:4], s.y_te[:, 2:4]
        sp_tr, sp_te = np.linalg.norm(v_tr, axis=1), np.linalg.norm(v_te, axis=1)
        thr = np.percentile(sp_tr, 25)
        m_tr, m_te = sp_tr > thr, sp_te > thr
        self.y_tr = (v_tr[m_tr] / sp_tr[m_tr, None]).astype(np.float32)
        self.y_te = (v_te[m_te] / sp_te[m_te, None]).astype(np.float32)
        self.trial_id = np.asarray(s.trial_id)[m_tr]
        self.z = ZScore().fit(x_tr[m_tr])
        self.ztr = self.z.transform(x_tr[m_tr]).astype(np.float32)
        self.zte = self.z.transform(s.x_te[m_te]).astype(np.float32)
        self.dec = LagDecoder.fit(self.ztr, self.y_tr, N_LAGS, alpha)
        self.V = np.linalg.svd(self.ztr, full_matrices=False)[2][:K_LAT].T

    @property
    def x_te(self):
        return self.zte * self.z.std + self.z.mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, choices=["C", "M"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="results/matched_output")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pairs = pd.read_csv(LADDER[args.subject])
    pairs = pairs[pairs.gap_target.isin(GAPS)][["train", "test", "gap_target", "days"]].reset_index(drop=True)
    raw = sessions_for(args.subject, args.alpha, set(pairs.train) | set(pairs.test))
    sess = {k: DirSess(s, args.alpha) for k, s in raw.items()}
    rows, t0 = [], time.time()
    for c, r in pairs.iterrows():
        rows.append(dict(subject=args.subject, **r.to_dict(), **ladder(sess[r.train], sess[r.test], [300], skip=("L4", "L5"))))
        if (c + 1) % 25 == 0 or c + 1 == len(pairs):
            pd.DataFrame(rows).to_csv(os.path.join(args.out, f"{args.subject}.csv"), index=False)
            print(f"{c + 1}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).assign(ret=lambda x: x.L2 / x.own, ret3=lambda x: x.L3_n300 / x.own,
                                   rec=lambda x: (x.L3_n300 - x.L2) / (x.own - x.L2))
    print(df.groupby("gap_target")[["L2", "L3_n300", "own", "ret", "ret3", "rec"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
