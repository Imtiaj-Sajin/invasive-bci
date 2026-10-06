"""Human replication of the drift anatomy on BrainGate closed-loop cursor sessions (decoding_<P>.tar.gz, Dryad
doi:10.5061/dryad.x0k6djj1h; e.g. T6: 124 sessions over 3.1 years, T5: 92 sessions over 7.2 years).

Per session: spike-band power (10 ms bins, summed to 20 ms), cursor and target positions, movement-attempt ("go")
periods. As is standard for offline analysis of closed-loop BCI data, the decoded variable is the *intended* movement
direction: the unit vector from cursor to target, during go periods only (bins outside go periods are dropped).
Train = first 80% of go periods, test = the rest. Same decoder (ridge, 8 lags of z-scored features) and the same oracle
ladder as for the monkeys (ibci.anatomy.ladder), with daily renormalization; all sessions of a participant are one
'style', and pairs are chosen at target gaps.

Usage: python scripts/replicate_braingate_decoding.py --participant T6 [--root D:/ibci-data/braingate] [--out results/replication_bg]
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.anatomy import K_LAT, N_LAGS, ladder  # noqa: E402
from ibci.linear import LagDecoder  # noqa: E402
from ibci.preprocess import ZScore  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from drift_anatomy import select_pairs  # noqa: E402

TRAIN_FRAC = 0.8


def load_session(path):
    """Return (day, X (T20, C) float32 SBP at 20 ms, Y (T20, 2) intention, trial index per 20 ms bin)."""
    d = loadmat(path, simplify_cells=True)
    day = int(d["post_implant_day"])
    sbp = np.asarray(d["neural"]["sbp"], dtype=np.float32)
    cur = np.asarray(d["cursor_position"], dtype=np.float32)
    tgt = np.asarray(d["target_position"], dtype=np.float32)
    tr = d["trials"]
    starts, ends = np.atleast_1d(tr["trial_start_index"]).astype(int), np.atleast_1d(tr["trial_end_index"]).astype(int)
    segs_x, segs_y, segs_t = [], [], []
    for k, (a, b) in enumerate(zip(starts, ends)):
        b = a + 2 * ((b - a) // 2)                     # even length so 10 ms bins pair up into 20 ms bins
        if b - a < 20:                                 # skip go periods shorter than 200 ms
            continue
        x = sbp[a:b].reshape(-1, 2, sbp.shape[1]).mean(1)
        v = (tgt[a:b] - cur[a:b]).reshape(-1, 2, 2).mean(1)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        segs_x.append(x)
        segs_y.append(np.where(n > 1e-6, v / np.maximum(n, 1e-6), 0.0))
        segs_t.append(np.full(len(x), k))
    X, Y, T = np.concatenate(segs_x), np.concatenate(segs_y).astype(np.float32), np.concatenate(segs_t)
    T = np.unique(T, return_inverse=True)[1]           # renumber kept trials 0..n-1
    return day, X, Y, T


class BGSess:
    """Duck-typed equivalent of ibci.anatomy.Sess for one BrainGate decoding session."""

    def __init__(self, key, day, X, Y, T, alpha=0.1):
        self.key, self.day, self.style = key, day, "cursor"
        n_tr = int(round(TRAIN_FRAC * (T.max() + 1)))
        tr = T < n_tr
        self.y_tr, self.y_te = Y[tr], Y[~tr]
        self.trial_id = T[tr]
        self.z = ZScore().fit(X[tr])
        self.ztr = self.z.transform(X[tr]).astype(np.float32)
        self.zte = self.z.transform(X[~tr]).astype(np.float32)
        self.dec = LagDecoder.fit(self.ztr, self.y_tr, N_LAGS, alpha)
        self.V = np.linalg.svd(self.ztr, full_matrices=False)[2][:K_LAT].T

    @property
    def x_te(self):
        return self.zte * self.z.std + self.z.mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--root", default=os.path.join(os.environ.get("IBCI_DATA", "D:/ibci-data"), "braingate"))
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 7, 30, 120, 480])
    ap.add_argument("--out", default="results/replication_bg")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.root, "decoding", args.participant, "*_decoding.mat")),
                   key=lambda p: int(re.search(r"_day_(\d+)_", p).group(1)))
    sessions = []
    for f in files:
        try:
            day, X, Y, T = load_session(f)
            if T.max() + 1 < 20:                       # need enough go periods to split
                continue
            sessions.append(BGSess(f"{args.participant}_day{day}", day, X, Y, T))
        except Exception as e:
            print("skip", os.path.basename(f), e, flush=True)
    print(f"{args.participant}: {len(sessions)} sessions over {sessions[-1].day - sessions[0].day} days", flush=True)
    pairs = select_pairs(sessions, args.gaps)
    print(f"{len(pairs)} pairs", flush=True)
    rows, t0 = [], time.time()
    for c, (i, j, g) in enumerate(pairs):
        si, sj = sessions[i], sessions[j]
        rows.append(dict(participant=args.participant, train=si.key, test=sj.key, gap_target=g, days=sj.day - si.day,
                         train_day=si.day, **ladder(si, sj, [300])))
        if (c + 1) % 20 == 0 or c + 1 == len(pairs):
            pd.DataFrame(rows).to_csv(os.path.join(args.out, f"{args.participant}_ladder.csv"), index=False)
            print(f"  {c + 1}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["ret_L2"] = df.L2 / df.own
    df["ret_L3"] = df.L3_n300 / df.own
    print(df.groupby("gap_target")[["L2", "L4u", "L3_n300", "L6p_n300", "own", "ret_L2", "ret_L3"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
