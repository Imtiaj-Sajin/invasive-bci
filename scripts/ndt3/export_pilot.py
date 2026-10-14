"""Export the NDT3 pilot data (threshold-crossing counts) and the linear reference results on the same pairs.

Pair selection (fixed before any NDT3 run, decision 0005): for each pilot individual, the valid manifest pairs of the
training sessions that have pairs at >= 4 of the 5 gaps; of those sessions, 3 evenly spaced in time; all their pairs.

Output (H:/ibci-data/ndt3_pilot by default):
  <subject>/<session>.npz   x_tr (T, C) int16 crossings, y_tr (T, D), trial_tr (T,), x_te, y_te  (paper's split)
  pairs.csv                 subject, train, test, gap_target, days and the R2 of the linear reference methods on
                            threshold crossings (ridge, 8 lags, penalty 1e4): reference (same-day), none, renorm,
                            procrustes, reweight / prior_refit / retrain with 50 and 300 labelled trials.

Usage: python scripts/ndt3/export_pilot.py [--subjects N T5 T8] [--n-train 3] [--out H:/ibci-data/ndt3_pilot]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from ibci import bench  # noqa: E402
from ibci.bench import baselines as B  # noqa: E402

ROOTS = {"T8": "H:/ibci-data", "T9": "G:/ibci-data", "T7": "F:/ibci-data", "T10": "F:/ibci-data"}
BUDGETS = (50, 300)


def session_time(key):
    """Sortable time of a session key: 'T5_day123' -> 123; '2020-01-27_CO' -> days since 2000-01-01."""
    if "_day" in key:
        return int(key.split("_day")[1])
    return int((np.datetime64(key[:10]) - np.datetime64("2000-01-01")).astype(int))


def pick_pairs(subject, n_train):
    m = bench.manifest(subject)
    m = m[m.valid & m.gap_target.isin(bench.GAPS)]
    cover = m.groupby("train").gap_target.nunique()
    keys = sorted(cover[cover >= 4].index, key=session_time)
    chosen = [keys[i] for i in np.linspace(0, len(keys) - 1, n_train).round().astype(int)]
    return m[m.train.isin(chosen)].reset_index(drop=True)


def counts(x):
    return np.rint(x).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["N", "T5", "T8"])
    ap.add_argument("--n-train", type=int, default=3)
    ap.add_argument("--out", default="H:/ibci-data/ndt3_pilot")
    args = ap.parse_args()
    rows = []
    for k in args.subjects:
        os.environ["IBCI_DATA"] = ROOTS.get(k, "D:/ibci-data")
        pairs = pick_pairs(k, args.n_train)
        os.makedirs(os.path.join(args.out, k), exist_ok=True)
        sess = bench._sessions(k, set(pairs.train) | set(pairs.test), "tx")
        for key in sorted(set(pairs.train) | set(pairs.test)):
            s = sess[key]
            x_tr = counts(s.ztr * s.z.std + s.z.mean)
            x_te = counts(s.zte * s.z.std + s.z.mean)
            np.savez_compressed(os.path.join(args.out, k, f"{key}.npz"), x_tr=x_tr, y_tr=s.y_tr.astype(np.float32),
                                trial_tr=s.trial_id.astype(np.int32), x_te=x_te, y_te=s.y_te.astype(np.float32),
                                day=s.day)
        for r in pairs.itertuples():
            t = bench.Task(k, r.train, r.test, int(r.gap_target), int(r.days), 0, True, sess[r.train], sess[r.test])
            row = {"subject": k, "train": r.train, "test": r.test, "gap_target": int(r.gap_target), "days": int(r.days),
                   "sbp_ref_r2": r.ref_r2, "reference": t.reference_r2()}
            for name, f in B.LABEL_FREE.items():
                row[name] = t.score(f(t))["r2"]
            for n in BUDGETS:
                t.n_labels = n
                for name, f in B.SUPERVISED.items():
                    row[f"{name}_n{n}"] = t.score(f(t))["r2"]
            rows.append(row)
            print(k, r.train, r.test, r.gap_target, f"ref {row['reference']:.3f} renorm {row['renorm']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "pairs.csv"), index=False, float_format="%.6f")
    print(f"{len(rows)} pairs written to {args.out}")


if __name__ == "__main__":
    main()
