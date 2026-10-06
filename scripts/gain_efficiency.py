"""How many labelled trials does each supervised correction need? (any subject; same pairs as the linear ladder)

For a random subset of training sessions (fixed seed), runs the correction ladder at n = 10, 20, 50, 100 and 300
labelled trials of the test day (supervised subspace rotation and full input remap skipped). Shows whether
re-learning one gain per channel, which recovers most of the loss in the human participants, works with very few
trials compared with ridge regression shrunk toward the old decoder.

Usage: python scripts/gain_efficiency.py --subject T6 [--max-train 30] [--out results/gain_efficiency]
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
from ibci.anatomy import ladder  # noqa: E402

N_LIST = [10, 20, 50, 100, 300]
BASE_ALPHA = {"N": 0.1, "C": 1.0, "M": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--max-train", type=int, default=30)
    ap.add_argument("--out", default="results/gain_efficiency")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    src = LADDER.get(args.subject, f"results/replication_bg/{args.subject}_ladder.csv")
    lin = pd.read_csv(src)
    lin = lin[lin.gap_target.isin(GAPS)][["train", "test", "gap_target", "days"]].reset_index(drop=True)
    rng = np.random.default_rng(0)
    trains = sorted(lin.train.unique())
    keep = set(rng.choice(trains, min(args.max_train, len(trains)), replace=False))
    pairs = lin[lin.train.isin(keep)].reset_index(drop=True)
    sess = sessions_for(args.subject, BASE_ALPHA.get(args.subject, 0.1), set(pairs.train) | set(pairs.test))
    rows, t0 = [], time.time()
    out_csv = os.path.join(args.out, f"{args.subject}.csv")
    for c, r in pairs.iterrows():
        sj = sess[r.test]
        n_avail = int(np.max(sj.trial_id)) + 1
        res = ladder(sess[r.train], sj, [n for n in N_LIST if n <= n_avail], skip=("L4", "L5"))
        rows.append(dict(subject=args.subject, **r.to_dict(), n_trials_available=n_avail, **res))
        if (c + 1) % 10 == 0 or c + 1 == len(pairs):
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"{c + 1}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    for n in N_LIST:
        for rung in ("L3", "L6p", "L6"):
            col = f"{rung}_n{n}"
            if col in df:
                df[f"ret_{col}"] = df[col] / df.own
    cols = [c for c in df.columns if c.startswith("ret_")]
    print(df.groupby("gap_target")[cols].median().round(2).T.to_string())


if __name__ == "__main__":
    main()
