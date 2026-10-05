"""Small-budget recalibration: choose the ridge-to-prior shrinkage from history instead of noisy per-session CV.

For the data-efficiency pairs (results/data_efficiency/ladder.csv), recompute ridge-to-prior (shrinking weights *and*
intercept toward the day-i decoder) on day j's first n labelled trials for every alpha on a grid. Compare policies:
  renorm        no labels (alpha = infinity)
  cv            per-pair k-fold CV over the n trials (as in the ladder)
  history       alpha(n) = the grid value maximising median R2 over pairs whose *training session differs*
                (leave-one-training-session-out; no information from the evaluated pair)
  history_gap   same, but also conditioned on the gap (only pairs with the same target gap)
  oracle        best alpha in hindsight per pair (upper bound, not a usable policy)

Usage: python scripts/recal_policy.py [--ladder results/data_efficiency/ladder.csv] [--out results/recal_policy]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.anatomy import ALPHAS, N_LAGS, Sess, cv_shrunk, r2  # noqa: E402
from ibci.data import link  # noqa: E402
from ibci.linear import lagged  # noqa: E402


def ridge_prior_icpt(H, Y, W0, b0, alpha):
    """Ridge with weights and intercept shrunk toward (W0, b0)."""
    Ha = np.c_[H, np.ones(len(H))].astype(np.float64)
    A = Ha.T @ Ha + alpha * np.eye(Ha.shape[1])
    th = np.linalg.solve(A, Ha.T @ Y + alpha * np.vstack([W0, b0[None]]))
    return th[:-1], th[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="results/data_efficiency/ladder.csv")
    ap.add_argument("--n-trials", type=int, nargs="+", default=[10, 20, 50, 100, 300])
    ap.add_argument("--out", default="results/recal_policy")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ref = pd.read_csv(args.ladder)
    cache = {}

    def sess(k):
        if k not in cache:
            cache[k] = Sess(k)
        return cache[k]

    rows = []
    for _, p in ref.iterrows():
        si, sj = sess(p.train), sess(p.test)
        H_tr, H_te = lagged(sj.ztr, N_LAGS), lagged(sj.zte, N_LAGS)
        for n in args.n_trials:
            lab = sj.trial_id < n
            row = dict(train=p.train, test=p.test, gap_target=p.gap_target, n=n,
                       renorm=r2(si.dec.predict(sj.zte), sj.y_te), own=r2(sj.dec.predict(sj.zte), sj.y_te))
            for a in ALPHAS:
                W, b = ridge_prior_icpt(H_tr[lab], sj.y_tr[lab], si.dec.W, si.dec.b, a)
                row[f"a{a:g}"] = r2(H_te @ W + b, sj.y_te)
            _, f = cv_shrunk(H_tr, sj.y_tr, sj.trial_id, n, ALPHAS, si.dec.W.astype(np.float64), 5,
                             scale_by_rows=False, intercept_anchor=si.dec.b)
            row["cv"] = r2(f(H_te), sj.y_te)
            rows.append(row)
        print(f"{p.train} -> {p.test} done", flush=True)
    df = pd.DataFrame(rows)
    acols = [f"a{a:g}" for a in ALPHAS]
    df["oracle"] = df[acols].max(axis=1)
    hist, hist_gap = [], []
    for _, r in df.iterrows():
        other = df[(df.n == r.n) & (df.train != r.train)]
        hist.append(r[other[acols].median().idxmax()])
        og = other[other.gap_target == r.gap_target]
        hist_gap.append(r[(og if len(og) >= 5 else other)[acols].median().idxmax()])
    df["history"], df["history_gap"] = hist, hist_gap
    df.to_csv(os.path.join(args.out, "policies.csv"), index=False)
    cols = ["renorm", "cv", "history", "history_gap", "oracle", "own"]
    summ = df.groupby(["n"])[cols].median().round(3)
    by_gap = df.groupby(["gap_target", "n"])[cols].median().round(3)
    worse = df.groupby("n").apply(lambda d: pd.Series({c: float((d[c] < d.renorm - 0.01).mean()) for c in ("cv", "history", "history_gap")}))
    print(summ.to_string()); print(by_gap.to_string()); print("share of pairs worse than renorm by > 0.01:"); print(worse.round(3).to_string())
    json.dump({"median_by_n": summ.to_dict(), "share_worse_than_renorm": worse.to_dict()},
              open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
