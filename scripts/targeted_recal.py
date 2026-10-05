"""Targeted recalibration: flag drifting channels without labels, then re-learn only those with few labelled trials.

For the data-efficiency pairs, with day j's first n labelled trials (n in --n-trials):
  full        ridge-to-prior on all 96 channels (weights + intercept shrunk to the old decoder)
  stats_k     re-learn only the k channels with the largest label-free change (|mean shift| in day-i sd units +
              |log sd ratio|), all other weights frozen (shrunk to the old weights)
  random_k    same with k random channels (control)
Shrinkage alpha for every method is chosen from history (leave-one-training-session-out median-best on the grid),
the policy that proved safe in scripts/recal_policy.py. Also reports renormalization only (no labels).

Usage: python scripts/targeted_recal.py [--ladder results/data_efficiency/ladder.csv] [--out results/targeted_recal]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.anatomy import N_LAGS, SessCache, r2  # noqa: E402
from ibci.linear import lagged  # noqa: E402

GRID = [1, 10, 100, 1e3, 1e4, 1e5]


def refit(H_tr, Y, H_te, dec, chans, alpha):
    L, C, K = dec.W3.shape
    cols = np.array([l * C + c for l in range(L) for c in chans])
    frozen = np.setdiff1d(np.arange(L * C), cols)
    off_tr = H_tr[:, frozen] @ dec.W[frozen] if len(frozen) else 0.0
    Ha = np.c_[H_tr[:, cols], np.ones(len(H_tr))].astype(np.float64)
    th = np.linalg.solve(Ha.T @ Ha + alpha * np.eye(Ha.shape[1]),
                         Ha.T @ (Y - off_tr) + alpha * np.vstack([dec.W[cols], dec.b[None]]))
    pred = H_te[:, cols] @ th[:-1] + th[-1]
    return pred + (H_te[:, frozen] @ dec.W[frozen] if len(frozen) else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="results/data_efficiency/ladder.csv")
    ap.add_argument("--n-trials", type=int, nargs="+", default=[10, 20, 50, 100])
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--out", default="results/targeted_recal")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ref = pd.read_csv(args.ladder)
    cache = SessCache(maxsize=60)
    rng = np.random.default_rng(0)
    rows = []
    for _, p in ref.iterrows():
        si, sj = cache[p.train], cache[p.test]
        H_tr, H_te = lagged(sj.ztr, N_LAGS), lagged(sj.zte, N_LAGS)
        C = sj.ztr.shape[1]
        score = np.abs(sj.z.mean - si.z.mean) / si.z.std + np.abs(np.log(sj.z.std / si.z.std))
        order = np.argsort(-score)
        rand = [rng.choice(C, max(args.ks), replace=False) for _ in range(3)]
        for n in args.n_trials:
            lab = sj.trial_id < n
            base = dict(train=p.train, test=p.test, gap_target=p.gap_target, n=n,
                        renorm=r2(si.dec.predict(sj.zte), sj.y_te), own=r2(sj.dec.predict(sj.zte), sj.y_te))
            for a in GRID:
                row = dict(base, alpha=a, full=r2(refit(H_tr[lab], sj.y_tr[lab], H_te, si.dec, np.arange(C), a), sj.y_te))
                for k in args.ks:
                    row[f"stats_k{k}"] = r2(refit(H_tr[lab], sj.y_tr[lab], H_te, si.dec, order[:k], a), sj.y_te)
                    row[f"random_k{k}"] = float(np.mean([r2(refit(H_tr[lab], sj.y_tr[lab], H_te, si.dec, rc[:k], a), sj.y_te)
                                                         for rc in rand]))
                rows.append(row)
        print(f"{p.train} -> {p.test} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "grid.csv"), index=False)
    methods = ["full"] + [f"{m}_k{k}" for m in ("stats", "random") for k in args.ks]
    out = []
    for (tr, te, n), d in df.groupby(["train", "test", "n"]):
        others = df[(df.n == n) & (df.train != tr)]
        rec = dict(train=tr, test=te, n=n, gap_target=d.gap_target.iloc[0], renorm=d.renorm.iloc[0], own=d.own.iloc[0])
        for m in methods:
            a_best = others.groupby("alpha")[m].median().idxmax()          # history-chosen alpha (no leakage)
            rec[m] = float(d[d.alpha == a_best][m].iloc[0])
        out.append(rec)
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(args.out, "history_policy.csv"), index=False)
    print(res.groupby("n")[["renorm"] + methods + ["own"]].median().round(3).to_string())
    print(res.groupby(["gap_target", "n"])[["renorm", "full", "stats_k16", "random_k16"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
