"""Targeted recalibration: flag drifting channels without labels, then re-learn only those with few labelled trials.

For the data-efficiency pairs, with day j's first n labelled trials (n in --n-trials):
  full        ridge-to-prior on all 96 channels (weights + intercept shrunk to the old decoder)
  stats_k     re-learn only the k channels with the largest label-free change (|mean shift| in day-i sd units +
              |log sd ratio|), all other weights frozen (shrunk to the old weights)
  importance_k  the k channels the old decoder relies on most (weight norm) - control for "important channels"
  wstats_k    label-free change x importance (decoder-weighted change)
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


class GramRefit:
    """All ridge-to-prior refits for one labelled set from a single Gram matrix.

    Re-learning channels S (all lags) with the rest frozen solves (G_SS + a I) th = r_S - G_SF W_F + a * prior_S,
    where G = Ha^T Ha and r = Ha^T Y over the labelled bins (Ha = lagged features plus an intercept column)."""

    def __init__(self, H_tr, Y, dec):
        self.Ha = np.c_[H_tr, np.ones(len(H_tr))].astype(np.float64)
        self.G = self.Ha.T @ self.Ha
        self.r = self.Ha.T @ Y.astype(np.float64)
        self.prior = np.vstack([dec.W, dec.b[None]]).astype(np.float64)   # (L*C + 1, K)
        self.L, self.C = dec.W3.shape[0], dec.W3.shape[1]

    def predict(self, H_te, chans, alpha):
        D = self.L * self.C
        cols = np.r_[np.array([l * self.C + c for l in range(self.L) for c in chans], dtype=int), D]  # + intercept
        frozen = np.setdiff1d(np.arange(D), cols)
        rhs = self.r[cols] - self.G[np.ix_(cols, frozen)] @ self.prior[frozen] + alpha * self.prior[cols]
        th = np.linalg.solve(self.G[np.ix_(cols, cols)] + alpha * np.eye(len(cols)), rhs)
        W = self.prior.copy()
        W[cols] = th
        return np.c_[H_te, np.ones(len(H_te))] @ W


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
        importance = np.linalg.norm(si.dec.W3, axis=(0, 2))        # how much the old decoder relies on each channel
        orders = {"stats": np.argsort(-score), "importance": np.argsort(-importance),
                  "wstats": np.argsort(-score * importance)}        # decoder-weighted label-free change
        rand = [rng.choice(C, max(args.ks), replace=False) for _ in range(3)]
        for n in args.n_trials:
            lab = sj.trial_id < n
            gr = GramRefit(H_tr[lab], sj.y_tr[lab], si.dec)
            base = dict(train=p.train, test=p.test, gap_target=p.gap_target, n=n,
                        renorm=r2(si.dec.predict(sj.zte), sj.y_te), own=r2(sj.dec.predict(sj.zte), sj.y_te))
            for a in GRID:
                row = dict(base, alpha=a, full=r2(gr.predict(H_te, np.arange(C), a), sj.y_te))
                for k in args.ks:
                    for name, order in orders.items():
                        row[f"{name}_k{k}"] = r2(gr.predict(H_te, order[:k], a), sj.y_te)
                    row[f"random_k{k}"] = float(np.mean([r2(gr.predict(H_te, rc[:k], a), sj.y_te) for rc in rand]))
                rows.append(row)
        print(f"{p.train} -> {p.test} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "grid.csv"), index=False)
    methods = ["full"] + [f"{m}_k{k}" for m in ("stats", "importance", "wstats", "random") for k in args.ks]
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
    print(res.groupby(["gap_target", "n"])[["renorm", "full", "stats_k16", "wstats_k16", "importance_k16", "random_k16"]]
          .median().round(3).to_string())


if __name__ == "__main__":
    main()
