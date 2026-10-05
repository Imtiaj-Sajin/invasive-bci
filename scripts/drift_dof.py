"""How many degrees of freedom does decoder-relevant drift occupy? Constrained corrections of a frozen decoder.

(The unconstrained 96x96 input remap can represent any new decoder, see the 2026-10-06 05:00 log entry, so it says
nothing about the structure of drift. These corrections are genuinely constrained.)

For each pair (decoder trained on day i, evaluated on later day j; z-scored SBP, renormalized on day j):
  rank-r remap   x~ = (I + U V^T) z + h,  U, V in R^{96 x r}, r in {1, 2, 4, 8}: decoder change confined to an
                 r-dimensional channel subspace (L-BFGS, lambda = 1e-3, the value CV picks for the full remap)
  k-channel      re-learn the decoder weights of only k channels (others frozen), ridge shrunk to the old weights
                 (alpha = 1e3, the value CV picks for ridge-to-prior), k in {4, 8, 16, 32, 64}; channels chosen by
                   'supervised'  largest weight change of the full ridge-to-prior refit (uses labels; upper-bound proxy)
                   'corr'        largest label-free change of the channel's correlation profile with all other channels
                   'stats'       largest label-free change of channel mean (in day-i sd units) + |log sd ratio|
                   'random'      random channels (mean of 3 draws)
Everything is reported as the share of the drift loss recovered: (R2 - R2_renorm) / (R2_fullrecal - R2_renorm),
where fullrecal = ridge-to-prior on all channels with day j's 300 labelled trials.

Usage: python scripts/drift_dof.py [--ladder results/anatomy/ladder.csv] [--per-gap 30] [--out results/drift_dof]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.anatomy import DEV, N_LAGS, SessCache, r2  # noqa: E402
from ibci.linear import lagged  # noqa: E402

RANKS = [1, 2, 4, 8]
KS = [4, 8, 16, 32, 64]
ALPHA_PRIOR = 1e3
LAM_REMAP = 1e-3


def ridge_prior_icpt(H, Y, W0, b0, alpha):
    Ha = np.c_[H, np.ones(len(H))].astype(np.float64)
    th = np.linalg.solve(Ha.T @ Ha + alpha * np.eye(Ha.shape[1]), Ha.T @ Y + alpha * np.vstack([W0, b0[None]]))
    return th[:-1], th[-1]


def fit_lowrank_remap(Z, Y, dec, r, lam=LAM_REMAP, iters=150, seed=0):
    torch.manual_seed(seed)
    Zt, Yt = torch.tensor(Z, device=DEV), torch.tensor(Y, device=DEV)
    C = Z.shape[1]
    W, b = torch.tensor(dec.W3, device=DEV), torch.tensor(dec.b, device=DEV)
    U = torch.zeros(C, r, device=DEV, requires_grad=True)
    V = (0.01 * torch.randn(C, r, device=DEV)).requires_grad_(True)
    h = torch.zeros(C, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([U, V, h], max_iter=iters, line_search_fn="strong_wolfe")

    def predict(Zin):
        X = Zin + (Zin @ V) @ U.T + h
        P = b.expand(len(X), -1).clone()
        for l in range(N_LAGS):
            P[l:] += X[: len(X) - l] @ W[l]
        return P

    def closure():
        opt.zero_grad()
        loss = ((predict(Zt) - Yt) ** 2).sum(1).mean() + lam * ((U ** 2).sum() + (V ** 2).sum() + (h ** 2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return lambda Zin: predict(torch.tensor(Zin, device=DEV)).detach().cpu().numpy()


def k_channel_refit(H_tr, Y, H_te, dec, chans):
    """Re-learn weights of channels ``chans`` (all lags) with the others frozen; returns test predictions."""
    L, C, K = dec.W3.shape
    cols = np.array([l * C + c for l in range(L) for c in chans])
    frozen = np.setdiff1d(np.arange(L * C), cols)
    off_tr = H_tr[:, frozen] @ dec.W[frozen]
    W, b = ridge_prior_icpt(H_tr[:, cols], Y - off_tr, dec.W[cols], dec.b, ALPHA_PRIOR)
    return H_te[:, cols] @ W + b + H_te[:, frozen] @ dec.W[frozen]


def corr_change(zi, zj):
    Ci, Cj = np.corrcoef(zi.T), np.corrcoef(zj.T)
    C = len(Ci)
    out = np.zeros(C)
    for c in range(C):
        m = np.arange(C) != c
        a, b = np.nan_to_num(Ci[c, m]), np.nan_to_num(Cj[c, m])
        out[c] = 1 - (np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="results/anatomy/ladder.csv")
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 7, 30, 120, 480])
    ap.add_argument("--per-gap", type=int, default=30)
    ap.add_argument("--out", default="results/drift_dof")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    ref = pd.read_csv(args.ladder)
    ref = pd.concat([d.sample(min(args.per_gap, len(d)), random_state=0) for g, d in ref.groupby("gap_target")
                     if g in args.gaps])
    cache = SessCache(maxsize=80)
    rows, t0 = [], time.time()
    for c_i, (_, p) in enumerate(ref.iterrows()):
        si, sj = cache[p.train], cache[p.test]
        H_tr, H_te = lagged(sj.ztr, N_LAGS), lagged(sj.zte, N_LAGS)
        renorm = r2(si.dec.predict(sj.zte), sj.y_te)
        Wf, bf = ridge_prior_icpt(H_tr, sj.y_tr, si.dec.W, si.dec.b, ALPHA_PRIOR)
        full = r2(H_te @ Wf + bf, sj.y_te)
        row = dict(train=p.train, test=p.test, gap_target=p.gap_target, renorm=renorm, fullrecal=full,
                   own=r2(sj.dec.predict(sj.zte), sj.y_te))
        for r in RANKS:
            row[f"rank{r}"] = r2(fit_lowrank_remap(sj.ztr, sj.y_tr, si.dec, r)(sj.zte), sj.y_te)
        C = sj.ztr.shape[1]
        dW = np.linalg.norm((Wf - si.dec.W).reshape(N_LAGS, C, -1), axis=(0, 2))
        mean_shift = np.abs(sj.z.mean - si.z.mean) / si.z.std + np.abs(np.log(sj.z.std / si.z.std))
        orders = {"supervised": np.argsort(-dW), "corr": np.argsort(-corr_change(si.ztr, sj.ztr)),
                  "stats": np.argsort(-mean_shift)}
        for k in KS:
            for name, order in orders.items():
                row[f"{name}_k{k}"] = r2(k_channel_refit(H_tr, sj.y_tr, H_te, si.dec, order[:k]), sj.y_te)
            row[f"random_k{k}"] = float(np.mean([r2(k_channel_refit(H_tr, sj.y_tr, H_te, si.dec,
                                                                    rng.choice(C, k, replace=False)), sj.y_te)
                                                 for _ in range(3)]))
        rows.append(row)
        if (c_i + 1) % 10 == 0 or c_i + 1 == len(ref):
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "dof.csv"), index=False)
            print(f"{c_i + 1}/{len(ref)} pairs, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    den = (df.fullrecal - df.renorm)
    keep = den > 0.02                                 # pairs with a meaningful drift loss
    rec = {}
    for col in [f"rank{r}" for r in RANKS] + [f"{n}_k{k}" for n in ("supervised", "corr", "stats", "random") for k in KS]:
        rec[col] = ((df[col] - df.renorm) / den)[keep]
    R = pd.DataFrame(rec)
    R["gap_target"] = df.gap_target[keep]
    summ = R.groupby("gap_target").median().T.round(3)
    summ["all"] = R.drop(columns="gap_target").median().round(3)
    print(summ.to_string())
    summ.to_csv(os.path.join(args.out, "recovery_summary.csv"))


if __name__ == "__main__":
    main()
