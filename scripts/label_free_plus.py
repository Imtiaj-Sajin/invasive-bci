"""Stronger label-free corrections, on the same pairs as the linear ladder.

  renorm      per-channel renormalization, frozen ridge decoder (the baseline, L2 in the ladder)
  coral       CORAL (Sun et al. 2016): day-j renormalized features are re-coloured so that their full covariance
              matches day i's, z' = C_i^{1/2} C_j^{-1/2} z (shrinkage-regularized covariances); frozen decoder
  fa_fixed    factor-analysis latent decoder (d latents, Degenhart et al. 2020): ridge on 8 lags of day-i FA
              posterior means; on day j the day-i FA model is reused on renormalized features
  fa_stab     the stabilizer of Degenhart et al. 2020: FA refitted on day j without labels, loadings aligned to
              day i by orthogonal Procrustes on stable channels (iteratively keeping the channels with the smallest
              loading residuals), latents inferred from stable channels only, frozen latent decoder
  fa_own      FA latent decoder trained on day j (reference for the latent decoders)
  own         ridge decoder trained on day j (reference)

Usage: python scripts/label_free_plus.py --subject N|C|M|T6|T5|T9 [--latents 10] [--out results/label_free_plus]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from decay_alpha import GAPS, LADDER, sessions_for  # noqa: E402
from ibci.anatomy import N_LAGS, r2  # noqa: E402
from ibci.linear import LagDecoder  # noqa: E402

BASE_ALPHA = {"N": 0.1, "C": 1.0, "M": 1.0}


def mat_pow(C, p, shrink=0.05):
    C = (1 - shrink) * C + shrink * np.trace(C) / len(C) * np.eye(len(C))
    w, V = np.linalg.eigh(C)
    return (V * np.maximum(w, 1e-8) ** p) @ V.T


class FAModel:
    def __init__(self, Z, d, seed=0):
        fa = FactorAnalysis(n_components=d, random_state=seed).fit(Z)
        self.L, self.psi, self.mu = fa.components_.T, fa.noise_variance_, fa.mean_      # L: (C, d)

    def latents(self, Z, L=None, rows=None):
        L = self.L if L is None else L
        rows = np.arange(len(self.psi)) if rows is None else rows
        Lr, pr = L[rows], self.psi[rows]
        A = Lr.T / pr                                                                  # (d, |rows|)
        M = np.linalg.solve(A @ Lr + np.eye(L.shape[1]), A)
        return (Z[:, rows] - self.mu[rows]) @ M.T


def stabilize(fa_i, fa_j, keep=0.6, iters=10):
    """Degenhart-style alignment of day-j loadings to day-i loadings on stable channels."""
    C = len(fa_i.psi)
    stable = np.ones(C, bool)
    for _ in range(iters):
        U, _, Vt = np.linalg.svd(fa_j.L[stable].T @ fa_i.L[stable])
        R = U @ Vt
        res = np.linalg.norm(fa_j.L @ R - fa_i.L, axis=1)
        stable = res <= np.quantile(res, keep)
    return fa_j.L @ R, np.flatnonzero(stable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--latents", type=int, default=10)
    ap.add_argument("--out", default="results/label_free_plus")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    alpha = BASE_ALPHA.get(args.subject, 0.1)
    src = LADDER.get(args.subject, f"results/replication_bg/{args.subject}_ladder.csv")
    pairs = pd.read_csv(src)
    pairs = pairs[pairs.gap_target.isin(GAPS)][["train", "test", "gap_target", "days"]].reset_index(drop=True)
    sess = sessions_for(args.subject, alpha, set(pairs.train) | set(pairs.test))
    fa, fdec, cov = {}, {}, {}

    def prep(s):
        if s.key not in fa:
            fa[s.key] = FAModel(s.ztr, args.latents)
            fdec[s.key] = LagDecoder.fit(fa[s.key].latents(s.ztr), s.y_tr, N_LAGS, alpha)
            cov[s.key] = np.cov(s.ztr, rowvar=False)

    rows, t0 = [], time.time()
    out_csv = os.path.join(args.out, f"{args.subject}.csv")
    for c, r in pairs.iterrows():
        si, sj = sess[r.train], sess[r.test]
        prep(si)
        prep(sj)
        T = mat_pow(cov[si.key], 0.5) @ mat_pow(cov[sj.key], -0.5)
        L_al, st = stabilize(fa[si.key], fa[sj.key])
        fj = fa[sj.key]
        lat_stab = FAModel.__new__(FAModel)
        lat_stab.L, lat_stab.psi, lat_stab.mu = L_al, fj.psi, fj.mu
        rows.append(dict(**r.to_dict(),
                         renorm=r2(si.dec.predict(sj.zte), sj.y_te),
                         coral=r2(si.dec.predict(sj.zte @ T.T), sj.y_te),
                         fa_fixed=r2(fdec[si.key].predict(fa[si.key].latents(sj.zte)), sj.y_te),
                         fa_stab=r2(fdec[si.key].predict(lat_stab.latents(sj.zte, rows=st)), sj.y_te),
                         fa_own=r2(fdec[sj.key].predict(fj.latents(sj.zte)), sj.y_te),
                         own=r2(sj.dec.predict(sj.zte), sj.y_te)))
        if (c + 1) % 50 == 0 or c + 1 == len(pairs):
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"{c + 1}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(df.groupby("gap_target")[["renorm", "coral", "fa_fixed", "fa_stab", "fa_own", "own"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
