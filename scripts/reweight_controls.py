"""Controls for the per-channel re-weighting result.

Does re-weighting recover accuracy because each channel keeps its tuning pattern, or because one free, signed weight
per channel can rebuild a low-dimensional readout from almost any decoder? For each pair (decoders with the tuned
penalty, alpha = 1e4, for the day-i, re-weighted and reference decoders), with day j's first 300 labelled trials:
  renorm       renormalized fixed decoder
  rew          re-weighting: one signed weight per channel and one offset per output (as in the ladder)
  rew_nonneg   the same with weights constrained to be non-negative (no sign flips)
  rew_perm     re-weighting of the day-i decoder with its channels randomly permuted (tuning patterns assigned to
               the wrong channels): the flexibility of re-weighting alone
  rew_perm_nn  permuted decoder with non-negative weights
  own          reference decoder trained on day j
Weights are shrunk toward 1 with the strength chosen by five-fold cross-validation (unconstrained fit), and the same
strength is used for the constrained fits.

Usage: python scripts/reweight_controls.py --subject T6 [--max-train 40] [--out results/reweight_controls]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from decay_alpha import GAPS, LADDER, sessions_for  # noqa: E402
from ibci.anatomy import LAMS_CF, cv_shrunk, gain_features, r2  # noqa: E402
from ibci.linear import LagDecoder  # noqa: E402


def permuted(dec, perm):
    W3 = dec.W3[:, perm, :]
    return LagDecoder(W3.reshape(-1, W3.shape[2]), dec.b, dec.n_lags)


def fit_nonneg(Ftr, Y, b0, lam):
    """min ||Y - sum_c g_c F_c - d||^2 + lam*T*(||g - 1||^2 + ||d - b0||^2), g >= 0."""
    T, C, K = Ftr.shape
    A = np.concatenate([Ftr.transpose(0, 2, 1).reshape(T * K, C), np.tile(np.eye(K), (T, 1))], axis=1)
    y = Y.reshape(-1)
    s = np.sqrt(lam * T)
    A_aug = np.vstack([A, s * np.eye(C + K)])
    y_aug = np.concatenate([y, s * np.r_[np.ones(C), b0]])
    lb = np.r_[np.zeros(C), -np.inf * np.ones(K)]
    sol = lsq_linear(A_aug, y_aug, bounds=(lb, np.inf * np.ones(C + K)), lsmr_tol="auto", max_iter=500)
    return sol.x[:C], sol.x[C:]


def run_rew(dec, sj, n=300):
    Ftr, Fte = gain_features(sj.ztr, dec), gain_features(sj.zte, dec)
    tid = np.asarray(sj.trial_id)
    lab = tid < n
    lam, f = cv_shrunk(Ftr, sj.y_tr, tid, n, LAMS_CF, np.ones(Ftr.shape[1]), 5, intercept_anchor=dec.b)
    full = r2(f(Fte), sj.y_te)
    g, d = fit_nonneg(Ftr[lab].astype(np.float64), sj.y_tr[lab].astype(np.float64), np.asarray(dec.b, float), lam)
    nonneg = r2(np.einsum("tck,c->tk", Fte, g) + d, sj.y_te)
    return full, nonneg, float((g <= 1e-6).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--max-train", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--out", default="results/reweight_controls")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lin = pd.read_csv(LADDER.get(args.subject, f"results/replication_bg/{args.subject}_ladder.csv"))
    lin = lin[lin.gap_target.isin(GAPS)][["train", "test", "gap_target", "days"]].reset_index(drop=True)
    rng = np.random.default_rng(0)
    trains = sorted(lin.train.unique())
    keep = set(rng.choice(trains, min(args.max_train, len(trains)), replace=False))
    pairs = lin[lin.train.isin(keep)].reset_index(drop=True)
    sess = sessions_for(args.subject, args.alpha, set(pairs.train) | set(pairs.test))
    rows, t0 = [], time.time()
    out_csv = os.path.join(args.out, f"{args.subject}.csv")
    for c, r in pairs.iterrows():
        si, sj = sess[r.train], sess[r.test]
        C = si.ztr.shape[1]
        rew, rew_nn, zero_frac = run_rew(si.dec, sj)
        perm = rng.permutation(C)
        rp, rp_nn, _ = run_rew(permuted(si.dec, perm), sj)
        rows.append(dict(subject=args.subject, **r.to_dict(), renorm=r2(si.dec.predict(sj.zte), sj.y_te), rew=rew,
                         rew_nonneg=rew_nn, nonneg_zero_frac=zero_frac, rew_perm=rp, rew_perm_nn=rp_nn,
                         own=r2(sj.dec.predict(sj.zte), sj.y_te)))
        if (c + 1) % 20 == 0 or c + 1 == len(pairs):
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"{c + 1}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    for col in ("rew", "rew_nonneg", "rew_perm", "rew_perm_nn"):
        df[f"rec_{col}"] = (df[col] - df.renorm) / (df.own - df.renorm)
    print(df.groupby("gap_target")[[c for c in df.columns if c.startswith("rec_")]].median().round(2).to_string())


if __name__ == "__main__":
    main()
