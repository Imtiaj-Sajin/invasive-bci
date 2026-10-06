"""Does a MINDFUL-style instability score predict decoder performance beyond elapsed time?

Pun et al. (Commun Biol 2024) score instability as the KL divergence between distributions of neural features and decoder
outputs in a reference vs a test window, and report strong raw correlations with decoder error (r = 0.91 in one
participant over 142 days). Raw correlations over a multi-day series are confounded by elapsed time, since both
instability and error grow with time. Here, for every LINK pair (decoder trained on day i, applied on later day j
with daily renormalization), we compute Gaussian KL divergences:
  kl_neural   neural features projected on day i's top-10 PCs (z-scored per day, as in adaptive z-scoring)
  kl_output   decoder outputs (4 kinematic predictions) on day j vs day i
  kl_sum      kl_neural + kl_output
and report, against R2 on day j:
  raw Spearman correlation, partial Spearman given log(days elapsed), and time-blocked cross-validated prediction
  of R2 from days only vs days + KL.

Usage: python scripts/mindful_test.py [--ladder results/anatomy/ladder.csv] [--out results/mindful_test]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.anatomy import SessCache, r2  # noqa: E402

K_PC = 10


def gauss_kl(X, Y, ridge=1e-3):
    """KL( N(mean_X, cov_X) || N(mean_Y, cov_Y) ) for samples X, Y (rows = observations)."""
    mx, my = X.mean(0), Y.mean(0)
    Sx = np.cov(X, rowvar=False) + ridge * np.eye(X.shape[1])
    Sy = np.cov(Y, rowvar=False) + ridge * np.eye(Y.shape[1])
    iSy = np.linalg.inv(Sy)
    d = X.shape[1]
    _, ldx = np.linalg.slogdet(Sx)
    _, ldy = np.linalg.slogdet(Sy)
    return 0.5 * (np.trace(iSy @ Sx) + (my - mx) @ iSy @ (my - mx) - d + ldy - ldx)


def partial_spearman(x, y, z):
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="results/anatomy/ladder.csv")
    ap.add_argument("--out", default="results/mindful_test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ref = pd.read_csv(args.ladder)
    cache = SessCache(maxsize=80)
    rows = []
    for _, p in ref.iterrows():
        si, sj = cache[p.train], cache[p.test]
        V = si.V[:, :K_PC]
        kl_n = gauss_kl(sj.zte @ V, si.zte @ V)
        out_i, out_j = si.dec.predict(si.zte), si.dec.predict(sj.zte)
        kl_o = gauss_kl(out_j, out_i)
        rows.append(dict(train=p.train, test=p.test, gap_target=p.gap_target, days=p.days, train_day=p.train_day,
                         r2=r2(out_j, sj.y_te), kl_neural=kl_n, kl_output=kl_o, kl_sum=kl_n + kl_o))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "pairs.csv"), index=False)
    df["logdays"] = np.log1p(df.days)
    res = {"n_pairs": len(df), "spearman_r2_logdays": float(stats.spearmanr(df.r2, df.logdays).correlation)}
    for f in ("kl_neural", "kl_output", "kl_sum"):
        res[f] = {"raw_spearman_with_r2": float(stats.spearmanr(df[f], df.r2).correlation),
                  "partial_given_logdays": partial_spearman(df[f], df.r2, df.logdays),
                  "spearman_with_logdays": float(stats.spearmanr(df[f], df.logdays).correlation)}
    # time-blocked CV (blocks of training days)
    blocks = pd.qcut(df.train_day, 5, labels=False, duplicates="drop")
    for name, cols in (("days_only", ["logdays"]), ("kl_only", ["kl_sum"]), ("days+kl", ["logdays", "kl_neural", "kl_output"])):
        pred = np.zeros(len(df))
        for b in np.unique(blocks):
            m = (blocks == b).values
            pred[m] = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0).fit(
                df.loc[~m, cols], df.r2[~m]).predict(df.loc[m, cols])
        res[f"cv_{name}"] = {"mae": float(np.abs(pred - df.r2).mean()), "spearman": float(stats.spearmanr(pred, df.r2).correlation)}
    import json
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
