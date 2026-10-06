"""Re-analysis of the public MINDFUL data (Pun et al., Commun Biol 2024; Dryad doi:10.5061/dryad.n2z34tn5s) with
elapsed time as a control.

Closed-loop cursor blocks with a fixed decoder: T11 (15 days), T5 (6 days). Per block: spike-band power and threshold
crossings (20 ms bins), the online decoder's velocity output, and instantaneous angular error (AE). Following MINDFUL,
non-overlapping 60 s windows are compared with the first day (reference) by a Gaussian KL divergence of
  neural    top-10 PCs (reference PCA) of log spike-band power (T11) or sqrt threshold crossings (T5, whose release
            has no SBP), z-scored with reference statistics
  output    the decoder's 2-d velocity output
and correlated with the window's median AE (within non-excluded trials). We then ask whether the KL score explains
AE beyond elapsed days: partial correlation given days, and leave-one-day-out prediction of AE from days only vs
days + KL (linear models).

Usage: python scripts/mindful_reanalysis.py [--root D:/ibci-data/mindful/MINDFUL_Data] [--out results/mindful_reanalysis]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

WIN = 3000  # 60 s of 20 ms bins
K_PC = 10


def gauss_kl(X, Y, ridge=1e-3):
    mx, my = X.mean(0), Y.mean(0)
    Sx = np.cov(X, rowvar=False) + ridge * np.eye(X.shape[1])
    Sy = np.cov(Y, rowvar=False) + ridge * np.eye(Y.shape[1])
    iSy = np.linalg.inv(Sy)
    return 0.5 * (np.trace(iSy @ Sx) + (my - mx) @ iSy @ (my - mx) - X.shape[1]
                  + np.linalg.slogdet(Sy)[1] - np.linalg.slogdet(Sx)[1])


def load_block(bdir):
    d = loadmat(os.path.join(bdir, "data.mat"), simplify_cells=True)
    t = loadmat(os.path.join(bdir, "task.mat"), simplify_cells=True)
    i = loadmat(os.path.join(bdir, "info.mat"), simplify_cells=True)
    if "spikePower" in d:                               # log spike-band power (T11)
        sp = np.log(np.asarray(d["spikePower"], dtype=np.float64) + 1e-6)
    else:                                               # T5 release has threshold crossings only
        sp = np.sqrt(np.asarray(d["nctx"], dtype=np.float64))
    vel = np.asarray(d["cursorVel"], dtype=np.float64)
    ae = np.asarray(i["angleError"], dtype=np.float64).ravel()
    in_trial = np.zeros(len(sp), bool)
    ss = np.atleast_2d(t["startStops"]).astype(int)
    excl = np.atleast_1d(t.get("excludeTrials", np.zeros(len(ss)))).astype(bool)
    for (a, b), e in zip(ss, excl):
        if not e:
            in_trial[max(a - 1, 0):b] = True        # MATLAB 1-based indices
    return sp, vel, ae, in_trial


def partial_r(x, y, z):
    ex = x - np.polyval(np.polyfit(z, x, 1), z)
    ey = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(ex, ey)[0, 1])


def analyse(pdir, name):
    days = sorted(glob.glob(os.path.join(pdir, "day_*")), key=lambda p: int(re.search(r"day_(\d+)", p).group(1)))
    blocks = {int(re.search(r"day_(\d+)", d).group(1)): sorted(glob.glob(os.path.join(d, "block_*"))) for d in days}
    d0 = min(blocks)
    ref = [load_block(b) for b in blocks[d0]]
    ref_sp = np.concatenate([r[0] for r in ref])
    mu, sd = ref_sp.mean(0), ref_sp.std(0) + 1e-6
    V = np.linalg.svd((ref_sp - mu) / sd, full_matrices=False)[2][:K_PC].T
    ref_pc = ((ref_sp - mu) / sd) @ V
    ref_vel = np.concatenate([r[1] for r in ref])
    rows = []
    for day, bl in blocks.items():
        for b in bl:
            sp, vel, ae, it = load_block(b)
            z = ((sp - mu) / sd) @ V
            for w in range(0, len(sp) - WIN + 1, WIN):
                sl = slice(w, w + WIN)
                m = it[sl] & np.isfinite(ae[sl])
                if m.sum() < 500:
                    continue
                rows.append(dict(participant=name, day=day, days=day - d0, block=os.path.basename(b),
                                 ae=float(np.median(ae[sl][m])), kl_neural=gauss_kl(z[sl], ref_pc),
                                 kl_output=gauss_kl(vel[sl], ref_vel)))
    df = pd.DataFrame(rows)
    df["kl_sum"] = df.kl_neural + df.kl_output
    df["log_kl"] = np.log(df.kl_sum + 1e-6)
    out = {"participant": name, "n_days": int(df.day.nunique()), "n_windows": len(df),
           "pearson_ae_days": float(np.corrcoef(df.ae, df.days)[0, 1])}
    for f in ("kl_neural", "kl_output", "log_kl"):
        out[f] = {"pearson_with_ae": float(np.corrcoef(df[f], df.ae)[0, 1]),
                  "partial_given_days": partial_r(df[f].values, df.ae.values, df.days.values.astype(float)),
                  "pearson_with_days": float(np.corrcoef(df[f], df.days)[0, 1])}
    # day-level (median per day), as in session-level reporting
    g = df.groupby("days").agg(ae=("ae", "median"), log_kl=("log_kl", "median")).reset_index()
    out["day_level"] = {"pearson_kl_ae": float(np.corrcoef(g.log_kl, g.ae)[0, 1]),
                        "pearson_days_ae": float(np.corrcoef(g.days, g.ae)[0, 1]),
                        "partial_kl_ae_given_days": partial_r(g.log_kl.values, g.ae.values, g.days.values.astype(float))
                        if len(g) > 3 else None}
    # leave-one-day-out prediction of window AE (reference day excluded from evaluation)
    df["log_kl_neural"] = np.log(df.kl_neural + 1e-6)
    df["log_kl_output"] = np.log(df.kl_output + 1e-6)
    models = {"days": ["days"], "kl": ["log_kl"], "days+kl": ["days", "log_kl"],
              "days+kl_neural": ["days", "log_kl_neural"], "days+kl_output": ["days", "log_kl_output"],
              "kl_output": ["log_kl_output"]}
    preds = {k: [] for k in models}
    truth = []
    for day in df.day.unique():
        if day == d0:
            continue
        tr, te = df[df.day != day], df[df.day == day]
        truth.extend(te.ae)
        for k, cols in models.items():
            A = np.c_[tr[cols].values, np.ones(len(tr))]
            coef = np.linalg.lstsq(A, tr.ae.values, rcond=None)[0]
            preds[k].extend(np.c_[te[cols].values, np.ones(len(te))] @ coef)
    truth = np.array(truth)
    out["lodo_mae"] = {k: float(np.mean(np.abs(np.array(v) - truth))) for k, v in preds.items()}
    return df, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/ibci-data/mindful/MINDFUL_Data")
    ap.add_argument("--out", default="results/mindful_reanalysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    allres = []
    for name in ("T11", "T5"):
        df, res = analyse(os.path.join(args.root, name), name)
        df.to_csv(os.path.join(args.out, f"{name}_windows.csv"), index=False)
        allres.append(res)
        print(json.dumps(res, indent=1), flush=True)
    json.dump(allres, open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
