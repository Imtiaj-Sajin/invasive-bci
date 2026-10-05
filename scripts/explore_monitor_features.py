"""Exploration: richer label-free monitor features for renormalized ridge decoders on LINK.

Adds output self-consistency (pred position derivative vs pred velocity), output smoothness, output-distribution
distance to the training kinematics, and cross-view disagreement (SBP decoder vs threshold-crossing decoder,
8-lag vs 1-lag decoder). Evaluates (a) partial Spearman with R2 given elapsed days and (b) time-blocked
cross-validated prediction of R2 with and without label-free features.

Usage: python scripts/explore_monitor_features.py [--out results/explore] [--max-horizon 60]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.data import link  # noqa: E402
from ibci.metrics import r2_per_dof  # noqa: E402
from ibci.preprocess import ZScore, add_history, split_bins  # noqa: E402

ALPHA = 0.1
K_ALIGN = 16


def ridge_fit(X, Y, alpha=ALPHA):
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ Yc)
    return W.astype(np.float32), (ym - xm @ W).astype(np.float32)


def autocorr(x, lag):
    x = x - x.mean(0)
    return float(((x[lag:] * x[:-lag]).sum(0) / np.maximum((x ** 2).sum(0), 1e-12)).mean())


def wasserstein_1d(a, b, n=200):
    q = np.linspace(0.005, 0.995, n)
    return float(np.mean([np.abs(np.quantile(a[:, k], q) - np.quantile(b[:, k], q)).mean() for k in range(a.shape[1])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/explore")
    ap.add_argument("--max-horizon", type=int, default=60, help="only pairs with 1..H days between train and test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    link.build_cache(verbose=False)
    sess = []
    for k in link.list_sessions():
        s = link.load_session(k)
        try:
            tr, te = split_bins(s.trial_start, s.sbp.shape[0])
        except ValueError:
            continue
        views = {}
        for name, x, lags in (("sbp8", s.sbp, 8), ("sbp1", s.sbp, 1), ("tc8", s.tc, 8)):
            z = ZScore().fit(x[tr])  # day-own normalization from unlabelled first 300 trials (renorm deployment)
            H_tr = add_history(z.transform(x[tr]), lags).reshape(tr.stop, -1)
            H_te = add_history(z.transform(x[te]), lags).reshape(-1, lags * x.shape[1])
            views[name] = dict(W=ridge_fit(H_tr, s.kin[tr]), H_te=H_te)
        zs = ZScore().fit(s.sbp[tr])
        V = np.linalg.svd(zs.transform(s.sbp[tr]), full_matrices=False)[2][:K_ALIGN].T   # (96, k) loadings
        sess.append(dict(key=k, day=s.day, style=s.style, views=views, y_tr=s.kin[tr], y_te=s.kin[te],
                         pcs=V[:, :10].T, V=V, xz_te=zs.transform(s.sbp[te]).astype(np.float32)))
    print(len(sess), "sessions", flush=True)

    rows = []
    for j, sj in enumerate(sess):
        for i in range(j + 1):
            si = sess[i]
            days = sj["day"] - si["day"]
            if si["style"] != sj["style"] or days > args.max_horizon:
                continue
            pred = {v: sj["views"][v]["H_te"] @ si["views"][v]["W"][0] + si["views"][v]["W"][1] for v in si["views"]}
            P = pred["sbp8"]
            r2 = r2_per_dof(P, sj["y_te"])
            dpos = np.diff(P[:, :2], axis=0)
            pv = [stats.pearsonr(dpos[:, f], P[1:, 2 + f])[0] for f in range(2)]
            ytrue_tr = si["y_tr"]
            cos = np.linalg.svd(si["pcs"] @ np.linalg.svd(sj["xz_te"] - sj["xz_te"].mean(0), full_matrices=False)[2][:10].T,
                                compute_uv=False)
            # alignment disagreement: decoder applied to day-j data reconstructed through its own top-k subspace vs
            # through the Procrustes-rotated subspace mapped into day-i coordinates
            Vi, Vj = si["V"], sj["V"]
            U, _, Wt = np.linalg.svd(Vj.T @ Vi)
            R = U @ Wt                                                    # (k, k) orthogonal, Vj R ~ Vi
            lat = sj["xz_te"] @ Vj
            Wd, bd = si["views"]["sbp8"]["W"]
            p_own = add_history(lat @ Vj.T, 8).reshape(len(lat), -1) @ Wd + bd
            p_ali = add_history(lat @ R @ Vi.T, 8).reshape(len(lat), -1) @ Wd + bd
            B = Wd.reshape(8, -1, Wd.shape[1]).sum(0)                     # (96, K) effective readout
            rows.append(dict(
                f_align_dis=float(((p_own - p_ali) ** 2).mean() / ytrue_tr.var(0).mean()),
                f_procrustes_res=float(np.linalg.norm(Vj @ R - Vi) ** 2 / K_ALIGN),
                f_readout_offsub=float(1 - (np.linalg.norm(Vj.T @ B) ** 2) / np.linalg.norm(B) ** 2),
                train=si["key"], test=sj["key"], days=days, r2=float(r2.mean()),
                own_r2=float(r2_per_dof(sj["views"]["sbp8"]["H_te"] @ sj["views"]["sbp8"]["W"][0] + sj["views"]["sbp8"]["W"][1], sj["y_te"]).mean()),
                f_posvel=float(np.mean(pv)),
                f_smooth_ratio=autocorr(P[:, 2:], 5) - autocorr(ytrue_tr[:, 2:], 5),
                f_out_w1=wasserstein_1d(P[:, :2], ytrue_tr[:, :2]),
                f_out_var=float(np.log(P.var(0).mean() / ytrue_tr.var(0).mean())),
                f_dis_tc=float(((pred["sbp8"] - pred["tc8"]) ** 2).mean() / ytrue_tr.var(0).mean()),
                f_dis_lag=float(((pred["sbp8"] - pred["sbp1"]) ** 2).mean() / ytrue_tr.var(0).mean()),
                f_subspace=float(1 - (cos ** 2).mean()),
                f_modulation=float(np.linalg.svd(sj["xz_te"] - sj["xz_te"].mean(0), compute_uv=False)[:5].__pow__(2).sum()
                                   / (sj["xz_te"] - sj["xz_te"].mean(0)).__pow__(2).sum()),
            ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "pairs.csv"), index=False)
    d = df[df.days > 0].copy()
    d["logdays"] = np.log1p(d.days)
    rr = stats.rankdata
    feats = [c for c in d.columns if c.startswith("f_")]
    res = {"n_pairs": int(len(d)), "n_sessions": len(sess), "spearman_logdays": float(stats.spearmanr(d.logdays, d.r2).correlation)}
    for f in feats:
        a = rr(d[f]) - np.polyval(np.polyfit(rr(d.logdays), rr(d[f]), 1), rr(d.logdays))
        c = rr(d.r2) - np.polyval(np.polyfit(rr(d.logdays), rr(d.r2), 1), rr(d.logdays))
        res[f] = {"spearman": round(float(stats.spearmanr(d[f], d.r2).correlation), 3),
                  "partial_given_days": round(float(np.corrcoef(a, c)[0, 1]), 3)}

    # time-blocked CV: hold out blocks of *test* sessions in time; predict cross-day R2
    test_day = d.test.str[:10].map(lambda s: (np.datetime64(s) - np.datetime64("2020-01-27")).astype(int))
    blocks = pd.qcut(test_day, 5, labels=False, duplicates="drop")
    out = {}
    for name, cols in (("days_only", ["logdays"]), ("label_free", feats), ("days+label_free", ["logdays"] + feats)):
        pred = np.zeros(len(d))
        for b in np.unique(blocks):
            m = blocks == b
            mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0)
            mdl.fit(d.loc[~m, cols], d.r2[~m])
            pred[m.values] = mdl.predict(d.loc[m, cols])
        out[name] = {"mae": round(float(np.abs(pred - d.r2).mean()), 4),
                     "spearman": round(float(stats.spearmanr(pred, d.r2).correlation), 3)}
    res["cv_predict_r2"] = out

    # unexpected bad days: residual from a time-only curve (fit out-of-block) below -1 SD
    from sklearn.metrics import roc_auc_score
    base = np.zeros(len(d))
    for b in np.unique(blocks):
        m = (blocks == b).values
        base[m] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=2, random_state=0).fit(
            d.loc[~m, ["logdays"]], d.r2[~m]).predict(d.loc[m, ["logdays"]])
    resid = d.r2.values - base
    bad = resid < -resid.std()
    res["unexpected_bad_auroc"] = {"n_bad": int(bad.sum())}
    for f in feats:
        auc = roc_auc_score(bad, d[f])
        res["unexpected_bad_auroc"][f] = round(float(max(auc, 1 - auc)), 3)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
