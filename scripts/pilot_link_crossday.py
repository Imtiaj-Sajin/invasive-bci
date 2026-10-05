"""Pilot: do label-free signals predict how a fixed decoder degrades across days on LINK?

For every session i a ridge decoder (8 lags of spike-band power, alpha=0.1, z-scored with day-i stats; LINK-paper
settings) is trained on the first 300 trials. It is applied unchanged to the held-out trials of every later
session j, giving R2[i, j]. For each pair (i, j) we also compute label-free features from day j's *unlabelled*
neural data, and ask how well they track R2[i, j] beyond what elapsed days alone explain.

Usage: python scripts/pilot_link_crossday.py [--max-sessions N] [--out results/pilot]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.data import link  # noqa: E402
from ibci.metrics import r2_per_dof  # noqa: E402
from ibci.preprocess import ZScore, add_history, split_bins  # noqa: E402

N_LAGS, ALPHA, N_ENS, N_PCS = 8, 0.1, 5, 10
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ridge_fit(X, Y, alpha):
    """Closed-form ridge with intercept. X (T, D), Y (T, K) -> W (D, K), b (K,)."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ Yc)
    return W, ym - xm @ W


def fold_norm(W, b, z: ZScore):
    """Fold per-channel z-scoring into ridge weights so decoders apply to raw history features."""
    Wl = W.reshape(N_LAGS, -1, W.shape[1])                     # (lags, C, K)
    Wf = Wl / z.std[None, :, None]
    bf = b - (Wl * (z.mean / z.std)[None, :, None]).sum((0, 1))
    return Wf.reshape(W.shape), bf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--out", default="results/pilot")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    link.build_cache(verbose=False)
    keys = link.list_sessions()
    if args.max_sessions:
        keys = keys[: args.max_sessions]
    print(f"{len(keys)} sessions", flush=True)

    sess, dec = [], []
    t0 = time.time()
    for k in keys:
        s = link.load_session(k)
        try:
            tr, te = split_bins(s.trial_start, s.sbp.shape[0])
        except ValueError:
            continue
        x_tr, x_te = s.sbp[tr], s.sbp[te]
        z = ZScore().fit(x_tr)
        H_tr = add_history(z.transform(x_tr), N_LAGS).reshape(len(x_tr), -1)
        W, b = ridge_fit(H_tr, s.kin[tr], ALPHA)
        # bootstrap-over-trials ensemble for disagreement
        bounds = np.r_[s.trial_start[:300], tr.stop]
        ens = []
        for _ in range(N_ENS):
            pick = rng.choice(300, 300, replace=True)
            idx = np.concatenate([np.arange(bounds[p], bounds[p + 1]) for p in pick])
            ens.append(ridge_fit(H_tr[idx], s.kin[tr][idx], ALPHA))
        pcs = np.linalg.svd(z.transform(x_tr), full_matrices=False)[2][:N_PCS]
        sess.append(dict(key=k, day=s.day, style=s.style, imp=s.imp, z_own=z,
                         tc_rate=s.tc[te].mean(0) / link.BIN_S,
                         H_raw=add_history(x_te, N_LAGS).reshape(len(x_te), -1),
                         # "renorm": day-j z-scoring fitted on day j's own (unlabelled) first 300 trials
                         H_renorm=add_history(z.transform(x_te), N_LAGS).reshape(len(x_te), -1),
                         y_te=s.kin[te], pcs=pcs, x_te=x_te))
        dec.append(dict(raw=(W, b), ens_raw=ens, folded=fold_norm(W, b, z),
                        ens_folded=[fold_norm(We, be, z) for We, be in ens], z=z))
    print(f"trained {len(dec)} decoders in {time.time() - t0:.0f}s", flush=True)

    def t32(a):
        return torch.tensor(np.asarray(a), device=DEV, dtype=torch.float32)

    rows = []
    for j, sj in enumerate(sess):
        idx = list(range(0, j + 1))  # decoders trained on day i <= j; i == j is the within-day reference
        for mode, H, wkey, ekey in (("fixed", sj["H_raw"], "folded", "ens_folded"),
                                    ("renorm", sj["H_renorm"], "raw", "ens_raw")):
            Hj = t32(H)
            P = torch.einsum("td,ndk->ntk", Hj, t32([dec[i][wkey][0] for i in idx])) + t32([dec[i][wkey][1] for i in idx])[:, None]
            We = t32([[e[0] for e in dec[i][ekey]] for i in idx])
            be = t32([[e[1] for e in dec[i][ekey]] for i in idx])
            disagree = (torch.einsum("td,nedk->netk", Hj, We) + be[:, :, None]).var(1).mean((1, 2)).cpu().numpy()
            P = P.cpu().numpy()
            for n, i in enumerate(idx):
                si = sess[i]
                zi = dec[i]["z"] if mode == "fixed" else sj["z_own"]
                xz = (sj["x_te"] - zi.mean) / zi.std
                pj = np.linalg.svd(xz - xz.mean(0), full_matrices=False)[2][:N_PCS]
                cosines = np.linalg.svd(si["pcs"] @ pj.T, compute_uv=False)
                r2 = r2_per_dof(P[n], sj["y_te"])
                live_i = si["tc_rate"] > 1.0
                imp_lr = np.abs(np.log(sj["imp"] / si["imp"]))
                rows.append(dict(
                    mode=mode, train=si["key"], test=sj["key"], train_style=si["style"], test_style=sj["style"],
                    days=sj["day"] - si["day"], r2=float(r2.mean()), r2_pos=float(r2[:2].mean()), r2_vel=float(r2[2:].mean()),
                    f_mean_shift=float(np.abs(xz.mean(0)).mean()),
                    f_std_logratio=float(np.abs(np.log(np.maximum(xz.std(0), 1e-6))).mean()),
                    f_subspace=float(1 - (cosines ** 2).mean()),
                    f_disagree=float(disagree[n]),
                    f_pred_vel_bias=float(np.abs(P[n][:, 2:].mean(0)).mean()),
                    f_pred_pos_var=float(np.log(P[n][:, :2].var(0).mean() / max(si["y_te"][:, :2].var(0).mean(), 1e-9))),
                    f_imp_logratio=float(np.nanmedian(imp_lr)) if np.isfinite(imp_lr).any() else np.nan,
                    f_dead=int((live_i & (sj["tc_rate"] < 1.0)).sum()),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "crossday_pairs.csv"), index=False)

    # ---- quick analysis: cross-day pairs only, same target style to avoid task confound
    summary = {"n_sessions": len(sess)}
    rr = stats.rankdata
    for mode in ("fixed", "renorm"):
        dm = df[df["mode"] == mode]
        d = dm[(dm.days > 0) & (dm.train_style == dm.test_style)].copy()
        d["logdays"] = np.log1p(d.days)
        out = {"n_pairs": int(len(d)), "within_day_r2_median": float(dm[dm.days == 0].r2.median()),
               "crossday_r2_median": float(d.r2.median()), "features": {}}
        for f in [c for c in d.columns if c.startswith("f_")] + ["logdays"]:
            e = d[np.isfinite(d[f])]
            if len(e) < 10 or e[f].nunique() < 2:
                continue
            res = {"n": int(len(e)), "spearman": float(stats.spearmanr(e[f], e.r2).correlation)}
            if f != "logdays":  # partial Spearman controlling for log elapsed days
                a = rr(e[f]) - np.polyval(np.polyfit(rr(e.logdays), rr(e[f]), 1), rr(e.logdays))
                c = rr(e.r2) - np.polyval(np.polyfit(rr(e.logdays), rr(e.r2), 1), rr(e.logdays))
                res["partial_given_days"] = float(np.corrcoef(a, c)[0, 1])
            out["features"][f] = res
        summary[mode] = out
    print(json.dumps(summary, indent=1))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
