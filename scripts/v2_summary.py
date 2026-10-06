"""Consolidated per-individual summary for manuscript v2 (results/v2_summary.json).

Collects, for every individual with results: decay of the renormalized fixed decoder with the default and the tuned
ridge penalty (strict 1-day and 2-day pairs separately), days to half retention, share of the loss recovered by
per-channel gains (linear and network decoders), network versus linear accuracy, label-free corrections (mean
updating, Procrustes rotation with and without variance normalization, CORAL, the Degenhart factor-analysis
stabilizer), and the data efficiency of supervised corrections. Paired comparisons use one value per training
session (mean over its pairs at gaps <= 120 days unless stated) with two-sided Wilcoxon signed-rank tests; medians
get 95% cluster-bootstrap intervals over training sessions.

Usage: python scripts/v2_summary.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.stats import cluster_bootstrap  # noqa: E402

GAPS = [1, 7, 30, 120, 480]
IND = {"N": ("Monkey N", "results/anatomy/ladder.csv"), "C": ("Monkey C", "results/replication/C_ladder.csv"),
       "M": ("Monkey M", "results/replication/M_ladder.csv"), "T6": ("Human T6", "results/replication_bg/T6_ladder.csv"),
       "T5": ("Human T5", "results/replication_bg/T5_ladder.csv"), "T9": ("Human T9", "results/replication_bg/T9_ladder.csv")}
TUNED = {"N": 1e4, "C": 1e4, "M": 1e4, "T6": 1e4, "T5": 1e4, "T9": 1e4}


def ci(d, col, n_boot=2000):
    if len(d) < 3:
        return None
    est, lo, hi = cluster_bootstrap(d, col, n_boot=n_boot)
    return [round(est, 3), round(lo, 3), round(hi, 3)]


def t_half(curve):
    lg = np.log(GAPS)
    for k in range(1, len(GAPS)):
        if np.isfinite(curve[k]) and curve[k] < 0.5 <= curve[k - 1]:
            return float(np.exp(lg[k - 1] + (curve[k - 1] - 0.5) / (curve[k - 1] - curve[k]) * (lg[k] - lg[k - 1])))
    return float("inf") if curve[-1] >= 0.5 else (1.0 if curve[0] < 0.5 else float("nan"))


def decay_block(d, l2="L2", own="own", l3="L3_n300"):
    d = d[d.gap_target.isin(GAPS)].assign(ret=lambda x: x[l2] / x[own])
    if l3 in d:
        d = d.assign(ret3=lambda x: x[l3] / x[own], rec=lambda x: (x[l3] - x[l2]) / (x[own] - x[l2]))
    out = {"n_pairs": int(len(d)), "n_train": int(d.train.nunique())}
    for g, q in d.groupby("gap_target"):
        e = {"n": int(len(q)), "retention": ci(q, "ret")}
        if "ret3" in q:
            e["retention_gains"] = round(float(q.ret3.median()), 3)
            e["loss_recovered_by_gains"] = round(float(q.rec.median()), 3)
        out[f"gap{int(g)}"] = e
    one = d[(d.gap_target == 1)]
    for k, q in one.groupby("days"):
        out[f"strict_{int(k)}day"] = {"n": int(len(q)), "retention": ci(q, "ret")}
    curve = [d[d.gap_target == g].ret.median() if (d.gap_target == g).any() else np.nan for g in GAPS]
    out["t_half_days"] = round(t_half(curve), 1)
    return out


def paired(d, a, b, max_gap=120):
    q = d[d.gap_target <= max_gap]
    s = q.groupby("train")[[a, b]].mean().dropna()
    if len(s) < 6:
        return None
    diff = s[a] - s[b]
    w = stats.wilcoxon(diff) if (diff != 0).any() else None
    return {"n_sessions": int(len(s)), "median_diff": round(float(diff.median()), 4),
            "frac_positive": round(float((diff > 0).mean()), 2), "P": float(w.pvalue) if w else None}


def main():
    out = {}
    for k, (name, path) in IND.items():
        if not os.path.exists(path):
            continue
        res = {"name": name, "default": decay_block(pd.read_csv(path))}
        tp = f"results/decay_alpha/{k}_alpha{TUNED[k]:g}.csv"
        if os.path.exists(tp) and len(pd.read_csv(tp)) >= len(pd.read_csv(path)[lambda x: x.gap_target.isin(GAPS)]) * 0.95:
            tuned = pd.read_csv(tp)
            res["tuned"] = decay_block(tuned)
            base = pd.read_csv(path)[["train", "test", "gap_target", "L2", "own"]]
            m = tuned.merge(base, on=["train", "test", "gap_target"], suffixes=("_t", "_d"))
            m = m.assign(ret_t=m.L2_t / m.own_t, ret_d=m.L2_d / m.own_d)
            res["tuned_vs_default_retention"] = paired(m, "ret_t", "ret_d", max_gap=480)
            res["tuned_vs_default_abs_r2"] = paired(m, "L2_t", "L2_d", max_gap=480)
        nn = f"results/nn_ladder/{k}.csv"
        if os.path.exists(nn):
            q = pd.read_csv(nn)
            q = q.assign(ret_nn=q.L2 / q.own, ret_lin=q.ridge_L2 / q.ridge_own,
                         rec_nn=(q.L3 - q.L2) / (q.own - q.L2), rec_lin=(q.ridge_L3 - q.ridge_L2) / (q.ridge_own - q.ridge_L2),
                         d_own=q.own - q.ridge_own)
            res["network"] = {"n_pairs": int(len(q)), "own_minus_linear": ci(q, "d_own"),
                              "own_nn_median": round(float(q.own.median()), 3),
                              "own_linear_median": round(float(q.ridge_own.median()), 3),
                              "retention_nn_vs_linear": paired(q, "ret_nn", "ret_lin", max_gap=480),
                              "by_gap": {int(g): {"ret_nn": round(float(x.ret_nn.median()), 3),
                                                  "ret_lin": round(float(x.ret_lin.median()), 3),
                                                  "rec_nn": round(float(x.rec_nn.median()), 3),
                                                  "rec_lin": round(float(x.rec_lin.median()), 3)}
                                         for g, x in q.groupby("gap_target")}}
        av = f"results/align_variants/{k}.csv"
        if os.path.exists(av):
            q = pd.read_csv(av)
            res["alignment"] = {
                "by_gap": q.groupby("gap_target")[["mean_only", "mean_plus_align", "renorm", "renorm_plus_align", "own"]]
                .median().round(3).to_dict("index"),
                "rotation_on_mean_updated": paired(q, "mean_plus_align", "mean_only"),
                "rotation_on_renormalized": paired(q, "renorm_plus_align", "renorm"),
                "renorm_vs_mean_only": paired(q, "renorm", "mean_only")}
        lf = f"results/label_free_plus/{k}.csv"
        if os.path.exists(lf):
            q = pd.read_csv(lf)
            res["label_free_plus"] = {
                "by_gap": q.groupby("gap_target")[["renorm", "coral", "fa_fixed", "fa_stab", "fa_own", "own"]]
                .median().round(3).to_dict("index"),
                "coral_vs_renorm": paired(q, "coral", "renorm", max_gap=480),
                "fa_stab_vs_fa_fixed": paired(q, "fa_stab", "fa_fixed", max_gap=480),
                "fa_stab_vs_renorm": paired(q, "fa_stab", "renorm", max_gap=480)}
        ge = f"results/gain_efficiency/{k}.csv"
        if os.path.exists(ge):
            q = pd.read_csv(ge)
            eff = {}
            for n in (10, 20, 50, 100, 300):
                for rung in ("L3", "L6p", "L6"):
                    c = f"{rung}_n{n}"
                    if c in q:
                        eff[c] = round(float((q[c] / q.own).median()), 3)
            eff["no_labels"] = round(float((q.L2 / q.own).median()), 3)
            res["data_efficiency"] = {"n_pairs": int(len(q)), "median_retention": eff}
        out[k] = res
    json.dump(out, open("results/v2_summary.json", "w"), indent=1, default=float)
    for k, r in out.items():
        d = r["default"]
        print(f"{r['name']:10s} 1d strict {d.get('strict_1day', {}).get('retention')}  t_half {d['t_half_days']}"
              + (f"  tuned t_half {r['tuned']['t_half_days']}" if "tuned" in r else ""))


if __name__ == "__main__":
    main()
