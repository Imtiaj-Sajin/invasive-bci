"""Formal tests quoted in the manuscript. Independent unit = training session (decoder analyses) or array.

Writes results/paper_stats.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.participants import humans, valid_pairs  # noqa: E402

out = {}
SUBJ = {"N": "results/anatomy/ladder.csv", "C": "results/replication/C_ladder.csv",
        "M": "results/replication/M_ladder.csv"}
HUMANS = humans()   # included participants (pre-specified reference-decoder rule)
SUBJ.update({p: f"results/replication_bg/{p}_ladder.csv" for p in HUMANS})
L = {k: valid_pairs(pd.read_csv(v)) for k, v in SUBJ.items()}

# 1. Overnight retention: does it differ across subjects? (one value per training session: mean over its 1-day pairs)
grp = []
for k, d in L.items():
    g = d[d.gap_target == 1].assign(r=lambda x: x.L2 / x.own).groupby("train").r.mean()
    grp.append(g.values)
    out.setdefault("overnight_n_sessions", {})[k] = int(len(g))
h = stats.kruskal(*grp)
out["overnight_kruskal"] = {"H": float(h.statistic), "df": len(grp) - 1, "P": float(h.pvalue)}

# 2. Renormalization vs label-free alignment and vs mean-only (session-level means across gaps <= 120 d), Wilcoxon two-sided
for k in ["N"] + HUMANS:
    d = L[k][L[k].gap_target <= 120]
    s = d.groupby("train")[["L1", "L2", "L4u"]].mean()
    for a, b in (("L2", "L4u"), ("L4u", "L1"), ("L2", "L1")):
        w = stats.wilcoxon(s[a], s[b])
        out.setdefault("renorm_vs_align", {})[f"{k}_{a}_vs_{b}"] = {
            "n_sessions": int(len(s)), "median_diff": float(np.median(s[a] - s[b])), "W": float(w.statistic), "P": float(w.pvalue)}

# 3. Regularization (1% rule alpha vs 0.1): session-level mean cross-day gain, Wilcoxon two-sided, per subject
for name, path in [("N", "results/reg_tradeoff/tradeoff.csv"), ("C", "results/reg_tradeoff_C/tradeoff.csv"),
                   ("M", "results/reg_tradeoff_M/tradeoff.csv")] + [
                      (p, f"results/reg_tradeoff_{p}/tradeoff.csv") for p in HUMANS
                      if os.path.exists(f"results/reg_tradeoff_{p}/tradeoff.csv")]:
    d = pd.read_csv(path)
    same = d[d.gap_target == 0].groupby("alpha").r2.median()
    a_rule = max(a for a, v in same.items() if v >= 0.99 * same.max())
    w = d.pivot_table(index=["train", "gap_target"], columns="alpha", values="r2").reset_index()
    x = w[w.gap_target > 0].assign(g=lambda q: q[a_rule] - q[0.1]).groupby("train").g.mean()
    sd = w[w.gap_target == 0].assign(g=lambda q: q[a_rule] - q[0.1]).set_index("train").g
    t = stats.wilcoxon(x)
    out.setdefault("regularization", {})[name] = {
        "alpha_rule": a_rule, "n_sessions": int(len(x)), "median_crossday_gain": float(np.median(x)),
        "W": float(t.statistic), "P": float(t.pvalue), "median_sameday_change": float(np.median(sd)),
        "frac_sessions_positive": float((x > 0).mean())}

# 4. Harmful recalibration: CV vs history, Fisher exact on pair counts per budget
p = pd.read_csv("results/recal_policy/policies.csv")
for n, q in p.groupby("n"):
    a = int((q.cv < q.renorm - 0.01).sum())
    b = int((q.history < q.renorm - 0.01).sum())
    f = stats.fisher_exact([[a, len(q) - a], [b, len(q) - b]])
    out.setdefault("harmful", {})[int(n)] = {"n_pairs": int(len(q)), "cv": a, "history": b, "odds_ratio": float(f[0]) if np.isfinite(f[0]) else None, "P": float(f[1])}

# 5. Edge effect: per-array one-sided Mann-Whitney P (from braingate_failure) combined with Fisher's method; sign test
pa = pd.read_csv("results/braingate_failure/per_array.csv")
e = pa.dropna(subset=["edge_vs_interior_p"])
fi = stats.combine_pvalues(e.edge_vs_interior_p)
out["edge_humans"] = {"n_arrays": int(len(e)), "fisher_chi2": float(fi.statistic), "df": 2 * int(len(e)), "P": float(fi.pvalue),
                      "n_individually_p05": int((e.edge_vs_interior_p < 0.05).sum())}
e10 = e[e.initially_active >= 10]
neg = int((e10.edge_slope < e10.interior_slope).sum())
out["edge_sign_test_arrays_ge10_active"] = {"n": int(len(e10)), "edge_faster": neg,
                                            "P_one_sided": float(stats.binomtest(neg, len(e10), 0.5, alternative="greater").pvalue)}
out["revival_humans"] = {"revived": int(pa.revived.sum()), "silenced": int(pa.silenced.sum())}
imp = pa.impedance_spearman_vs_day.dropna()
out["impedance_humans"] = {"n_arrays": int(len(imp)), "declining": int((imp < 0).sum()), "median_rho": float(imp.median())}
json.dump(out, open("results/paper_stats.json", "w"), indent=1)
print(json.dumps(out, indent=1))
