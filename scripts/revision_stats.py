"""Additional tests requested in review (written to results/revision_stats.json).

1. Regularization: the 1% rule compared with each training session's own best same-day penalty (an optimistic
   same-day choice, since it uses the held-out same-day trials), and with a per-session version of the 1% rule.
2. Harmful recalibration: exact McNemar tests on paired outcomes (cross-validation vs history policy), Holm-adjusted
   over the five data budgets, and the shrinkage values the history policy chose.
3. Long electrode silences: revival after >= 30-day silence versus the shuffled-order null (Fisher's exact test).
4. True one-day versus two-day pairs in the 'overnight' set, per subject.
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
SUBJ = {"N": "results/reg_tradeoff/tradeoff.csv", "C": "results/reg_tradeoff_C/tradeoff.csv",
        "M": "results/reg_tradeoff_M/tradeoff.csv"}
for p in humans():
    if os.path.exists(f"results/reg_tradeoff_{p}/tradeoff.csv"):
        SUBJ[p] = f"results/reg_tradeoff_{p}/tradeoff.csv"
for name, path in SUBJ.items():
    d = pd.read_csv(path)
    same_med = d[d.gap_target == 0].groupby("alpha").r2.median()
    a_rule = max(a for a, v in same_med.items() if v >= 0.99 * same_med.max())
    a_best_med = same_med.idxmax()
    sd = d[d.gap_target == 0].pivot_table(index="train", columns="alpha", values="r2")
    a_best = sd.idxmax(axis=1)                                    # per-session best same-day alpha
    a_rule_s = sd.apply(lambda r: max(a for a, v in r.items() if v >= r.max() - 0.01 * abs(r.max())), axis=1)
    cd = d[d.gap_target > 0].pivot_table(index=["train", "gap_target"], columns="alpha", values="r2").reset_index()
    cd["best"] = [r[a_best[r.train]] for _, r in cd.iterrows()]
    cd["rule_s"] = [r[a_rule_s[r.train]] for _, r in cd.iterrows()]
    cd["rule"] = cd[a_rule]
    res = {"alpha_rule_median_curve": a_rule, "alpha_best_median_curve": float(a_best_med),
           "per_session_best_alpha_counts": {str(k): int(v) for k, v in a_best.value_counts().items()},
           "per_session_rule_alpha_counts": {str(k): int(v) for k, v in a_rule_s.value_counts().items()}}
    for lab, col in (("rule_vs_sessionbest", "rule"), ("sessionrule_vs_sessionbest", "rule_s")):
        g = cd.assign(gain=cd[col] - cd.best).groupby("train").gain.mean()
        w = stats.wilcoxon(g) if (g != 0).any() else None
        res[lab] = {"n_sessions": int(len(g)), "median_gain": float(g.median()), "frac_positive": float((g > 0).mean()),
                    "P": float(w.pvalue) if w else None}
    out.setdefault("regularization", {})[name] = res

p = pd.read_csv("results/recal_policy/policies.csv")
ps = []
for n, q in p.groupby("n"):
    a = (q.cv < q.renorm - 0.01).to_numpy()
    b = (q.history < q.renorm - 0.01).to_numpy()
    b01, b10 = int((a & ~b).sum()), int((~a & b).sum())
    pv = float(stats.binomtest(b01, b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0
    ps.append(pv)
    out.setdefault("mcnemar", {})[int(n)] = {"cv_only_harmful": b01, "history_only_harmful": b10, "P_exact": pv}
order = np.argsort(ps)
m = len(ps)
adj = np.minimum(1, np.maximum.accumulate([ps[i] * (m - k) for k, i in enumerate(order)]))
for k, i in enumerate(order):
    out["mcnemar"][int(sorted(p.n.unique())[i])]["P_holm"] = float(adj[k])

r = json.load(open("results/failure_robustness/summary.json"))["revival"]
a, b = r["long_silence"], r["shuffle_long"]
f = stats.fisher_exact([[a["revived"], a["silenced"] - a["revived"]], [b["revived"], b["silenced"] - b["revived"]]])
out["long_silence_vs_shuffle"] = {"observed": f"{a['revived']}/{a['silenced']}", "shuffle": f"{b['revived']}/{b['silenced']}",
                                  "odds_ratio": float(f[0]), "P": float(f[1])}

LAD = {"N": "results/anatomy/ladder.csv", "C": "results/replication/C_ladder.csv", "M": "results/replication/M_ladder.csv"}
for p_ in humans():
    if os.path.exists(f"results/replication_bg/{p_}_ladder.csv"):
        LAD[p_] = f"results/replication_bg/{p_}_ladder.csv"
for name, path in LAD.items():
    d = valid_pairs(pd.read_csv(path))
    d = d[d.gap_target == 1].assign(r=lambda x: x.L2 / x.own)
    out.setdefault("one_vs_two_day", {})[name] = {
        f"{int(k)}d": {"n": int(len(g)), "median_retention": float(g.r.median())} for k, g in d.groupby("days")}
json.dump(out, open("results/revision_stats.json", "w"), indent=1)
print(json.dumps(out, indent=1))
