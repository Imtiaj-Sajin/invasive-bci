"""Do decay rates differ between individuals? Linear mixed model on the tuned-decoder pairs (results/decay_alpha).

Outcome: retention of the renormalized fixed decoder (L2 / own) on valid pairs, floored at 0 so that every failed
decoder (R2 <= 0) counts as no retention; the raw ratio has extreme tails when a broken decoder is divided by a small
reference (monkey C reaches -50). Predictor: x = log2(days between training and test session), centred at 30 days.
Fixed effects: an intercept and slope per individual (full model) or one shared curve (null). Random effects: crossed
intercepts for the training session and for the test session, since pairs share both. Models are fitted by maximum
likelihood and compared by likelihood-ratio tests:
  curves  : one curve for everyone      vs  one curve per individual  (df = 2 (K - 1))
  slopes  : individual levels, one slope vs  individual levels and slopes (df = K - 1)
for all nine individuals and for the human participants alone. Sensitivities: floor at -1, and the absolute loss
own - L2 (no ratio).

Usage: python scripts/heterogeneity_model.py [--out results/heterogeneity]
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.participants import humans, individuals, valid_pairs  # noqa: E402


def load():
    rows = []
    for k in individuals():
        d = valid_pairs(pd.read_csv(f"results/decay_alpha/{k}_alpha10000.csv"))
        rows.append(d.assign(subject=k, train=k + ":" + d.train.astype(str), test=k + ":" + d.test.astype(str)))
    d = pd.concat(rows, ignore_index=True)
    d["x"] = np.log2(d.days.clip(lower=1) / 30.0)
    d["ret"] = d.L2 / d.own
    return d


def fit(d, y, fixed):
    """ML fit with crossed random intercepts for training and test sessions (one group holds all pairs)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm(f"{y} ~ {fixed}", d, groups=np.ones(len(d)), re_formula="0",
                        vc_formula={"train": "0 + C(train)", "test": "0 + C(test)"})
        return m.fit(reml=False, method=["lbfgs", "powell"])


def lrt(small, big, df):
    stat = max(2 * (big.llf - small.llf), 0.0)
    return {"chi2": round(float(stat), 2), "df": int(df), "P": float(stats.chi2.sf(stat, df))}


def compare(d, y):
    K = d.subject.nunique()
    shared = fit(d, y, "x")
    levels = fit(d, y, "0 + C(subject) + x")
    full = fit(d, y, "0 + C(subject) + C(subject):x")
    slopes = {}
    for k in d.subject.unique():
        name = f"C(subject)[{k}]:x"
        b, se = float(full.params[name]), float(full.bse[name])
        slopes[k] = {"slope": round(b, 4), "ci": [round(b - 1.96 * se, 4), round(b + 1.96 * se, 4)],
                     "level_30d": round(float(full.params[f"C(subject)[{k}]"]), 3)}
    return {"n_pairs": int(len(d)), "n_individuals": int(K), "curves": lrt(shared, full, 2 * (K - 1)),
            "slopes": lrt(levels, full, K - 1), "per_individual": slopes,
            "var_train": round(float(full.vcomp[list(full.model.exog_vc.names).index("train")]), 4),
            "var_test": round(float(full.vcomp[list(full.model.exog_vc.names).index("test")]), 4),
            "var_resid": round(float(full.scale), 4)}


def kruskal_by_gap(d, y):
    """Distribution-free check that does not assume equal variances: at each gap, one value per training session (the
    mean of its pairs), Kruskal-Wallis test across individuals."""
    out = {}
    for g, q in d.groupby("gap_target"):
        per = q.groupby(["subject", "train"])[y].mean().reset_index()
        groups = [v[y].to_numpy() for _, v in per.groupby("subject") if len(v) >= 3]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
            out[int(g)] = {"H": round(float(h), 2), "P": float(p), "n_individuals": len(groups),
                           "n_sessions": int(sum(len(x) for x in groups))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/heterogeneity")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = load()
    d["ret0"] = d.ret.clip(lower=0)
    d["ret1"] = d.ret.clip(lower=-1)
    d["loss"] = d.own - d.L2
    hum = d[d.subject.isin(humans())]
    out = {"outcome": "retention L2/own floored at 0; x = log2(days/30); slope = change per doubling of the gap",
           "main": compare(d, "ret0"), "humans": compare(hum, "ret0"),
           "floor_minus1": compare(d, "ret1"), "abs_loss": compare(d, "loss"),
           "humans_abs_loss": compare(hum, "loss"), "humans_floor_minus1": compare(hum, "ret1")}
    out["kruskal_by_gap"] = kruskal_by_gap(d, "ret")             # rank-based: floor irrelevant
    out["kruskal_by_gap_humans"] = kruskal_by_gap(hum, "ret")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    for key in ("main", "humans", "floor_minus1", "humans_floor_minus1", "abs_loss", "humans_abs_loss"):
        r = out[key]
        print(f"{key:16s} n={r['n_pairs']} K={r['n_individuals']} curves chi2={r['curves']['chi2']} "
              f"P={r['curves']['P']:.2g}  slopes chi2={r['slopes']['chi2']} P={r['slopes']['P']:.2g}")
    for k, v in out["main"]["per_individual"].items():
        print(f"  {k:4s} retention at 30 d {v['level_30d']:.2f}, per doubling {v['slope']:+.3f} {v['ci']}")


if __name__ == "__main__":
    main()
