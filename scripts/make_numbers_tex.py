"""Write manuscript/numbers.tex: every number quoted in the v2 manuscript, generated from the result files.

The manuscript uses \\val{key}; a key missing here prints as a red '??' in the PDF (and is listed by this script's
--check option), so stale or missing numbers cannot slip into the text unnoticed.

Usage: python scripts/make_numbers_tex.py [--check manuscript/natcomms/main.tex] [--out FILE ...]
"""
import argparse
import json
import os
import re

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.participants import excluded_humans, humans  # noqa: E402

HUM = humans()   # included participants (pre-specified reference-decoder rule)
IND = ["N", "C", "M"] + HUM
GAPS = [1, 7, 30, 120, 480]
V = {}


def f2(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.2f}".replace("-", "$-$")


def f3(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.3f}".replace("-", "$-$")


def pct(x):
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.0f}".replace("-", "$-$")


def pval(p):
    if p is None or not np.isfinite(p):
        return "--"
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def days(x):
    if x is None or not np.isfinite(x):
        return "more than 480"
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def put(k, v):
    V[k] = v


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default=None)
    ap.add_argument("--out", nargs="+", default=["manuscript/natcomms/numbers.tex"])
    args = ap.parse_args()
    s = load("results/v2_summary.json")
    for k in IND:
        if k not in s:
            continue
        r = s[k]
        d = r["default"]
        put(f"{k}:pairs", str(d["n_pairs"]))
        put(f"{k}:ntrain", str(d["n_train"]))
        put(f"{k}:thalf", days(d["t_half_days"]))
        g1 = d.get("gap1", {}).get("retention")
        if g1:
            put(f"{k}:ret1", f2(g1[0]))
            put(f"{k}:ret1ci", f"{f2(g1[1])}--{f2(g1[2])}")
        for dd in (1, 2):
            e = d.get(f"strict_{dd}day")
            if e:
                put(f"{k}:strict{dd}n", str(e["n"]))
                if e["retention"]:
                    put(f"{k}:strict{dd}", f2(e["retention"][0]))
                    put(f"{k}:strict{dd}ci", f"{f2(e['retention'][1])}--{f2(e['retention'][2])}")
        rets = [d[f"gap{g}"]["retention"][0] for g in GAPS if d.get(f"gap{g}", {}).get("retention")]
        recs = [d[f"gap{g}"].get("loss_recovered_by_gains") for g in GAPS if f"gap{g}" in d]
        rg = [d[f"gap{g}"].get("retention_gains") for g in GAPS if f"gap{g}" in d]
        for g in GAPS:
            if f"gap{g}" in d and d[f"gap{g}"].get("retention"):
                put(f"{k}:ret{g}", f2(d[f"gap{g}"]["retention"][0]))
                put(f"{k}:gainret{g}", f2(d[f"gap{g}"].get("retention_gains")))
                put(f"{k}:rec{g}", pct(d[f"gap{g}"].get("loss_recovered_by_gains")))
        if recs:
            put(f"{k}:recmin", pct(min(recs)))
            put(f"{k}:recmax", pct(max(recs)))
            put(f"{k}:gainretmin", f2(min(rg)))
            put(f"{k}:gainretmax", f2(max(rg)))
        if "tuned" in r:
            put(f"{k}:thalftuned", days(r["tuned"]["t_half_days"]))
            for g in GAPS:
                e = r["tuned"].get(f"gap{g}", {}).get("retention")
                if e:
                    put(f"{k}:tret{g}", f2(e[0]))
            t = r.get("tuned_vs_default_retention") or {}
            a = r.get("tuned_vs_default_abs_r2") or {}
            put(f"{k}:tunedretdiff", f2(t.get("median_diff")))
            put(f"{k}:tunedretp", pval(t.get("P")))
            put(f"{k}:tunedabsdiff", f3(a.get("median_diff")))
            put(f"{k}:tunedabsp", pval(a.get("P")))
            put(f"{k}:tunedfrac", pct(a.get("frac_positive")))
        if "network" in r:
            n = r["network"]
            put(f"{k}:nnpairs", str(n["n_pairs"]))
            put(f"{k}:nnown", f2(n["own_nn_median"]))
            put(f"{k}:linown", f2(n["own_linear_median"]))
            if n["own_minus_linear"]:
                put(f"{k}:nnowndiff", f2(n["own_minus_linear"][0]))
                put(f"{k}:nnowndiffci", f"{f2(n['own_minus_linear'][1])}--{f2(n['own_minus_linear'][2])}")
            rr = n.get("retention_nn_vs_linear") or {}
            put(f"{k}:nnretdiff", f2(rr.get("median_diff")))
            put(f"{k}:nnretp", pval(rr.get("P")))
            bg = n["by_gap"]
            rn = [v["rec_nn"] for v in bg.values()]
            put(f"{k}:nnrecmin", pct(min(rn)))
            put(f"{k}:nnrecmax", pct(max(rn)))
        for blk, items in (("alignment", [("rotation_on_mean_updated", "rotmean"), ("rotation_on_renormalized", "rotren"),
                                          ("renorm_vs_mean_only", "renmean")]),
                           ("label_free_plus", [("coral_vs_renorm", "coral"), ("fa_stab_vs_fa_fixed", "stabfix"),
                                                ("fa_stab_vs_renorm", "stabren")])):
            if blk in r:
                for src, key in items:
                    e = r[blk].get(src) or {}
                    put(f"{k}:{key}diff", f3(e.get("median_diff")))
                    put(f"{k}:{key}p", pval(e.get("P")))
                    put(f"{k}:{key}n", str(e.get("n_sessions", "--")))
                    put(f"{k}:{key}frac", pct(e.get("frac_positive")))
        if "data_efficiency" in r:
            for c, v in r["data_efficiency"]["median_retention"].items():
                put(f"{k}:eff{c.replace('_', '')}", f2(v))

    # pooled human data efficiency
    frames = [pd.read_csv(f"results/gain_efficiency/{k}.csv") for k in HUM
              if os.path.exists(f"results/gain_efficiency/{k}.csv")]
    if frames:
        e = pd.concat(frames)
        put("hum:effpairs", str(len(e)))
        put("hum:effnone", f2(float((e.L2 / e.own).median())))
        for n in (10, 20, 50, 100, 300):
            for rung in ("L3", "L6p", "L6"):
                c = f"{rung}_n{n}"
                if c in e:
                    put(f"hum:eff{rung}n{n}", f2(float((e[c] / e.own).median())))
                    put(f"hum:effpct{rung}n{n}", pct(float((e[c] / e.own).median())))
    # weight angles: reduction of the decoder-weight angle when one scale per channel is allowed
    red_all = []
    for k in IND:
        p = f"results/weight_angles/{k}.csv"
        if os.path.exists(p):
            q = pd.read_csv(p)
            med = q.groupby("gap_target")[["angle_global", "angle_channel", "angle_null"]].median()
            red = med.angle_global - med.angle_channel
            red_all += list(red.values)
            put(f"{k}:angred", f"{red.median():.0f}")
            put(f"{k}:anggl1", f"{med.angle_global.iloc[0]:.0f}")
            put(f"{k}:anggl480", f"{med.angle_global.iloc[-1]:.0f}")
            put(f"{k}:angch1", f"{med.angle_channel.iloc[0]:.0f}")
            put(f"{k}:angch480", f"{med.angle_channel.iloc[-1]:.0f}")
            put(f"{k}:angnull", f"{med.angle_null.median():.0f}")
    if red_all:
        put("ang:redmin", f"{min(red_all):.0f}")
        put("ang:redmax", f"{max(red_all):.0f}")
    # matched-output control
    for k in ("C", "M"):
        p = f"results/matched_output_tuned/{k}.csv" if os.path.exists(f"results/matched_output_tuned/{k}.csv") else f"results/matched_output/{k}.csv"
        if os.path.exists(p):
            q = pd.read_csv(p).assign(rec=lambda x: (x.L3_n300 - x.L2) / (x.own - x.L2))
            med = q.groupby("gap_target").rec.median()
            put(f"{k}:matchrecmin", pct(med.min()))
            put(f"{k}:matchrecmax", pct(med.max()))
            put(f"{k}:matchrec480", pct(med.get(480, np.nan)))
            put(f"{k}:matchrec1", pct(med.get(1, np.nan)))
    # regularization
    rule = pd.read_csv("results/reg_tradeoff/rule_across_subjects.csv")
    names = {"N (LINK)": "N", "C": "C", "M": "M", **{f"{p} (human)": p for p in HUM}}
    ps = load("results/paper_stats.json")
    rs = load("results/revision_stats.json")
    for full, k in names.items():
        q = rule[rule.subject == full]
        if not len(q):
            continue
        a = q.alpha_rule.iloc[0]
        put(f"{k}:rulealpha", f"10^{{{int(np.log10(a))}}}" if np.log10(a) == int(np.log10(a)) else f"{a / 10 ** int(np.log10(a)):.0f}\\times10^{{{int(np.log10(a))}}}")
        put(f"{k}:rulesame", f3(float(q[q.gap == 0].gain.iloc[0])))
        cd = q[q.gap > 0]
        put(f"{k}:rulemin", f3(float(cd.gain.min())))
        put(f"{k}:rulemax", f3(float(cd.gain.max())))
        reg = ps.get("regularization", {}).get(k, {})
        if reg:
            put(f"{k}:rulemed", f3(reg["median_crossday_gain"]))
            put(f"{k}:rulep", pval(reg["P"]))
            put(f"{k}:rulepos", str(int(round(reg["frac_sessions_positive"] * reg["n_sessions"]))))
            put(f"{k}:rulen", str(reg["n_sessions"]))
        rv = rs.get("regularization", {}).get(k, {})
        if rv:
            put(f"{k}:rulevsbest", f3(rv["rule_vs_sessionbest"]["median_gain"]))
            put(f"{k}:rulevsbestp", pval(rv["rule_vs_sessionbest"]["P"]))
    k_ = ps.get("overnight_kruskal", {})
    put("kw:H", f"{k_.get('H', np.nan):.1f}")
    put("kw:P", pval(k_.get("P")))
    put("kw:df", str(k_.get("df", "--")))
    for n, m in rs.get("mcnemar", {}).items():
        put(f"mcn:{n}:p", pval(m["P_exact"]))
        put(f"mcn:{n}:holm", pval(m["P_holm"]))
        put(f"mcn:{n}:cv", str(m["cv_only_harmful"]))
    h = ps.get("harmful", {})
    for n, m in h.items():
        put(f"harm:{n}:cv", f"{100 * m['cv'] / m['n_pairs']:.1f}")
        put(f"harm:{n}:hist", f"{100 * m['history'] / m['n_pairs']:.1f}")
    # electrodes
    fr = load("results/failure_robustness/summary.json")
    for k, v in fr.get("revival", {}).items():
        put(f"rev:{k}:frac", pct(v["fraction"]))
        put(f"rev:{k}:sil", f"{v['silenced']:,}")
        put(f"rev:{k}:rev", f"{v['revived']:,}")
    for k, v in fr.get("edge", {}).items():
        kk = {"edge_minus_interior": "raw", "edge_coef_adj_initial_rate": "adj", "rho_distance_slope": "dist"}[k]
        put(f"edge:{kk}:neg", str(v["negative"]))
        put(f"edge:{kk}:n", str(v["n_arrays"]))
        put(f"edge:{kk}:sign", pval(v["sign_test_P_one_sided"]))
        put(f"edge:{kk}:wil", pval(v["wilcoxon_P_one_sided"]))
        put(f"edge:{kk}:med", f2(v["median"]))
    ls = rs.get("long_silence_vs_shuffle", {})
    if ls:
        put("rev:long:P", pval(ls["P"]))
        put("rev:long:OR", f"{ls['odds_ratio']:.1f}")
    # simulator: calibration and held-out validation against two baselines
    import sys
    sys.path.insert(0, "scripts")
    from make_v2_figures import early_curve_baseline
    e = load("results/sim_calib_split/calibration.json")
    late = load("results/sim_calib_late/calibration.json")
    if e:
        put("sim:loss", f3(e["loss"]))
        put("sim:rms", f2(np.sqrt(e["loss"] / 15)))
        put("sim:losslate", f3(late["loss"]))
        put("sim:mix0", f2(e["params"]["s_mix0"]))
        put("sim:mix0late", f2(late["params"]["s_mix0"]))
        put("sim:taurho", f"{e['params']['tau_rho']:.0f}")
        put("sim:taurholate", f"{late['params']['tau_rho']:.0f}")
        put("sim:hoff", f"{e['params']['h_off']:.4f}")
        put("sim:hon", f"{e['params']['h_on']:.4f}")
    v = pd.read_csv("results/sim_validation/validation.csv")
    sv = v.groupby(["group", "kind"]).abs_err.median().unstack()
    base = early_curve_baseline()
    for g, key in (("targeted", "fit"), ("untargeted", "unfit"), ("efficiency", "budget")):
        put(f"sim:{key}:sim", f2(sv.loc[g, "sim"]))
        put(f"sim:{key}:nodrift", f2(sv.loc[g, "nodrift"]))
        if base is not None:
            put(f"sim:{key}:early", f2(base.get(g, np.nan)))
    aug = load("results/augment/summary.json")
    if aug:
        a = pd.read_csv("results/augment/augment.csv")
        for col in ("sim", "ridge_reg", "pert", "multi", "sim_reg"):
            if col in a:
                g = (a[col] - a.base)
                put(f"aug:{col}", f3(float(g.median())))
    # monitoring
    mt = load("results/mindful_test/summary.json")
    if mt:
        put("mon:days", f2(abs(mt["spearman_r2_logdays"])))
        put("mon:neuralraw", f2(abs(mt["kl_neural"]["raw_spearman_with_r2"])))
        put("mon:neuralpart", f2(abs(mt["kl_neural"]["partial_given_logdays"])))
        put("mon:outraw", f2(abs(mt["kl_output"]["raw_spearman_with_r2"])))
        put("mon:outpart", f2(abs(mt["kl_output"]["partial_given_logdays"])))
        put("mon:maedays", f3(mt["cv_days_only"]["mae"]))
        put("mon:maeboth", f3(mt["cv_days+kl"]["mae"]))
        put("mon:maekl", f3(mt["cv_kl_only"]["mae"]))
    mr = load("results/mindful_reanalysis/summary.json")
    for r in mr or []:
        p = r["participant"]
        put(f"{p}:mf:ndays", str(r["n_days"]))
        put(f"{p}:mf:win", str(r["n_windows"]))
        put(f"{p}:mf:rdays", f2(r["pearson_ae_days"]))
        put(f"{p}:mf:nraw", f2(r["kl_neural"]["pearson_with_ae"]))
        put(f"{p}:mf:npart", f2(r["kl_neural"]["partial_given_days"]))
        put(f"{p}:mf:oraw", f2(r["kl_output"]["pearson_with_ae"]))
        put(f"{p}:mf:opart", f2(r["kl_output"]["partial_given_days"]))
        for kk, lab in (("days", "days"), ("days+kl_neural", "dn"), ("days+kl_output", "do"), ("kl_output", "o"),
                        ("kl", "k")):
            put(f"{p}:mf:{lab}", f"{r['lodo_mae'][kk]:.1f}")
        put(f"{p}:mf:dayr", f2(r["day_level"]["partial_kl_ae_given_days"] if r["day_level"]["partial_kl_ae_given_days"]
                                is not None else np.nan))
    # half-life confidence intervals (decay_summary.csv: default penalty)
    ds = pd.read_csv("results/decay_summary.csv")
    for _, r in ds.iterrows():
        k = r.subject.split()[-1]
        lo, hi = [float(x) for x in str(r.t_half_ci).strip('[]').split(',')]
        put(f"{k}:thalfci", f"{days(lo)}--{days(hi)}")
    # tuned-penalty recovery by re-weighting
    for k in IND:
        t = s.get(k, {}).get("tuned")
        if t:
            rec = [t[f"gap{g}"].get("loss_recovered_by_gains") for g in GAPS if f"gap{g}" in t]
            rg = [t[f"gap{g}"].get("retention_gains") for g in GAPS if f"gap{g}" in t]
            put(f"{k}:trecmin", pct(min(rec)))
            put(f"{k}:trecmax", pct(max(rec)))
            put(f"{k}:tgainretmin", f2(min(rg)))
            put(f"{k}:tgainretmax", f2(max(rg)))
    # fitted weights versus electrode changes (humans)
    from scipy import stats as _st
    for k in HUM:
        p = f"results/gain_mechanism/{k}.csv"
        if os.path.exists(p):
            q = pd.read_csv(p)
            for name in ("rate", "imp"):
                c = f"rho_{name}"
                if c in q and q[c].notna().sum() >= 6:
                    sv = q.groupby("train")[c].mean().dropna()
                    put(f"{k}:mech:{name}:rho", f2(float(sv.median())))
                    put(f"{k}:mech:{name}:pos", pct(float((sv > 0).mean())))
                    put(f"{k}:mech:{name}:n", str(len(sv)))
                    put(f"{k}:mech:{name}:p", pval(float(_st.wilcoxon(sv).pvalue)))
                    put(f"{k}:mech:{name}:null", f2(float(q[c + "_null"].median())))
    nl = load("results/failure_robustness/null_summary.json")
    for k, v in nl.items():
        put(f"null:{k}:obs", pct(v["fraction_obs"]))
        put(f"null:{k}:mean", pct(v["fraction_null_mean"]))
        put(f"null:{k}:ci", f"{pct(v['fraction_null_ci'][0])}--{pct(v['fraction_null_ci'][1])}")
        put(f"null:{k}:p", pval(v["P_upper"]))
        put(f"null:{k}:sil", f"{v['silenced_obs']:,}")
        put(f"null:{k}:silnull", f"{v['silenced_null_mean']:.0f}")
        put(f"null:{k}:arrays", v["arrays_above_null_median"].replace("/", " of "))
    if os.path.exists("results/amplitude_T5.json"):
        am = load("results/amplitude_T5.json")
        put("amp:n", str(am["n_electrodes"]))
        put("amp:early", f"{am['early_median_uv']:.0f}")
        put("amp:late", f"{am['late_median_uv']:.0f}")
        put("amp:ratio", f"{am['ratio_median']:.2f}")
        put("amp:iqr", f"{am['ratio_q25']:.2f}--{am['ratio_q75']:.2f}")
        put("amp:smaller", str(am["n_smaller"]))
        put("amp:p", pval(am["wilcoxon_p"]))
    # re-weighting controls (true vs channel-permuted decoder, signed vs non-negative weights)
    for k in IND:
        p = f"results/reweight_controls/{k}.csv"
        if os.path.exists(p):
            q = pd.read_csv(p)
            for c in ("renorm", "rew", "rew_nonneg", "rew_perm", "rew_perm_nn"):
                put(f"{k}:ctl:{c}", f2(float((q[c] / q.own).median())))
            put(f"{k}:ctl:pairs", str(len(q)))
            put(f"{k}:ctl:zero", pct(float(q.nonneg_zero_frac.median())))
    group_keys()
    write(args)


def _num(s):
    """Parse a formatted value back to a number ('$-$0.12' -> -0.12); None if it is not numeric."""
    try:
        return float(str(s).replace("$-$", "-").replace("\\%", ""))
    except ValueError:
        return None


def _join(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def group_keys():
    """Summaries across the included human participants: count, list, and min/max (with who) of per-person values.

    Each group value is the min or max of the per-person value as printed, so the text and the per-person tables agree.
    """
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    put("hum:n", str(len(HUM)))
    put("hum:nword", words.get(len(HUM), str(len(HUM))))
    put("ind:nword", words.get(len(HUM) + 3, str(len(HUM) + 3)))
    put("hum:list", _join(HUM))
    ex = excluded_humans()
    put("hum:excln", str(len(ex)))
    put("hum:excllist", _join(sorted(ex)) if ex else "none")
    for k, r in ex.items():
        put(f"{k}:refr2", f2(r))
    single = ["thalf", "thalftuned", "strict1", "ret1", "nnown", "linown", "rulemed", "rulevsbest", "pairs", "ntrain",
              "mech:rate:rho", "mech:rate:pos", "mech:imp:rho", "angred", "nnowndiff"] + [f"ret{g}" for g in GAPS] + \
             [f"tret{g}" for g in GAPS] + [f"ctl:{c}" for c in ("renorm", "rew", "rew_perm", "rew_nonneg", "rew_perm_nn")]
    for s in single:
        vals = [(h, _num(V.get(f"{h}:{s}"))) for h in HUM]
        vals = [(h, v) for h, v in vals if v is not None]
        if not vals:
            continue
        lo, hi = min(vals, key=lambda x: x[1]), max(vals, key=lambda x: x[1])
        put(f"hum:{s}:min", V[f"{lo[0]}:{s}"])
        put(f"hum:{s}:max", V[f"{hi[0]}:{s}"])
        put(f"hum:{s}:minwho", lo[0])
        put(f"hum:{s}:maxwho", hi[0])
        put(f"hum:{s}:nval", str(len(vals)))
    for lo_s, hi_s, name in (("trecmin", "trecmax", "trec"), ("tgainretmin", "tgainretmax", "tgainret"),
                             ("nnrecmin", "nnrecmax", "nnrec"), ("recmin", "recmax", "rec"),
                             ("gainretmin", "gainretmax", "gainret")):
        los = [(h, _num(V.get(f"{h}:{lo_s}"))) for h in HUM]
        his = [(h, _num(V.get(f"{h}:{hi_s}"))) for h in HUM]
        los, his = [x for x in los if x[1] is not None], [x for x in his if x[1] is not None]
        if los and his:
            put(f"hum:{name}:min", V[f"{min(los, key=lambda x: x[1])[0]}:{lo_s}"])
            put(f"hum:{name}:max", V[f"{max(his, key=lambda x: x[1])[0]}:{hi_s}"])
    for name in ("rate", "imp"):          # how many participants show a significant weight-electrode link
        ps = [(h, V.get(f"{h}:mech:{name}:p")) for h in HUM if f"{h}:mech:{name}:p" in V]
        sig = [h for h, p in ps if p and ("times10" in p or (_num(p) is not None and _num(p) < 0.05))]
        put(f"hum:mech:{name}:ntested", str(len(ps)))
        put(f"hum:mech:{name}:nsig", str(len(sig)))
        put(f"hum:mech:{name}:sigwho", _join(sig) if sig else "none")


def write(args):
    lines = ["% Generated by scripts/make_numbers_tex.py from results/ -- do not edit by hand.",
             r"\makeatletter",
             r"\newcommand{\val}[1]{\ifcsname val@#1\endcsname\csname val@#1\endcsname\else"
             r"\textcolor{red}{\textbf{??#1}}\fi}"]
    for k, v in sorted(V.items()):
        lines.append(rf"\expandafter\def\csname val@{k}\endcsname{{{v}}}")
    lines.append(r"\makeatother")
    for out in args.out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{len(V)} numbers written to " + ", ".join(args.out))
    if args.check:
        used = set(re.findall(r"\\val\{([^}]+)\}", open(args.check, encoding="utf-8").read()))
        missing = sorted(used - set(V))
        print(f"{len(used)} keys used, {len(missing)} missing" + (": " + ", ".join(missing) if missing else ""))


if __name__ == "__main__":
    main()
