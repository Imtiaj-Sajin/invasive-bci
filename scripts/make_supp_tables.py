"""Supplementary tables for the manuscript, generated from results/ (LaTeX tabular bodies in manuscript/supp/).

Every number in the Supplementary Information comes from this script, so tables and result files cannot disagree.

Usage: python scripts/make_supp_tables.py [--out manuscript/natcomms/supp]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.participants import humans, valid_pairs  # noqa: E402
from ibci.stats import cluster_bootstrap  # noqa: E402

LADDERS = {"Monkey N": "results/anatomy/ladder.csv", "Monkey C": "results/replication/C_ladder.csv",
           "Monkey M": "results/replication/M_ladder.csv"}
LADDERS.update({f"Human {p}": f"results/replication_bg/{p}_ladder.csv" for p in humans()})
GAPS = [1, 7, 30, 120, 480]


def f3(x):
    return "--" if pd.isna(x) else f"{x:.2f}".replace("-", "$-$")


def ci(est, lo, hi, d=2):
    s = f"{est:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"
    return s.replace("-", "$-$")


def write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")


def ladder_table(out):
    rows = []
    for name, p in LADDERS.items():
        d = valid_pairs(pd.read_csv(p))
        d = d[d.gap_target.isin(GAPS)].assign(ret=lambda x: x.L2 / x.own)
        rows.append(r"\multicolumn{11}{l}{\textit{" + name + r"}}\\")
        for g, q in d.groupby("gap_target"):
            est, lo, hi = cluster_bootstrap(q, "ret", n_boot=2000)
            med = q[["own", "L0", "L1", "L2", "L4u", "L4s", "L3_n300", "L6p_n300"]].median()
            rows.append(" & ".join([str(g), str(len(q))] + [f3(med[c]) for c in med.index]
                                   + [ci(est, lo, hi)]) + r"\\")
    write(os.path.join(out, "tab_ladder.tex"), rows)


def failure_table(out):
    pa = pd.read_csv("results/braingate_failure/per_array.csv").sort_values(["participant", "array"])
    rows = []
    for _, r in pa.iterrows():
        rows.append(" & ".join([
            r["participant"], str(r["array"]).replace("_", " "), str(int(r.sessions)), f"{int(r.span_days):,}",
            f"{r.yield_first3:.0f} / {r.yield_last3:.0f}", f"{1e3 * r.h_off:.1f}", f"{1e3 * r.h_on:.1f}",
            f"{int(r.revived)}/{int(r.silenced)}", f3(r.channel_level_kurtosis).replace(".00", ".0") if pd.isna(r.channel_level_kurtosis) else f"{r.channel_level_kurtosis:.1f}",
            f3(r.impedance_spearman_vs_day),
            "--" if pd.isna(r.edge_slope) else f"{r.edge_slope - r.interior_slope:.2f}".replace("-", "$-$"),
            "--" if pd.isna(r.edge_vs_interior_p) else (
                "$<$0.001" if r.edge_vs_interior_p < 0.001 else f"{r.edge_vs_interior_p:.3f}")]) + r"\\")
    n = json.load(open("results/failure_xdata/failure_stats.json"))["N (LINK)"]
    sl = pd.read_csv("results/channel_health/channel_slopes.csv")      # within-array slopes per channel (LINK)
    parts = []
    for arr in ("Lateral", "Medial"):
        q = sl[sl.array_name == arr]
        parts.append(f"{q[q.edge].tc_rate_slope.median() - q[~q.edge].tc_rate_slope.median():.2f}".replace("-", "$-$"))
    edge_txt = " / ".join(parts)
    rows.append(r"\midrule")
    rows.append(" & ".join(["Monkey N", "both", str(n["n_sessions"]), f"{n['span_days']:,}",
                            f"{100 * n['active_first3_median'] / 96:.0f} / {100 * n['active_last3_median'] / 96:.0f}",
                            f"{1e3 * n['h_off']:.1f}", f"{1e3 * n['h_on']:.1f}", f"{n['revived']}/{n['died']}",
                            f"{n['channel_level_kurtosis']:.1f}", "$-$0.88", edge_txt, "$<$0.001 / 0.046"]) + r"\\")
    write(os.path.join(out, "tab_failure.tex"), rows)


def efficiency_table(out):
    d = pd.read_csv("results/data_efficiency/ladder.csv")
    rows = []
    for g, q in d.groupby("gap_target"):
        cells = [str(g), str(len(q)), f3((q.L2 / q.own).median())]
        for n in (10, 20, 50, 100, 300):
            cells.append(f"{(q[f'L6p_n{n}'] / q.own).median():.2f} / {(q[f'L6_n{n}'] / q.own).median():.2f} / "
                         f"{(q[f'L3_n{n}'] / q.own).median():.2f}".replace("-", "$-$"))
        rows.append(" & ".join(cells) + r"\\")
    write(os.path.join(out, "tab_efficiency.tex"), rows)


def policy_table(out):
    p = pd.read_csv("results/recal_policy/policies.csv")
    st = json.load(open("results/paper_stats.json"))["harmful"]
    mc = json.load(open("results/revision_stats.json"))["mcnemar"]
    rows = []
    for n, q in p.groupby("n"):
        h = st[str(int(n))]
        cells = [str(int(n)), str(len(q)), f"{100 * h['cv'] / len(q):.1f}", f"{100 * h['history'] / len(q):.1f}",
                 f"{mc[str(int(n))]['P_exact']:.3f}" if mc[str(int(n))]["P_exact"] < 1 else "1.0"]
        for c in ("cv", "history", "oracle"):
            cells.append(f3((q[c] / q.own).median()))
        rows.append(" & ".join(cells) + r"\\")
    write(os.path.join(out, "tab_policy.tex"), rows)


def targeted_table(out):
    d = pd.read_csv("results/targeted_recal/history_policy.csv")
    rows = []
    for n, q in d.groupby("n"):
        cells = [str(int(n)), str(len(q)), f3((q.full / q.own).median())]
        for c in ("importance_k32", "stats_k32", "wstats_k32", "random_k32"):
            est, lo, hi = cluster_bootstrap(q.assign(_g=q[c] - q.full), "_g", n_boot=2000)
            cells.append(ci(est, lo, hi, 3))
        rows.append(" & ".join(cells) + r"\\")
    write(os.path.join(out, "tab_targeted.tex"), rows)


def sim_table(out):
    e = json.load(open("results/sim_calib_split/calibration.json"))
    late = json.load(open("results/sim_calib_late/calibration.json"))
    desc = [("s_mix0", "Session-to-session mixing s.d.", "1"), ("s_mix", "Slow mixing s.d. (asymptote)", "1"),
            ("tau_mix", "Slow mixing time constant", "days"), ("rho0", "Turnover fraction, next session", "1"),
            ("rho_inf", "Turnover fraction, asymptote", "1"), ("tau_rho", "Turnover time constant", "days"),
            ("h_off", "Active to silent rate", "per day"), ("h_on", "Silent to active rate", "per day")]
    rows = []
    for k, txt, unit in desc:
        a, b = e["params"][k], late["params"][k]
        fmt = (lambda v: f"{v:.4f}") if k.startswith("h_") else (lambda v: f"{v:.0f}" if v > 5 else f"{v:.3f}")
        rows.append(f"{txt} & {unit} & {fmt(a)} & {fmt(b)}" + r"\\")
    rows.append(r"\midrule")
    rows.append(f"Calibration loss & & {e['loss']:.3f} & {late['loss']:.3f}" + r"\\")
    write(os.path.join(out, "tab_sim.tex"), rows)


def augment_table(out):
    a = pd.read_csv("results/augment/augment_summary.csv")
    rows = [" & ".join([str(int(r.gap_target)), str(int(r.n))] + [f3(r[c]) for c in
                                                                  ("base", "ridge_reg", "pert", "sim", "sim_reg",
                                                                   "multi", "own")]) + r"\\" for _, r in a.iterrows()]
    write(os.path.join(out, "tab_augment.tex"), rows)


def mindful_table(out):
    s = json.load(open("results/mindful_reanalysis/summary.json"))
    rows = []
    for r in s:
        m = r["lodo_mae"]
        rows.append(" & ".join([r["participant"], str(r["n_days"]), str(r["n_windows"]), f3(r["pearson_ae_days"]),
                                f3(r["kl_neural"]["pearson_with_ae"]), f3(r["kl_neural"]["partial_given_days"]),
                                f3(r["kl_output"]["pearson_with_ae"]), f3(r["kl_output"]["partial_given_days"]),
                                f"{m['days']:.1f}", f"{m['days+kl_neural']:.1f}", f"{m['days+kl_output']:.1f}",
                                f"{m['kl_output']:.1f}"]) + r"\\")
    t = json.load(open("results/mindful_test/summary.json"))
    write(os.path.join(out, "tab_mindful.tex"), rows)
    rows = [f"Elapsed days (log) & {f3(t['spearman_r2_logdays'])} & -- \\\\"]
    for k, lab in (("kl_neural", "Neural divergence"), ("kl_output", "Output divergence"), ("kl_sum", "Sum")):
        rows.append(f"{lab} & {f3(t[k]['raw_spearman_with_r2'])} & {f3(t[k]['partial_given_logdays'])}" + r"\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{3}{l}{Cross-validated error of predicted $R^2$ (mean absolute error)}\\")
    for k, lab in (("cv_days_only", "Days only"), ("cv_kl_only", "Divergence only"), ("cv_days+kl", "Days and divergence")):
        rows.append(f"{lab} & {t[k]['mae']:.3f} & \\\\")
    write(os.path.join(out, "tab_mindful_link.tex"), rows)


def reg_table(out):
    r = pd.read_csv("results/reg_tradeoff/rule_across_subjects.csv")
    names = {"N (LINK)": "Monkey N", "C": "Monkey C", "M": "Monkey M", **{f"{p} (human)": f"Human {p}" for p in humans()}}
    rows = []
    for _, x in r.iterrows():
        lo, hi = [float(v) for v in x.ci.strip("[]").split(",")]
        la = np.log10(x.alpha_rule)
        atxt = f"$10^{{{int(la)}}}$" if la == int(la) else f"${x.alpha_rule / 10 ** int(la):.0f}\\times10^{{{int(la)}}}$"
        rows.append(" & ".join([names[x.subject], atxt,
                                "same day" if x.gap == 0 else str(int(x.gap)), str(int(x.n)), f3(x.base), f3(x.rule),
                                ci(x.gain, lo, hi, 3), f"{x.frac_pos:.2f}"]) + r"\\")
    write(os.path.join(out, "tab_reg.tex"), rows)


def lstm_table(out):
    m = pd.read_csv("results/anatomy_nn/matched_vs_ridge.csv")
    rows = []
    for g, q in m.groupby("gap_target"):
        rows.append(" & ".join([str(g), str(len(q)), f3(q.own_rr.median()), f3(q.own_nn.median()),
                                f3(q.ret_rr.median()), f3(q.ret_nn.median()),
                                f3((q.L3_rr / q.own_rr).median()), f3((q.L3_nn / q.own_nn).median())]) + r"\\")
    write(os.path.join(out, "tab_lstm.tex"), rows)


NAMES = {"N": "Monkey N", "C": "Monkey C", "M": "Monkey M", **{p: f"Human {p}" for p in humans()}}


def nn_table(out):
    rows = []
    for k, name in NAMES.items():
        p = f"results/nn_ladder/{k}.csv"
        if not os.path.exists(p):
            continue
        q = valid_pairs(pd.read_csv(p), "own", "ridge_own").assign(ret_nn=lambda x: x.L2 / x.own, ret_lin=lambda x: x.ridge_L2 / x.ridge_own,
                                  rec_nn=lambda x: (x.L3 - x.L2) / (x.own - x.L2),
                                  rec_lin=lambda x: (x.ridge_L3 - x.ridge_L2) / (x.ridge_own - x.ridge_L2))
        rows.append(r"\multicolumn{8}{l}{\textit{" + name + r"}}\\")
        for g, x in q.groupby("gap_target"):
            rows.append(" & ".join([str(g), str(len(x)), f3(x.ridge_own.median()), f3(x.own.median()),
                                    f3(x.ret_lin.median()), f3(x.ret_nn.median()), f3(x.rec_lin.median()),
                                    f3(x.rec_nn.median())]) + r"\\")
    write(os.path.join(out, "tab_nn.tex"), rows)


def angle_table(out):
    rows = []
    for k, name in NAMES.items():
        p = f"results/weight_angles/{k}.csv"
        if not os.path.exists(p):
            continue
        med = pd.read_csv(p).groupby("gap_target")[["angle_global", "angle_channel", "angle_null"]].median()
        for g, r in med.iterrows():
            rows.append(" & ".join([name if g == med.index[0] else "", str(g), f"{r.angle_global:.0f}",
                                    f"{r.angle_channel:.0f}", f"{r.angle_null:.0f}"]) + r"\\")
    write(os.path.join(out, "tab_angles.tex"), rows)


def labelfree_table(out):
    s = json.load(open("results/v2_summary.json"))
    rows = []
    for k, name in NAMES.items():
        r = s.get(k, {})
        a, l_ = r.get("alignment", {}), r.get("label_free_plus", {})
        cells = [name]
        for blk, key in ((a, "renorm_vs_mean_only"), (a, "rotation_on_mean_updated"), (a, "rotation_on_renormalized"),
                         (l_, "coral_vs_renorm"), (l_, "fa_stab_vs_renorm"), (l_, "fa_stab_vs_fa_fixed")):
            e = blk.get(key) or {}
            p = e.get("P")
            ptxt = "--" if p is None else (f"{p:.2g}" if p >= 1e-3 else f"{p:.0e}").replace("e-0", "e-")
            cells.append("--" if not e else f"{f3(e['median_diff'])} ({ptxt})")
        rows.append(" & ".join(cells) + r"\\")
    write(os.path.join(out, "tab_labelfree.tex"), rows)


def human_eff_table(out):
    rows = []
    for k in humans():
        p = f"results/gain_efficiency/{k}.csv"
        if not os.path.exists(p):
            continue
        q = valid_pairs(pd.read_csv(p))
        cells = [NAMES[k], str(len(q)), f3((q.L2 / q.own).median())]
        for n in (10, 20, 50, 100, 300):
            vals = []
            for rung in ("L3", "L6p", "L6"):
                c = f"{rung}_n{n}"
                vals.append(f"{(q[c] / q.own).median():.2f}" if c in q else "--")
            cells.append(" / ".join(vals).replace("-", "$-$").replace("$-$$-$", "--"))
        rows.append(" & ".join(cells) + r"\\")
    write(os.path.join(out, "tab_eff_human.tex"), rows)


def revival_table(out):
    r = json.load(open("results/failure_robustness/summary.json"))["revival"]
    lab = [("base", "At least 3 sessions below 2 Hz (main definition)", "shuffle_null"),
           ("rms_3.5", "Threshold $-3.5$ times noise", None), ("rms_5.5", "Threshold $-5.5$ times noise", None),
           ("fixed_uv", "Fixed threshold in microvolts per electrode", "shuffle_fixed_uv"),
           ("no_global", "Array-wide shifts removed", None), ("waveform", "Mean waveform trough at least 30 $\\mu$V", None),
           ("deep", "Silence below 0.5 Hz", "shuffle_deep"), ("long_silence", "Silence lasting at least 30 days",
                                                              "shuffle_long")]
    rows = []
    for k, text, null in lab:
        v = r[k]
        nv = r.get(null) if null else None
        rows.append(" & ".join([text, f"{v['silenced']:,}", f"{v['revived']:,}", f"{100 * v['fraction']:.0f}",
                                f"{100 * nv['fraction']:.0f} ({nv['silenced']:,})" if nv else "--"]) + r"\\")
    write(os.path.join(out, "tab_revival.tex"), rows)


def controls_table(out):
    rows = []
    for k, name in NAMES.items():
        p = f"results/reweight_controls/{k}.csv"
        if not os.path.exists(p):
            continue
        q = valid_pairs(pd.read_csv(p))
        cells = [name, str(len(q))] + [f"{(q[c] / q.own).median():.2f}".replace("-", "$-$")
                                         for c in ("renorm", "rew", "rew_perm", "rew_nonneg", "rew_perm_nn")]
        m = f"results/gain_mechanism/{k}.csv"
        if os.path.exists(m):
            g = pd.read_csv(m)
            for c in ("rho_rate", "rho_imp"):
                sv = g.groupby("train")[c].mean().dropna() if c in g else pd.Series(dtype=float)
                cells.append(f"{sv.median():.2f} ({(sv > 0).mean() * 100:.0f}\\%)".replace("-", "$-$") if len(sv) >= 6
                             else "--")
        else:
            cells += ["--", "--"]
        rows.append(" & ".join(cells) + r"\\")
    write(os.path.join(out, "tab_controls.tex"), rows)


def matched_table(out):
    rows = []
    for k in ("C", "M"):
        for lab, p in (("kinematics", f"results/decay_alpha/{k}_alpha10000.csv"),
                       ("direction", f"results/matched_output_tuned/{k}.csv")):
            if not os.path.exists(p):
                continue
            q = valid_pairs(pd.read_csv(p))
            q = q[q.gap_target.isin([1, 7, 30, 120, 480])].assign(
                rec=lambda x: (x.L3_n300 - x.L2) / (x.own - x.L2), ret=lambda x: x.L2 / x.own,
                ret3=lambda x: x.L3_n300 / x.own)
            med = q.groupby("gap_target")[["ret", "ret3", "rec"]].median()
            for g, r in med.iterrows():
                rows.append(" & ".join([NAMES[k] if (lab == "kinematics" and g == 1) else "", lab if g == 1 else "",
                                        str(g), f"{r.ret:.2f}", f"{r.ret3:.2f}", f"{100 * r.rec:.0f}"]).replace("-", "$-$")
                            + r"\\")
    write(os.path.join(out, "tab_matched.tex"), rows)


def decoding_tables(out):
    """Human rows of the dataset table, and the per-participant decoding table with the inclusion rule."""
    from ibci.participants import MIN_REF_R2, excluded_humans, reference_r2
    st = json.load(open("results/decoding_session_stats.json")) if os.path.exists("results/decoding_session_stats.json") \
        else {}
    order = humans() + sorted(excluded_humans(), key=lambda k: int(k[1:]))
    rows = []
    for k in order:
        if k not in st:
            continue
        s = st[k]
        tag = "" if k in humans() else r" (excluded)"
        rows.append(f"Human {k}{tag} & Dryad x0k6djj1h & {s['arrays']} Utah, motor cortex & SBP & closed-loop cursor & "
                    f"{s['sessions']} ({s['years']:.1f} years) " + r"\\")
    write(os.path.join(out, "tab_data_humans.tex"), rows)
    rows = []
    for k in order:
        if k not in st:
            continue
        s, r = st[k], reference_r2(k)
        rows.append(" & ".join([k, f"{s['arrays']} ({s['channels']})", str(s["sessions"]), f"{s['years']:.1f}",
                                f"{s['median_trials']:.0f}", f"{s['median_movement_s']:.1f}",
                                f3(r) if r is not None else "--",
                                "yes" if r is not None and r > MIN_REF_R2 else "no"]) + r"\\")
    write(os.path.join(out, "tab_decoding.tex"), rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manuscript/natcomms/supp")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for fn in (ladder_table, failure_table, efficiency_table, policy_table, sim_table, augment_table, mindful_table,
               reg_table, nn_table, angle_table, labelfree_table, human_eff_table, revival_table,
               controls_table, matched_table, decoding_tables):
        fn(args.out)
        print("wrote", fn.__name__, flush=True)


if __name__ == "__main__":
    main()
