"""Supplementary tables for the manuscript, generated from results/ (LaTeX tabular bodies in manuscript/supp/).

Every number in the Supplementary Information comes from this script, so tables and result files cannot disagree.

Usage: python scripts/make_supp_tables.py [--out manuscript/supp]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.stats import cluster_bootstrap  # noqa: E402

LADDERS = {"Monkey N": "results/anatomy/ladder.csv", "Monkey C": "results/replication/C_ladder.csv",
           "Monkey M": "results/replication/M_ladder.csv", "Human T6": "results/replication_bg/T6_ladder.csv"}
LADDERS.update({f"Human {p}": f"results/replication_bg/{p}_ladder.csv" for p in ("T5", "T9")
                if os.path.exists(f"results/replication_bg/{p}_ladder.csv")})
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
        d = pd.read_csv(p)
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
    rows = []
    for n, q in p.groupby("n"):
        h = st[str(int(n))]
        cells = [str(int(n)), str(len(q)), f"{100 * h['cv'] / len(q):.1f}", f"{100 * h['history'] / len(q):.1f}",
                 f"{h['P']:.3f}" if h["P"] < 1 else "1.0"]
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
    names = {"N (LINK)": "Monkey N", "C": "Monkey C", "M": "Monkey M", "T6 (human)": "Human T6",
             "T5 (human)": "Human T5", "T9 (human)": "Human T9"}
    rows = []
    for _, x in r.iterrows():
        lo, hi = [float(v) for v in x.ci.strip("[]").split(",")]
        rows.append(" & ".join([names[x.subject], f"$10^{{{int(np.log10(x.alpha_rule))}}}$",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manuscript/supp")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for fn in (ladder_table, failure_table, efficiency_table, policy_table, targeted_table, sim_table, augment_table,
               mindful_table, reg_table, lstm_table):
        fn(args.out)
        print("wrote", fn.__name__, flush=True)


if __name__ == "__main__":
    main()
