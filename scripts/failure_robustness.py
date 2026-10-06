"""Robustness of the electrode-failure findings in 20 human Utah arrays (BrainGate yield data).

Revival controls (silenced = initially active electrode below 2 Hz for >= 3 consecutive *observed* sessions; revived =
later >= 4 Hz for >= 3 consecutive observed sessions; no imputation of missing sessions in any variant):
  base          -4.5 RMS threshold, as in the paper but without forward/back filling
  rms_3.5/5.5   other RMS multiples
  fixed_uv      a fixed absolute threshold per electrode (its -4.5 RMS value in microvolts over the first three
                sessions); in each session the RMS multiple whose threshold is closest is used, so changes in
                noise level cannot move the effective threshold
  no_global     sessions with an array-wide shift (|median log-rate change| > 1) removed
  deep          silence must be < 0.5 Hz (not just < 2 Hz) and revival >= 4 Hz
  waveform      active also requires a mean-waveform trough of at least 30 microvolts
  long_silence  silent run must last >= 30 days
  shuffle_*     same statistics after randomly permuting each electrode's session order (expected if
                session values had no temporal structure); computed for base, deep, long_silence and fixed_uv
Edge controls (initially active electrodes only; per-array regression of log-rate slope on edge + initial log rate;
array-level sign and Wilcoxon tests, so electrodes within an array are not treated as independent).

Usage: python scripts/failure_robustness.py [--root D:/ibci-data/braingate] [--out results/failure_robustness]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

TH = ["-3", "-3.5", "-4", "-4.5", "-5", "-5.5"]


def load_all(root, cache):
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "yield", "**", "*_yield.mat"), recursive=True)):
        d = loadmat(f, simplify_cells=True)
        el = pd.DataFrame(d["electrodes"])[["electrode_id", "array", "x", "y"]]
        fr = pd.DataFrame(d["firing_rates"])
        fr["th"] = fr.threshold_description.astype(str).str.replace("0$", "", regex=True).str.rstrip(".")
        piv_r = fr.pivot_table(index="electrode_id", columns="th", values="firing_rate")
        piv_v = fr.pivot_table(index="electrode_id", columns="th", values="threshold_value")
        el = el.set_index("electrode_id")
        for t in TH:
            el[f"r{t}"] = piv_r.get(t)
            el[f"v{t}"] = piv_v.get(t)
        wf = d.get("spike_waveforms", {})
        if isinstance(wf, dict) and "mean_waveforms" in wf:
            amp = {int(e): float(np.min(w)) for e, w in zip(np.atleast_1d(wf["electrode_id"]), wf["mean_waveforms"])}
            el["wf"] = pd.Series(amp)
        el["day"] = int(d["post_implant_day"])
        el["participant"] = str(d["participant_id"])
        rows.append(el.reset_index())
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(cache, index=False)
    return df


def runs(mask, k=3):
    """True at index i if mask[i:i+k] are all True."""
    m = np.asarray(mask, bool)
    if len(m) < k:
        return np.zeros(len(m), bool)
    out = np.zeros(len(m), bool)
    out[: len(m) - k + 1] = np.all(np.lib.stride_tricks.sliding_window_view(m, k), axis=1)
    return out


def revival_counts(series, days, lo=2.0, hi=4.0, silent_max=None, min_silent_days=0):
    """series: dict electrode -> (rates over observed sessions). Returns (silenced, revived)."""
    sil = rev = 0
    for e, (r, dd) in series.items():
        if len(r) < 7 or np.nanmean(r[:3]) < lo:
            continue
        below = runs(r < (silent_max if silent_max is not None else lo))
        if not below.any():
            continue
        i0 = int(np.argmax(below))
        j = i0
        while j < len(r) and r[j] < lo:
            j += 1
        if dd[min(j, len(r)) - 1] - dd[i0] < min_silent_days:
            continue
        sil += 1
        rev += int(runs(r[i0:] >= hi).any())
    return sil, rev


def electrode_series(g, col, keep_sessions=None, extra_mask=None):
    out = {}
    for e, q in g.groupby("electrode_id"):
        q = q.sort_values("day")
        if keep_sessions is not None:
            q = q[q.day.isin(keep_sessions)]
        r = q[col].to_numpy(float)
        if extra_mask is not None:
            r = np.where(extra_mask(q), r, 0.0)
        ok = np.isfinite(r)
        out[e] = (r[ok], q.day.to_numpy()[ok])
    return out


def fixed_uv_rate(g):
    """Rate at the RMS multiple whose microvolt threshold is closest to the electrode's initial -4.5 RMS value."""
    g = g.sort_values("day").copy()
    first_days = np.sort(g.day.unique())[:3]
    ref = g[g.day.isin(first_days)].groupby("electrode_id")["v-4.5"].median()
    V = g[[f"v{t}" for t in TH]].to_numpy(float)
    R = g[[f"r{t}" for t in TH]].to_numpy(float)
    target = g.electrode_id.map(ref).to_numpy(float)[:, None]
    idx = np.nanargmin(np.abs(np.nan_to_num(V, nan=1e9) - target), axis=1)
    g["r_fixed"] = R[np.arange(len(g)), idx]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ.get("IBCI_DATA", "D:/ibci-data"), "braingate"))
    ap.add_argument("--out", default="results/failure_robustness")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load_all(args.root, os.path.join(args.out, "electrode_thresholds.csv.gz"))
    rng = np.random.default_rng(0)
    rev_rows, edge_rows = [], []
    for (p, a), g in df.groupby(["participant", "array"]):
        g = fixed_uv_rate(g)
        # array-wide shifts: median log-rate change between consecutive sessions
        med = g.groupby("day")["r-4.5"].median().sort_index()
        dl = np.diff(np.log(med.to_numpy() + 0.1))
        bad = set(med.index[1:][np.abs(dl) > 1])
        keep = [d for d in med.index if d not in bad]
        variants = {
            "base": electrode_series(g, "r-4.5"),
            "rms_3.5": electrode_series(g, "r-3.5"),
            "rms_5.5": electrode_series(g, "r-5.5"),
            "fixed_uv": electrode_series(g, "r_fixed"),
            "no_global": electrode_series(g, "r-4.5", keep_sessions=keep),
            "waveform": electrode_series(g, "r-4.5", extra_mask=lambda q: (q.wf.to_numpy(float) <= -30)),
        }
        row = dict(participant=p, array=a)
        for k, s in variants.items():
            row[f"{k}_sil"], row[f"{k}_rev"] = revival_counts(s, None)
        row["deep_sil"], row["deep_rev"] = revival_counts(variants["base"], None, silent_max=0.5)
        row["long_silence_sil"], row["long_silence_rev"] = revival_counts(variants["base"], None, min_silent_days=30)
        shuf = {e: (rng.permutation(r), d) for e, (r, d) in variants["base"].items()}
        row["shuffle_null_sil"], row["shuffle_null_rev"] = revival_counts(shuf, None)
        row["shuffle_deep_sil"], row["shuffle_deep_rev"] = revival_counts(shuf, None, silent_max=0.5)
        row["shuffle_long_sil"], row["shuffle_long_rev"] = revival_counts(shuf, None, min_silent_days=30)
        shuf_f = {e: (rng.permutation(r), d) for e, (r, d) in variants["fixed_uv"].items()}
        row["shuffle_fixed_uv_sil"], row["shuffle_fixed_uv_rev"] = revival_counts(shuf_f, None)
        rev_rows.append(row)

        # edge: initially active electrodes only, slope of log rate per year, adjusted for initial log rate
        q = g.pivot_table(index="day", columns="electrode_id", values="r-4.5").sort_index()
        yrs = q.index.to_numpy(float) / 365.25
        xy = g.drop_duplicates("electrode_id").set_index("electrode_id").loc[q.columns]
        init = q.iloc[:3].mean()
        act = (init >= 2.0).to_numpy()
        if act.sum() < 10:
            continue
        slopes, r0 = [], []
        for c in q.columns[act]:
            v = q[c].to_numpy(float)
            ok = np.isfinite(v)
            slopes.append(stats.linregress(yrs[ok], np.log(v[ok] + 0.1)).slope)
            r0.append(np.log(init[c] + 0.1))
        slopes, r0 = np.array(slopes), np.array(r0)
        edge = (xy.x.isin([0, 9]) | xy.y.isin([0, 9])).to_numpy()[act]
        dist = np.maximum(np.abs(xy.x.to_numpy() - 4.5), np.abs(xy.y.to_numpy() - 4.5))[act]
        if edge.sum() < 3 or (~edge).sum() < 3:
            continue
        X = np.c_[np.ones(len(slopes)), edge.astype(float), r0]
        beta = np.linalg.lstsq(X, slopes, rcond=None)[0]
        edge_rows.append(dict(participant=p, array=a, n_active=int(act.sum()), n_edge=int(edge.sum()),
                              edge_minus_interior=float(np.median(slopes[edge]) - np.median(slopes[~edge])),
                              edge_coef_adj_initial_rate=float(beta[1]),
                              rho_distance_slope=float(stats.spearmanr(dist, slopes).correlation)))
    rev = pd.DataFrame(rev_rows)
    edge = pd.DataFrame(edge_rows)
    rev.to_csv(os.path.join(args.out, "revival_variants.csv"), index=False)
    edge.to_csv(os.path.join(args.out, "edge_arrays.csv"), index=False)
    summ = {"revival": {}, "edge": {}}
    for k in ["base", "rms_3.5", "rms_5.5", "fixed_uv", "no_global", "waveform", "deep", "long_silence", "shuffle_null", "shuffle_deep",
              "shuffle_long", "shuffle_fixed_uv"]:
        s, r = int(rev[f"{k}_sil"].sum()), int(rev[f"{k}_rev"].sum())
        summ["revival"][k] = {"silenced": s, "revived": r, "fraction": r / s if s else None,
                              "arrays_with_silenced": int((rev[f"{k}_sil"] > 0).sum())}
    for col in ("edge_minus_interior", "edge_coef_adj_initial_rate", "rho_distance_slope"):
        v = edge[col].dropna()
        neg = int((v < 0).sum())
        summ["edge"][col] = {"n_arrays": int(len(v)), "negative": neg, "median": float(v.median()),
                             "sign_test_P_one_sided": float(stats.binomtest(neg, len(v), 0.5, alternative="greater").pvalue),
                             "wilcoxon_P_one_sided": float(stats.wilcoxon(v, alternative="less").pvalue)}
    json.dump(summ, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
