"""Per-channel health trajectories over 3.5 years of LINK, and the electrode failure process.

For every session and channel we compute:
  tc_rate    threshold-crossing rate (Hz) over the whole session
  sbp_mean   mean spiking-band power;  sbp_std  its standard deviation
  tune_r2    cross-validated R2 of a ridge encoding model sbp_c(t) ~ lagged finger kinematics (tuning strength)
  imp        impedance stored with the session (NaN when absent)
Outputs a long table (results/channel_health/channel_day.csv) plus a JSON summary of:
  - activity/tuning trends over time and across arrays,
  - "death" events (sustained loss of activity) and revivals, abrupt vs gradual changes,
  - spatial clustering of failure (neighbours vs distant pairs) and edge vs interior electrodes,
  - relationship between impedance and activity.

Usage: python scripts/channel_health.py [--out results/channel_health]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.data import link  # noqa: E402

LEAD = 10  # motor cortex leads movement: encode sbp(t) from kinematics t..t+180 ms


def tuning_r2(sbp, kin, n_folds=5):
    """Cross-validated R2 per channel of smoothed sbp_c(t) ~ ridge(lagged kinematics). Contiguous folds.

    SBP is smoothed with a 100 ms causal boxcar first; single 20 ms bins are too noisy for per-channel tuning."""
    T = len(kin)
    k = np.ones(5) / 5
    sbp = np.stack([np.convolve(sbp[:, c], k)[:T] for c in range(sbp.shape[1])], axis=1)
    X = np.concatenate([np.roll(kin, -l, axis=0) for l in range(LEAD)], axis=1)
    X[T - LEAD:] = 0
    X = np.c_[X, np.ones(T)]
    pred = np.zeros_like(sbp, dtype=np.float64)
    for f in np.array_split(np.arange(T), n_folds):
        m = np.ones(T, bool)
        m[f] = False
        A = X[m].T @ X[m] + 1e-2 * np.eye(X.shape[1])
        pred[f] = X[f] @ np.linalg.solve(A, X[m].T @ sbp[m])
    ss_res = ((sbp - pred) ** 2).sum(0)
    ss_tot = ((sbp - sbp.mean(0)) ** 2).sum(0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-12)


def runs_below(x, thr):
    """Boolean mask of sessions belonging to runs of >= 3 consecutive sessions below thr."""
    b = x < thr
    out = np.zeros_like(b)
    i = 0
    while i < len(b):
        if b[i]:
            j = i
            while j < len(b) and b[j]:
                j += 1
            if j - i >= 3:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/channel_health")
    ap.add_argument("--alive-hz", type=float, default=2.0, help="TC rate threshold for an 'active' channel")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    link.build_cache(verbose=False)
    layout = link.electrode_layout()
    dist = link.chebyshev_distance(layout)
    rows = []
    keys = link.list_sessions()
    for k in keys:
        s = link.load_session(k)
        r2 = tuning_r2(s.sbp.astype(np.float64), s.kin.astype(np.float64))
        tc = s.tc.mean(0) / link.BIN_S
        for c in range(s.sbp.shape[1]):
            rows.append(dict(key=k, date=s.date, day=s.day, style=s.style, ch=c, tc_rate=float(tc[c]),
                             sbp_mean=float(s.sbp[:, c].mean()), sbp_std=float(s.sbp[:, c].std()),
                             tune_r2=float(r2[c]), imp=float(s.imp[c])))
    df = pd.DataFrame(rows)
    df = df.merge(layout.reset_index().rename(columns={"index": "ch"}), on="ch")
    df.to_csv(os.path.join(args.out, "channel_day.csv"), index=False)

    # one value per (day, channel): average the CO and RD sessions of the same day
    num = ["tc_rate", "sbp_mean", "sbp_std", "tune_r2", "imp"]
    daily = df.groupby(["day", "ch"])[num].mean().reset_index()
    days = np.sort(daily.day.unique())
    piv = {m: daily.pivot(index="day", columns="ch", values=m).loc[days] for m in num}
    C = piv["tc_rate"].shape[1]
    summary = {"n_sessions": len(keys), "n_days": int(len(days)), "span_days": int(days[-1] - days[0])}

    # --- trends: yearly medians across channels
    yr = (days // 365).astype(int)
    trend = {}
    for m in ["tc_rate", "sbp_mean", "tune_r2", "imp"]:
        v = piv[m]
        trend[m] = {f"year{y}": float(np.nanmedian(v.values[yr == y])) for y in np.unique(yr)}
        rho = stats.spearmanr(days, np.nanmedian(v.values, axis=1), nan_policy="omit").correlation
        trend[m]["spearman_vs_day"] = float(rho)
    active = (piv["tc_rate"] > args.alive_hz).sum(1)
    trend["n_active"] = {f"year{y}": float(np.median(active.values[yr == y])) for y in np.unique(yr)}
    tuned = (piv["tune_r2"] > 0.05).sum(1)
    trend["n_tuned_r2>0.05"] = {f"year{y}": float(np.median(tuned.values[yr == y])) for y in np.unique(yr)}
    summary["trends"] = trend

    # --- death / revival events on the TC-rate trajectories
    tc = piv["tc_rate"].values  # (days, C)
    death_day, revived = np.full(C, np.nan), np.zeros(C, bool)
    for c in range(C):
        below = runs_below(tc[:, c], args.alive_hz)
        was_alive = tc[:5, c].mean() > args.alive_hz
        if was_alive and below.any():
            first = np.argmax(below)
            death_day[c] = days[first] - days[0]
            revived[c] = (tc[first:, c] > args.alive_hz * 2).rolling(3).min().any() if False else \
                bool((pd.Series(tc[first:, c] > 2 * args.alive_hz).rolling(3).min() == 1).any())
    init_alive = tc[:5].mean(0) > args.alive_hz
    summary["deaths"] = {"initially_active": int(init_alive.sum()), "died": int(np.isfinite(death_day).sum()),
                         "revived_after_death": int(revived.sum()),
                         "death_day_quartiles": [float(q) for q in np.nanpercentile(death_day, [25, 50, 75])]
                         if np.isfinite(death_day).any() else None}

    # abrupt vs gradual: distribution of session-to-session log changes in TC rate (heavy tails => abrupt events)
    ltc = np.log(tc + 0.1)
    dl = np.diff(ltc, axis=0)
    dt = np.diff(days)[:, None] * np.ones((1, C))
    short = dt <= 3
    v = dl[short]
    summary["tc_change_short_gaps"] = {"n": int(v.size), "sd": float(v.std()), "kurtosis_excess": float(stats.kurtosis(v)),
                                       "frac_abs_gt_1": float((np.abs(v) > 1).mean())}

    # --- spatial structure of decline: correlation of per-channel slopes (log TC vs time) with neighbour distance
    slopes = np.array([stats.linregress(days, ltc[:, c]).slope * 365 for c in range(C)])
    pairs_d, pairs_s = [], []
    for a in range(C):
        for b in range(a + 1, C):
            pairs_d.append(dist[a, b])
            pairs_s.append(abs(slopes[a] - slopes[b]))
    pairs_d, pairs_s = np.array(pairs_d), np.array(pairs_s)
    near, far = pairs_s[pairs_d <= 1], pairs_s[(pairs_d >= 3)]
    summary["spatial"] = {"slope_absdiff_neighbours_median": float(np.median(near)),
                          "slope_absdiff_far_median": float(np.median(far)),
                          "mannwhitney_p_neighbours_more_similar": float(stats.mannwhitneyu(near, far, alternative="less").pvalue)}
    # permutation test on the death indicator: are neighbours of dead channels more likely dead?
    dead = np.isfinite(death_day) & init_alive
    adj = (dist == 1)
    obs = (adj & dead[:, None] & dead[None]).sum() / max((adj & init_alive[:, None] & init_alive[None]).sum(), 1)
    rng = np.random.default_rng(0)
    null = []
    idx = np.flatnonzero(init_alive)
    for _ in range(2000):
        perm = dead.copy()
        perm[idx] = rng.permutation(dead[idx])
        null.append((adj & perm[:, None] & perm[None]).sum() / max((adj & init_alive[:, None] & init_alive[None]).sum(), 1))
    summary["spatial"]["dead_neighbour_pair_fraction"] = float(obs)
    summary["spatial"]["perm_p_clustered"] = float((np.array(null) >= obs).mean())

    # edge vs interior (physical 8x8 grid: row or col in {0, 7})
    edge = layout["row"].isin([0, 7]).values | layout["col"].isin([0, 7]).values
    summary["edge_vs_interior"] = {
        "n_edge": int(edge.sum()), "n_interior": int((~edge).sum()),
        "slope_logtc_per_year_edge_median": float(np.median(slopes[edge])),
        "slope_logtc_per_year_interior_median": float(np.median(slopes[~edge])),
        "mannwhitney_p_edge_declines_faster": float(stats.mannwhitneyu(slopes[edge], slopes[~edge], alternative="less").pvalue),
        "died_frac_edge": float(dead[edge & init_alive].mean()) if (edge & init_alive).any() else None,
        "died_frac_interior": float(dead[~edge & init_alive].mean()) if (~edge & init_alive).any() else None,
    }
    summary["by_array"] = {a: {"slope_logtc_per_year_median": float(np.median(slopes[(layout.array_name == a).values])),
                               "died": int(dead[(layout.array_name == a).values].sum())}
                           for a in layout.array_name.unique()}

    # --- impedance vs activity
    imp = piv["imp"].values
    ok = np.isfinite(imp)
    summary["impedance"] = {
        "days_with_impedance": int(np.isfinite(imp).any(1).sum()),
        "spearman_imp_vs_tc_rate_pooled": float(stats.spearmanr(imp[ok], tc[ok]).correlation),
        "spearman_logimp_vs_day_median_channel": float(stats.spearmanr(days[np.isfinite(imp).any(1)],
                                                                        np.nanmedian(imp[np.isfinite(imp).any(1)], 1)).correlation),
    }
    # within-channel: does a channel's impedance track its own activity over time?
    within = []
    for c in range(C):
        m = ok[:, c]
        if m.sum() > 20:
            within.append(stats.spearmanr(imp[m, c], tc[m, c]).correlation)
    summary["impedance"]["within_channel_spearman_median"] = float(np.nanmedian(within)) if within else None

    print(json.dumps(summary, indent=1))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
