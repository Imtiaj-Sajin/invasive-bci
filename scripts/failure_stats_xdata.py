"""Electrode failure statistics on activity alone, comparable across datasets and species.

Uses each dataset's channel-level spike counts per 20 ms bin:
  link   LINK threshold crossings (monkey N, 96 ch, 3.5 years)
  perich DANDI 000688 spikes summed per electrode (monkeys C and M; sorted units, so 'activity' = isolated units)
  h2     FALCON H2 threshold crossings (human T5, 192 ch, ~17 months)
Per session and channel we compute the event rate (Hz). Sessions with session-wide rate inflation (median channel rate
> 3x the median over sessions; threshold/preprocessing changes) are dropped first. Statistics (same for every subject):
  - active channels per session (rate > --alive-hz) and its trend per year,
  - deaths: initially active channels (first 3 sessions) that later stay below threshold for >= 3 consecutive
    sessions; revivals: dead channels that later exceed 2x threshold for >= 3 consecutive sessions,
  - alive<->silent switching rates (two-state Markov chain fitted to channels active in >= 3 sessions),
  - death rate per channel-year (Kaplan-Meier-free crude rate: deaths / channel-years at risk),
  - abruptness: excess kurtosis and tail share of log-rate changes between sessions <= 7 days apart, both raw and
    after removing the session-wide (common-mode, median across channels) change, plus the count of global events.

Usage: python scripts/failure_stats_xdata.py [--datasets link perich h2] [--out results/failure_xdata]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from channel_health import runs_below  # noqa: E402


MAX_DAY = None  # set from --max-day (time-split fitting, LINK only)


def rates_link():
    from ibci.data import link
    rows = []
    for k in link.list_sessions():
        s = link.load_session(k)
        if MAX_DAY is not None and s.day >= MAX_DAY:
            continue
        rows.append((s.day, s.tc.mean(0) / link.BIN_S))
    return {"N (LINK)": rows}


def rates_perich():
    from ibci.data import perich
    out = {}
    for subj, name in (("C", "C (000688)"), ("M", "M (000688)")):
        keys = perich.list_sessions(subj)
        if not keys:
            continue
        sess = [perich.load_session(k) for k in keys]
        X, labels = perich.aligned_counts(sess)
        d0 = np.datetime64(sess[0].date)
        out[name] = [(int((np.datetime64(s.date) - d0).astype(int)), x.mean(0) / perich.BIN_S) for s, x in zip(sess, X)]
    return out


def rates_h2():
    from ibci.data import falcon_h2
    # full-length held-in sessions only: the held-out calibration files are ~80 s snippets with inflated rates
    files = [f for f in falcon_h2.list_files("calib") if "held-in" in os.path.basename(f)]
    rows, d0 = [], None
    for f in files:
        date, tc, _ = falcon_h2.load_tc(f)
        d0 = d0 or np.datetime64(date)
        rows.append((int((np.datetime64(date) - d0).astype(int)), tc.mean(0) / falcon_h2.BIN_S))
    return {"T5 (FALCON H2)": rows} if rows else {}


def fit_switching(days, alive, max_gap=400):
    """Fit alive<->silent rates (per day) of a two-state Markov chain to all within-channel session pairs.

    alive: (sessions, C) bool. Uses pairs (t1 < t2, t2 - t1 <= max_gap) and maximises the likelihood of the observed
    state at t2 given the state at t1 (closed-form transition probabilities), over a log-spaced grid."""
    from scipy.optimize import minimize
    i, j = np.triu_indices(len(days), 1)
    dt = days[j] - days[i]
    m = dt <= max_gap
    i, j, dt = i[m], j[m], dt[m]
    a0, a1 = alive[i], alive[j]                       # (pairs, C)

    def nll(x):
        off, on = np.exp(x)
        tot = off + on
        e = np.exp(-tot * dt)[:, None]
        p_s_given_a = off / tot * (1 - e)             # alive -> silent
        p_a_given_s = on / tot * (1 - e)              # silent -> alive
        p = np.where(a0, np.where(a1, 1 - p_s_given_a, p_s_given_a), np.where(a1, p_a_given_s, 1 - p_a_given_s))
        return -np.log(np.clip(p, 1e-9, 1)).sum()

    res = minimize(nll, np.log([1e-3, 1e-3]), method="Nelder-Mead")
    return [float(v) for v in np.exp(res.x)]


def drop_inflated(rows, factor=3.0):
    """Remove sessions with session-wide rate inflation (median channel rate > factor x the median over sessions),
    e.g. threshold/preprocessing changes; returns (kept rows, number dropped)."""
    med = np.array([np.median(r[1]) for r in rows])
    ok = med <= factor * np.median(med)
    return [r for r, k in zip(rows, ok) if k], int((~ok).sum())


def summarize(rows, alive_hz):
    rows, n_dropped = drop_inflated(rows)
    rows = sorted(rows, key=lambda r: r[0])
    days = np.array([r[0] for r in rows], dtype=float)
    # merge same-day sessions
    ud = np.unique(days)
    R = np.stack([np.mean([r[1] for r in rows if r[0] == d], axis=0) for d in ud])   # (sessions, C)
    days = ud
    C = R.shape[1]
    active = (R > alive_hz).sum(1)
    yrs = days / 365.25
    slope = stats.linregress(yrs, active).slope if len(days) > 2 else np.nan
    init = R[:3].mean(0) > alive_hz
    death_t = np.full(C, np.nan)
    revived = np.zeros(C, bool)
    for c in np.flatnonzero(init):
        below = runs_below(R[:, c], alive_hz)
        if below.any():
            i0 = int(np.argmax(below))
            death_t[c] = days[i0]
            rev = pd.Series(R[i0:, c] > 2 * alive_hz).rolling(3).min() == 1
            revived[c] = bool(rev.any())
    dead = np.isfinite(death_t)
    at_risk_years = np.where(dead, death_t, days[-1])[init].sum() / 365.25
    lr = np.log(R + 0.1)
    dl = np.diff(lr, axis=0)[np.diff(days) <= 7]
    v = dl[:, init].ravel() if dl.size else np.array([])
    # session-wide (common-mode) events: median log change across channels; channel-level = deviation from it
    cm = np.median(dl, axis=1, keepdims=True) if dl.size else np.zeros((0, 1))
    v_ch = (dl - cm)[:, init].ravel() if dl.size else np.array([])
    global_events = int((np.abs(cm.ravel()) > 1).sum())
    ever = (R > alive_hz).sum(0) >= 3                 # channels that were active in >= 3 sessions
    h_off, h_on = fit_switching(days, (R > alive_hz)[:, ever])
    return {
        "sessions_dropped_rate_inflation": n_dropped,
        "h_off": h_off, "h_on": h_on, "silent_fraction_stationary": h_off / (h_off + h_on),
        "n_ever_active": int(ever.sum()),
        "n_sessions": int(len(days)), "span_days": int(days[-1] - days[0]), "n_channels": int(C),
        "active_first3_median": float(np.median(active[:3])), "active_last3_median": float(np.median(active[-3:])),
        "active_trend_per_year": float(slope),
        "initially_active": int(init.sum()), "died": int(dead.sum()), "revived": int(revived.sum()),
        "deaths_per_channel_year": float(dead.sum() / max(at_risk_years, 1e-9)),
        "median_death_day": float(np.nanmedian(death_t)) if dead.any() else None,
        "lograte_change_kurtosis": float(stats.kurtosis(v)) if v.size > 10 else None,
        "frac_abs_logchange_gt1": float((np.abs(v) > 1).mean()) if v.size else None,
        "channel_level_kurtosis": float(stats.kurtosis(v_ch)) if v_ch.size > 10 else None,
        "channel_level_frac_gt1": float((np.abs(v_ch) > 1).mean()) if v_ch.size else None,
        "global_events_abs_median_change_gt1": global_events,
        "n_short_gap_session_pairs": int(dl.shape[0]),
        "n_changes": int(v.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["link", "perich", "h2"])
    ap.add_argument("--alive-hz", type=float, default=2.0)
    ap.add_argument("--max-day", type=int, default=None, help="LINK only: use sessions before this day")
    ap.add_argument("--out", default="results/failure_xdata")
    args = ap.parse_args()
    global MAX_DAY
    MAX_DAY = args.max_day
    os.makedirs(args.out, exist_ok=True)
    loaders = {"link": rates_link, "perich": rates_perich, "h2": rates_h2}
    res = {}
    for d in args.datasets:
        try:
            for name, rows in loaders[d]().items():
                res[name] = summarize(rows, args.alive_hz)
        except Exception as e:  # dataset not downloaded yet
            print(f"skip {d}: {e}")
    df = pd.DataFrame(res).T
    print(df.to_string())
    df.to_csv(os.path.join(args.out, "failure_stats.csv"))
    json.dump(res, open(os.path.join(args.out, "failure_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
