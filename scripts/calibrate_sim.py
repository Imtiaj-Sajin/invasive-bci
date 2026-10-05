"""Calibrate the drift simulator so its oracle ladder matches the real one on LINK.

Targets (from scripts/drift_anatomy.py output): for each gap, the median over real pairs of
    L2/own  (renormalized fixed decoder),  L3/own  (+ per-channel gains),  L5/own  (+ full input remap)
Simulation: for base sessions b, a decoder trained on real day b is evaluated on a simulated day b+gap obtained by
applying one sampled drift realization to day b's own training and test segments; the same three rungs and the
simulated own-day decoder are computed, using fixed regularization (the median selected on real data).
Parameters fitted (Nelder-Mead in transformed space, common random numbers): s_mix0, s_mix, tau_mix, rho0,
rho_inf, tau_rho (instant + slow components of mixing and turnover).
Alive/silent switching rates come from scripts/failure_stats_xdata.py (--failure-json), not from the ladder.

Usage: python scripts/calibrate_sim.py --ladder results/anatomy/ladder.csv [--n-base 16] [--out results/sim_calib]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci import sim  # noqa: E402
from ibci.anatomy import Sess, fit_gain, fit_remap, gain_features, pred_gain, pred_remap, r2  # noqa: E402
from ibci.data import link  # noqa: E402
from ibci.linear import LagDecoder  # noqa: E402
from ibci.preprocess import ZScore  # noqa: E402

RUNGS = ["L2", "L3_n300", "L5_n300"]


def real_targets(ladder_csv, gaps, max_day=None):
    df = pd.read_csv(ladder_csv)
    if max_day is not None:  # both sessions of a pair must lie in the calibration period
        df = df[df.train_day + df.days < max_day]
    out = {}
    for g in gaps:
        d = df[df.gap_target == g]
        out[g] = {r: float(np.median(d[r] / d["own"])) for r in RUNGS}
        out[g]["n"] = int(len(d))
    return out


def sim_ladder(sessions, base_idx, gaps, p: sim.SimParams, dist, seed=0, lam_gain=1e-2, lam_remap=1e-1, iters=30):
    res = {g: {r: [] for r in RUNGS} for g in gaps}
    for b in base_idx:
        s = sessions[b]
        for g in gaps:
            rng = np.random.default_rng([seed, b, g])
            d = sim.sample_drift(s.ztr, g, p, dist, rng)
            xtr, xte = sim.apply_drift(s.ztr, d, rng), sim.apply_drift(s.zte, d, rng)
            z = ZScore().fit(xtr)                                   # renormalize on the simulated day (unlabelled)
            ztr, zte = z.transform(xtr), z.transform(xte)
            own = r2(LagDecoder.fit(ztr, s.y_tr).predict(zte), s.y_te)
            l2 = r2(s.dec.predict(zte), s.y_te)
            l3 = r2(pred_gain(gain_features(zte, s.dec), fit_gain(gain_features(ztr, s.dec), s.y_tr, lam_gain)), s.y_te)
            l5 = r2(pred_remap(zte, fit_remap(ztr, s.y_tr, s.dec, lam_remap, iters=iters)), s.y_te)
            for r, v in zip(RUNGS, (l2, l3, l5)):
                res[g][r].append(v / own)
    return {g: {r: float(np.median(v)) for r, v in res[g].items()} for g in gaps}


def unpack(x, base: sim.SimParams):
    p = sim.SimParams(**base.to_dict())
    p.s_mix0 = float(np.exp(x[0]))
    p.s_mix = float(np.exp(x[1]))
    p.tau_mix = float(np.exp(x[2]))
    p.rho0 = float(1 / (1 + np.exp(-x[3])))
    p.rho_inf = float(np.clip(p.rho0 + (1 - p.rho0) / (1 + np.exp(-x[4])), 0, 0.999))
    p.tau_rho = float(np.exp(x[5]))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="results/anatomy/ladder.csv")
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 7, 30, 120, 480])
    ap.add_argument("--n-base", type=int, default=16)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--failure-json", default=None, help="failure_stats.json with fitted h_off / h_on for the subject")
    ap.add_argument("--subject", default="N (LINK)")
    ap.add_argument("--max-day", type=int, default=None, help="use only sessions before this day (time-split calibration)")
    ap.add_argument("--out", default="results/sim_calib")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    targets = real_targets(args.ladder, args.gaps, args.max_day)
    gaps = [g for g in args.gaps if targets[g]["n"] >= 3]
    print("targets", json.dumps(targets, indent=1), flush=True)

    link.build_cache(verbose=False)
    dist = link.chebyshev_distance(link.electrode_layout())
    keys = link.list_sessions()
    if args.max_day is not None:
        keys = [k for k in keys if (np.datetime64(k[:10]) - np.datetime64("2020-01-27")).astype(int) < args.max_day]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(keys), size=min(args.n_base, len(keys)), replace=False))
    sessions = {i: Sess(keys[i]) for i in pick}

    base = sim.SimParams()
    if args.failure_json:
        fs = json.load(open(args.failure_json))[args.subject]
        base.h_off, base.h_on = fs["h_off"], fs["h_on"]
    history = []
    t0 = time.time()

    def loss(x):
        p = unpack(x, base)
        sl = sim_ladder(sessions, pick, gaps, p, dist)
        err = sum((sl[g][r] - targets[g][r]) ** 2 for g in gaps for r in RUNGS)
        fitted_keys = ("s_mix0", "s_mix", "tau_mix", "rho0", "rho_inf", "tau_rho")
        history.append(dict(loss=err, **{k: v for k, v in p.to_dict().items() if k in fitted_keys}))
        print(f"[{len(history)}] {time.time() - t0:.0f}s loss={err:.4f} "
              + " ".join(f"{k}={getattr(p, k):.3g}" for k in fitted_keys), flush=True)
        return err

    x0 = np.array([np.log(0.3), np.log(0.5), np.log(30.0), -2.0, -1.0, np.log(200.0)])
    opt = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": args.maxiter, "xatol": 0.05, "fatol": 1e-4})
    best = unpack(opt.x, base)
    fitted = sim_ladder(sessions, pick, gaps, best, dist, seed=1)   # fresh noise for the reported fit
    out = {"params": best.to_dict(), "loss": float(opt.fun), "targets": targets, "sim_fresh_seed": fitted,
           "base_sessions": [keys[i] for i in pick]}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(args.out, "calibration.json"), "w"), indent=1)
    pd.DataFrame(history).to_csv(os.path.join(args.out, "history.csv"), index=False)


if __name__ == "__main__":
    main()
