"""Train on simulated futures: does simulator-based augmentation make decoders last longer on real future days?

Strict time split on LINK: the simulator is calibrated on sessions before --split-day (see calibrate_sim.py with
--max-day), and everything below is evaluated only on decoders trained on sessions after --split-day, tested on
later real sessions with daily unsupervised renormalization (deployment setting).

Decoders (ridge on 8 lags of z-scored SBP), all trained on day i's first 300 trials:
  base        alpha = 0.1 (LINK-paper setting)
  ridge_reg   alpha chosen to maximise cross-day R2 on the calibration period (regularization-only control)
  pert        ad hoc perturbation augmentation (Sussillo 2016 style): random channel dropout + per-channel gain noise
  sim         augmentation with K simulated futures, dt ~ log-uniform[1, dt_max], from the calibrated simulator
  sim_reg     sim with the calibration-period alpha
Reference (uses more labelled data): multi   day i plus the previous --n-prev sessions of the same style.

Usage: python scripts/sim_augment.py --calib results/sim_calib/calibration.json [--split-day 700] [--out results/augment]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci import sim  # noqa: E402
from ibci.anatomy import Sess, r2  # noqa: E402
from ibci.data import link  # noqa: E402
from ibci.linear import LagDecoder, lagged, ridge  # noqa: E402
from ibci.preprocess import ZScore  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from drift_anatomy import select_pairs  # noqa: E402

N_LAGS = 8


def aug_ridge(Zs, Ys, alpha):
    """Ridge on a list of (Z, Y) blocks, lagging each block separately (no lag leakage across blocks)."""
    H = np.concatenate([lagged(Z, N_LAGS) for Z in Zs])
    W, b = ridge(H, np.concatenate(Ys), alpha)
    return LagDecoder(W, b, N_LAGS)


def renorm(X):
    return ZScore().fit(X).transform(X)


def pert_copies(Z, k, rng, p_drop=0.1, gain_sd=0.2):
    out = []
    for _ in range(k):
        keep = rng.random(Z.shape[1]) > p_drop
        gain = np.exp(rng.normal(0, gain_sd, Z.shape[1]))
        out.append(renorm(Z * keep * gain))
    return out


def sim_copies(Z, k, p, dist, rng, dt_max):
    out = []
    for _ in range(k):
        dt = float(np.exp(rng.uniform(0, np.log(dt_max))))
        d = sim.sample_drift(Z, dt, p, dist, rng)
        out.append(renorm(sim.apply_drift(Z, d, rng)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="results/sim_calib/calibration.json")
    ap.add_argument("--split-day", type=int, default=700)
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 7, 30, 120, 240])
    ap.add_argument("--k-copies", type=int, default=8)
    ap.add_argument("--dt-max", type=float, default=120.0)
    ap.add_argument("--n-prev", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=30)
    ap.add_argument("--out", default="results/augment")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    calib = json.load(open(args.calib))
    p = sim.SimParams(**{k: v for k, v in calib["params"].items() if k in sim.SimParams.__dataclass_fields__})
    dist = link.chebyshev_distance(link.electrode_layout())
    sessions = []
    for k in link.list_sessions():
        try:
            sessions.append(Sess(k))
        except ValueError:
            pass

    # regularization-only control: alpha that maximises cross-day R2 within the calibration period
    cal = [s for s in sessions if s.day < args.split_day]
    cal_pairs = [q for q in select_pairs(cal, args.gaps)][:: max(1, len(select_pairs(cal, args.gaps)) // 60)]
    alphas = [0.1, 1, 10, 100, 1000, 1e4]
    cal_score = {a: np.mean([r2(LagDecoder.fit(cal[i].ztr, cal[i].y_tr, N_LAGS, a).predict(cal[j].zte), cal[j].y_te)
                             for i, j, _ in cal_pairs]) for a in alphas}
    a_reg = max(cal_score, key=cal_score.get)
    print("calibration-period alpha scores", {k: round(v, 3) for k, v in cal_score.items()}, "-> alpha", a_reg, flush=True)

    ev = [s for s in sessions if s.day >= args.split_day]
    pairs = select_pairs(ev, args.gaps)
    train_idx = sorted({i for i, _, _ in pairs})
    if args.max_train and len(train_idx) > args.max_train:
        train_idx = sorted(rng.choice(train_idx, args.max_train, replace=False))
    rows, t0 = [], time.time()
    for i in train_idx:
        si = ev[i]
        prev = [s for s in sessions if s.style == si.style and s.day < si.day][-args.n_prev:]
        decs = {
            "base": si.dec,
            "ridge_reg": LagDecoder.fit(si.ztr, si.y_tr, N_LAGS, a_reg),
            "pert": aug_ridge([si.ztr] + pert_copies(si.ztr, args.k_copies, rng), [si.y_tr] * (args.k_copies + 1), 0.1),
        }
        sims = sim_copies(si.ztr, args.k_copies, p, dist, rng, args.dt_max)
        decs["sim"] = aug_ridge([si.ztr] + sims, [si.y_tr] * (args.k_copies + 1), 0.1)
        decs["sim_reg"] = aug_ridge([si.ztr] + sims, [si.y_tr] * (args.k_copies + 1), a_reg)
        if prev:
            decs["multi"] = aug_ridge([si.ztr] + [s.ztr for s in prev], [si.y_tr] + [s.y_tr for s in prev], 0.1)
        for (ii, j, g) in pairs:
            if ii != i:
                continue
            sj = ev[j]
            row = dict(train=si.key, test=sj.key, gap_target=g, days=sj.day - si.day,
                       own=r2(sj.dec.predict(sj.zte), sj.y_te))
            for name, d in decs.items():
                row[name] = r2(d.predict(sj.zte), sj.y_te)
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "augment.csv"), index=False)
        print(f"train {si.key} done ({len(rows)} rows, {time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    cols = [c for c in ["base", "ridge_reg", "pert", "sim", "sim_reg", "multi", "own"] if c in df]
    summ = df.groupby("gap_target")[cols].median().round(3)
    summ["n"] = df.groupby("gap_target").size()
    print(summ.to_string())
    summ.to_csv(os.path.join(args.out, "augment_summary.csv"))
    # paired improvement over base, per gap
    imp = {c: df.groupby("gap_target").apply(lambda d: float(np.median(d[c] - d["base"]))).round(4).to_dict()
           for c in cols if c not in ("base", "own")}
    json.dump({"alpha_reg": a_reg, "cal_alpha_scores": cal_score, "median_paired_gain_vs_base": imp},
              open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(imp, indent=1))


if __name__ == "__main__":
    main()
