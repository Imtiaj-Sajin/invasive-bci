"""Validate the calibrated simulator on statistics it was NOT fitted to.

The calibration (calibrate_sim.py --max-day D) only matched three rungs (L2, L3, L5 at n=300) on pairs inside days < D.
Here we run the full oracle ladder (ibci.anatomy.ladder, same code as for real data) on simulated future days built
from base sessions in the *held-out period* (day >= D), and compare with the real ladder for pairs in that period:
  - held-out time (days >= D),
  - untargeted rungs (L4u label-free Procrustes, L4s stable-channel alignment, L4 latent rotation, L6p, L6),
  - untargeted labelled-data budgets (n = 20, 50, 100) from the data-efficiency ladder.
Reports per gap and rung the real and simulated medians (ratios to own-day) and their absolute differences, plus a
"no-drift" reference (simulator with all drift switched off) to show the comparison is not trivial.

Usage: python scripts/validate_sim.py --calib results/sim_calib_split/calibration.json --split-day 700
       [--ladder results/anatomy/ladder.csv] [--eff results/data_efficiency/ladder.csv] [--n-base 12]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci import sim  # noqa: E402
from ibci.anatomy import K_LAT, N_LAGS, Sess, ladder  # noqa: E402
from ibci.data import link  # noqa: E402
from ibci.linear import LagDecoder  # noqa: E402
from ibci.preprocess import ZScore  # noqa: E402


class SimSess:
    """A simulated future session with the attributes ibci.anatomy.ladder needs."""

    def __init__(self, base: Sess, d: sim.Drift, rng, dt):
        xtr = sim.to_raw(sim.apply_drift(base.ztr, d, rng), d, base.z.mean, base.z.std)
        xte = sim.to_raw(sim.apply_drift(base.zte, d, rng), d, base.z.mean, base.z.std)
        self.key, self.day, self.style = f"sim_{base.key}_{dt}", base.day + dt, base.style
        self.y_tr, self.y_te, self.trial_id = base.y_tr, base.y_te, base.trial_id
        self.z = ZScore().fit(xtr)
        self.ztr, self.zte = self.z.transform(xtr).astype(np.float32), self.z.transform(xte).astype(np.float32)
        self.dec = LagDecoder.fit(self.ztr, self.y_tr, N_LAGS, 0.1)
        self.V = np.linalg.svd(self.ztr, full_matrices=False)[2][:K_LAT].T

    @property
    def x_te(self):
        return self.zte * self.z.std + self.z.mean


def ratios(df, cols):
    return {c: float(np.median(df[c] / df["own"])) for c in cols if c in df}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="results/sim_calib_split/calibration.json")
    ap.add_argument("--split-day", type=int, default=700)
    ap.add_argument("--ladder", default="results/anatomy/ladder.csv")
    ap.add_argument("--eff", default="results/data_efficiency/ladder.csv")
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 7, 30, 120, 480])
    ap.add_argument("--n-trials", type=int, nargs="+", default=[20, 50, 100, 300])
    ap.add_argument("--n-base", type=int, default=12)
    ap.add_argument("--out", default="results/sim_validation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cal = json.load(open(args.calib))
    p = sim.SimParams(**{k: v for k, v in cal["params"].items() if k in sim.SimParams.__dataclass_fields__})
    off = sim.SimParams(**{**p.to_dict(), "s_mix0": 0, "s_mix": 0, "rho0": 0, "rho_inf": 0, "h_off": 0, "h_jump": 0})
    dist = link.chebyshev_distance(link.electrode_layout())
    keys = [k for k in link.list_sessions()
            if (np.datetime64(k[:10]) - np.datetime64("2020-01-27")).astype(int) >= args.split_day]
    rng = np.random.default_rng(1)
    base_keys = sorted(rng.choice(keys, min(args.n_base, len(keys)), replace=False))

    rows = []
    for bk in base_keys:
        b = Sess(bk)
        for g in args.gaps:
            for name, prm in (("sim", p), ("nodrift", off)):
                r = np.random.default_rng([7, hash(bk) % 10000, g])
                d = sim.sample_drift(b.ztr, g, prm, dist, r, alive=b.alive)
                sj = SimSess(b, d, r, g)
                rows.append(dict(kind=name, base=bk, gap_target=g, **ladder(b, sj, args.n_trials)))
        print("base", bk, "done", flush=True)
    simdf = pd.DataFrame(rows)
    simdf.to_csv(os.path.join(args.out, "sim_ladder.csv"), index=False)

    real = pd.read_csv(args.ladder)
    real = real[real.train_day >= args.split_day]
    eff = pd.read_csv(args.eff)
    eff = eff[eff.train_day >= args.split_day]
    n_eff = [n for n in args.n_trials if n != 300]
    target = ["L2", "L3_n300", "L5_n300"]
    untargeted = ["L4u", "L4s", "L4_n300", "L6p_n300", "L6_n300"]
    eff_cols = [f"{m}_n{n}" for n in n_eff for m in ("L3", "L5", "L6p", "L6")]
    out = []
    for g in args.gaps:
        rg, eg = real[real.gap_target == g], eff[eff.gap_target == g]
        for kind in ("sim", "nodrift"):
            sg = simdf[(simdf.kind == kind) & (simdf.gap_target == g)]
            for group, cols, ref in (("targeted", target, rg), ("untargeted", untargeted, rg), ("efficiency", eff_cols, eg)):
                if len(ref) < 3:
                    continue
                rr, ss = ratios(ref, cols), ratios(sg, cols)
                for c in cols:
                    if c in rr and c in ss:
                        out.append(dict(gap=g, kind=kind, group=group, rung=c, real=rr[c], sim=ss[c],
                                        abs_err=abs(rr[c] - ss[c]), n_real=len(ref)))
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(args.out, "validation.csv"), index=False)
    summ = res.groupby(["kind", "group"]).abs_err.agg(["median", "mean", "count"]).round(3)
    print(summ.to_string())
    print(res[res.kind == "sim"].pivot_table(index="rung", columns="gap", values=["real", "sim"]).round(2).to_string())
    json.dump({"summary": summ.reset_index().to_dict("records"), "params": p.to_dict(), "base_sessions": base_keys},
              open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
