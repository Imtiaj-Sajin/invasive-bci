"""Example: simulate chronic drift on a real session and measure how a decoder decays.

Takes one LINK session as the base, samples simulated recordings 1-480 days later with the calibrated simulator, and
reports the R2 of a ridge decoder trained on the base day (with daily renormalization) on each simulated future day.

Run from the repo root:  python examples/simulate_drift.py [--calib results/sim_calib_split/calibration.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci import sim  # noqa: E402
from ibci.anatomy import Sess, r2  # noqa: E402
from ibci.data import link  # noqa: E402
from ibci.preprocess import ZScore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="results/sim_calib_split/calibration.json")
    ap.add_argument("--session", default=None, help="LINK session key, e.g. 2020-03-10_CO (default: first session)")
    ap.add_argument("--n-real", type=int, default=5, help="random drift realizations per gap")
    args = ap.parse_args()

    params = json.load(open(args.calib))["params"]
    p = sim.SimParams(**{k: v for k, v in params.items() if k in sim.SimParams.__dataclass_fields__})
    dist = link.chebyshev_distance(link.electrode_layout())        # electrode grid distances (for local mixing)
    base = Sess(args.session or link.list_sessions()[0])           # z-scored SBP, kinematics, trained ridge decoder
    print(f"base session {base.key}: own-day R2 = {r2(base.dec.predict(base.zte), base.y_te):.3f}")

    for dt in (1, 7, 30, 120, 480):
        vals = []
        for seed in range(args.n_real):
            rng = np.random.default_rng(seed)
            drift = sim.sample_drift(base.ztr, dt, p, dist, rng, alive=base.alive)   # one realization of change
            future = sim.apply_drift(base.zte, drift, rng)                          # simulated held-out segment
            z = ZScore().fit(sim.apply_drift(base.ztr, drift, rng))                 # renormalize on the new day
            vals.append(r2(base.dec.predict(z.transform(future)), base.y_te))
        print(f"  {dt:4d} days later: decoder R2 = {np.mean(vals):+.3f} (sd {np.std(vals):.3f} over {args.n_real} draws)")


if __name__ == "__main__":
    main()
