"""Is cross-day drift spatially local on the Utah array?

For decoder pairs (train day i -> later test day j), learn an input remap x~ = (I + D*mask) z + h in front of the
frozen day-i decoder, restricting which channel pairs may mix:
  diag     only self-connections (per-channel gain)
  local1   Chebyshev grid distance <= 1 on the same array (3x3 neighbourhood)
  local2   distance <= 2 (5x5 neighbourhood)
  far1/far2  same number of off-diagonal entries per channel as local1/local2, drawn at random from channels at
             distance >= 3 or on the other array (parameter-matched control)
  full     unrestricted 96 x 96
If local masks recover most of what 'full' recovers and clearly more than parameter-matched far masks, the drift
that matters for decoding is spatially local remixing (signals shifting between neighbouring electrodes).

Usage: python scripts/remap_locality.py [--gaps 1 7 30 120 480] [--n-trials 300 50] [--out results/locality]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.anatomy import LAMS, Sess, fit_remap, holdout_split, pred_remap, r2  # noqa: E402
from ibci.data import link  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from drift_anatomy import select_pairs  # noqa: E402


def build_masks(dist, rng):
    C = len(dist)
    masks = {"diag": np.eye(C), "full": np.ones((C, C))}
    for r in (1, 2):
        local = (dist <= r).astype(float)
        masks[f"local{r}"] = local
        far = np.eye(C)
        pool_all = dist >= 3
        for c in range(C):
            k = int(local[c].sum()) - 1
            pool = np.flatnonzero(pool_all[c])
            far[c, rng.choice(pool, size=min(k, len(pool)), replace=False)] = 1
        masks[f"far{r}"] = far
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 7, 30, 120, 480])
    ap.add_argument("--n-trials", type=int, nargs="+", default=[300, 50])
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--out", default="results/locality")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    link.build_cache(verbose=False)
    masks = build_masks(link.chebyshev_distance(link.electrode_layout()), rng)
    print({k: int(v.sum()) for k, v in masks.items()}, flush=True)
    sessions = []
    for k in link.list_sessions():
        try:
            sessions.append(Sess(k))
        except ValueError:
            pass
    pairs = select_pairs(sessions, args.gaps)
    if args.max_train:
        keep = set(rng.choice(len(sessions), min(args.max_train, len(sessions)), replace=False))
        pairs = [p for p in pairs if p[0] in keep]
    print(f"{len(sessions)} sessions, {len(pairs)} pairs", flush=True)

    rows, t0 = [], time.time()
    for c, (i, j, g) in enumerate(pairs):
        si, sj = sessions[i], sessions[j]
        row = dict(train=si.key, test=sj.key, gap_target=g, days=sj.day - si.day,
                   L2=r2(si.dec.predict(sj.zte), sj.y_te), own=r2(sj.dec.predict(sj.zte), sj.y_te))
        for n in args.n_trials:
            lab, fit_m, val_m = holdout_split(sj.trial_id, n)
            Y = sj.y_tr
            for name, mk in masks.items():
                errs = [float(((pred_remap(sj.ztr[val_m], fit_remap(sj.ztr[fit_m], Y[fit_m], si.dec, lam, mask=mk))
                                - Y[val_m]) ** 2).sum()) for lam in LAMS]
                lam = LAMS[int(np.argmin(errs))]
                row[f"{name}_n{n}"] = r2(pred_remap(sj.zte, fit_remap(sj.ztr[lab], Y[lab], si.dec, lam, mask=mk)), sj.y_te)
        rows.append(row)
        if (c + 1) % 10 == 0 or c + 1 == len(pairs):
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "locality.csv"), index=False)
            print(f"{c + 1}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    cols = ["L2"] + [f"{m}_n{n}" for n in args.n_trials for m in masks] + ["own"]
    print(df.groupby("gap_target")[cols].median().round(3).T.to_string())


if __name__ == "__main__":
    main()
