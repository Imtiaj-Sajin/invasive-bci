"""Drift anatomy on LINK: attribute cross-day decoder loss to kinds of change with an "oracle ladder".

For a ridge decoder trained on session i (8 lags of z-scored spike-band power) and tested on held-out trials of a
later session j (same target style), we evaluate increasingly powerful corrections of day-j inputs, keeping the
decoder frozen unless stated:

  L0 fixed      day-i z-scoring reused                                   (no adaptation)
  L1 recentre   day-j channel means, day-i scales                        (unsupervised)
  L2 renorm     day-j means and scales                                   (unsupervised; deployment baseline)
  L3 gain       + per-channel gain g (96) and offsets, fitted with labels
  L4 latent     day-j top-k PCs mapped by Q (k x k) onto day-i PCs; Q fitted with labels (L4u = Procrustes, no labels)
  L5 remap      full linear input map M (96 x 96) + offsets, fitted with labels
  L6 retrain    new decoder trained on day j                             (oracle reference)
  L6p prior     ridge on day j shrunk toward the day-i decoder           (practical recalibration)

Supervised rungs use day j's first n labelled trials (n in --n-trials); normalization always uses day j's first 300
trials without labels. Regularization strengths are chosen by k-fold CV over contiguous blocks of the labelled
trials (5 folds; 3 for the L-BFGS remap), logged as lam_<rung>_n<n>, then the rung is refit on all labelled trials.

Usage: python scripts/drift_anatomy.py [--gaps 1 2 4 7 14 30 60 120 240 480 900] [--n-trials 300] [--out results/anatomy]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.anatomy import SessCache, SessMeta, ladder  # noqa: E402
from ibci.data import link  # noqa: E402


def select_pairs(sessions, gaps, tol=0.25):
    """For each train session and each target gap, the same-style later session whose gap is closest (within tol)."""
    pairs = []
    for i, si in enumerate(sessions):
        cands = [(j, sj.day - si.day) for j, sj in enumerate(sessions) if sj.style == si.style and sj.day > si.day]
        for g in gaps:
            if not cands:
                break
            j, d = min(cands, key=lambda c: abs(c[1] - g))
            if abs(d - g) <= max(1, tol * g):
                pairs.append((i, j, g))
    return sorted(set(pairs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 2, 4, 7, 14, 30, 60, 120, 240, 480, 900])
    ap.add_argument("--n-trials", type=int, nargs="+", default=[300])
    ap.add_argument("--max-train", type=int, default=0, help="subsample train sessions (0 = all)")
    ap.add_argument("--pairs-from", default=None, help="CSV with train/test/gap_target columns: evaluate exactly these pairs")
    ap.add_argument("--resume", action="store_true", help="skip pairs already in <out>/ladder.csv and append")
    ap.add_argument("--out", default="results/anatomy")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    link.build_cache(verbose=False)
    keys = link.list_sessions()
    sessions = [SessMeta(k) for k in keys]
    cache = SessCache()
    pairs = select_pairs(sessions, args.gaps)
    if args.pairs_from:
        idx = {s.key: i for i, s in enumerate(sessions)}
        ref = pd.read_csv(args.pairs_from)
        pairs = [(idx[a], idx[b], int(g)) for a, b, g in zip(ref.train, ref.test, ref.gap_target)]
    elif args.max_train:
        rng = np.random.default_rng(0)
        keep = set(rng.choice(len(sessions), min(args.max_train, len(sessions)), replace=False))
        pairs = [p for p in pairs if p[0] in keep]
    out_csv = os.path.join(args.out, "ladder.csv")
    rows = []
    if args.resume and os.path.exists(out_csv):
        rows = pd.read_csv(out_csv).to_dict("records")
        done = {(r["train"], r["test"], int(r["gap_target"])) for r in rows}
        pairs = [p for p in pairs if (sessions[p[0]].key, sessions[p[1]].key, p[2]) not in done]
        print(f"resume: {len(rows)} pairs already done", flush=True)
    print(f"{len(sessions)} sessions, {len(pairs)} pairs to run, n_trials={args.n_trials}", flush=True)

    t0 = time.time()
    for c, (i, j, g) in enumerate(pairs):
        si, sj = cache[sessions[i].key], cache[sessions[j].key]
        res = ladder(si, sj, args.n_trials)
        rows.append(dict(train=si.key, test=sj.key, style=si.style, gap_target=g, days=sj.day - si.day,
                         train_day=si.day, **res))
        if (c + 1) % 25 == 0 or c + 1 == len(pairs):
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"{c + 1}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    n = max(args.n_trials)
    cols = ["L0", "L1", "L2", "L4u", f"L3_n{n}", f"L4_n{n}", f"L5_n{n}", f"L6p_n{n}", f"L6_n{n}", "own"]
    summary = df.groupby("gap_target")[cols].median().round(3)
    summary["n_pairs"] = df.groupby("gap_target").size()
    print(summary.to_string())
    summary.to_csv(os.path.join(args.out, "ladder_summary.csv"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"n_sessions": len(sessions), "n_pairs": len(pairs), "args": vars(args)}, f, indent=1)


if __name__ == "__main__":
    main()
