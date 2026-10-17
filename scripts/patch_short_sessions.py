"""Recompute the session pairs affected by the cross-validation fold fix of 2026-10-17 (ibci.anatomy._fold_blocks).

Before the fix, a test session with at most 60 labelled training trials (n/k at the 300-trial budget) gave one fold
with no training rows; every candidate's CV error was NaN and the smallest penalty was chosen. Only two T9 test days
are affected (days 629 and 728; 13 pairs, 9 of them valid). This script recomputes exactly those rows in the files whose
rows are deterministic per pair, and leaves every other row unchanged:
  results/replication_bg/T9_ladder.csv      default penalty (alpha 0.1), ladder without L4/L5 (as in the main run)
  results/decay_alpha/T9_alpha10000.csv      tuned penalty, ladder without L4/L5
  results/latent_ceiling/T9.csv              tuned penalty, ceiling rungs at 300 trials and all trials
reweight_controls.py and gain_mechanism.py draw random permutations in pair order, so they are rerun in full instead.

Usage: IBCI_DATA=G:/ibci-data python scripts/patch_short_sessions.py [--dry-run]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from ibci.anatomy import ladder  # noqa: E402
from decay_alpha import sessions_for  # noqa: E402
from latent_ceiling import ceiling  # noqa: E402

SUBJECT = "T9"
MAX_TRIALS_BAD = 60        # 300 trials / 5 folds


def short_test_days():
    """Test sessions with at most 60 labelled training trials (trial counts recorded by latent_ceiling.py)."""
    c = pd.read_csv(f"results/latent_ceiling/{SUBJECT}.csv")
    return set(c.test[c.n_trials_available <= MAX_TRIALS_BAD])


def patch(path, alpha, compute, dry):
    df = pd.read_csv(path)
    short = short_test_days()
    idx = df.index[df.test.isin(short)]
    sess = sessions_for(SUBJECT, alpha, set(df.train[idx]) | set(df.test[idx]))
    for k in short:                                   # check the recorded trial counts
        assert int(np.max(sess[k].trial_id)) + 1 <= MAX_TRIALS_BAD, k
    print(f"{path}: {len(idx)} rows with test days {sorted(short)}", flush=True)
    for i in idx:
        r = df.loc[i]
        new = compute(sess[r.train], sess[r.test])
        for k, v in new.items():
            if k in df.columns:
                old = df.at[i, k]
                if isinstance(v, float) and not (pd.isna(old) and pd.isna(v)) and abs(float(old) - v) > 1e-12:
                    print(f"  {r.train} -> {r.test} {k}: {old:.4f} -> {v:.4f}")
                df.at[i, k] = v
    if not dry:
        df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    patch("results/replication_bg/T9_ladder.csv", 0.1,
          lambda si, sj: ladder(si, sj, [300], skip=("L4", "L5")), a.dry_run)
    patch("results/decay_alpha/T9_alpha10000.csv", 1e4,
          lambda si, sj: ladder(si, sj, [300], skip=("L4", "L5")), a.dry_run)

    def ceil_rows(si, sj):
        n_avail = int(np.max(sj.trial_id)) + 1
        res = ceiling(si, sj, sorted({300, n_avail}))
        tag = f"_n{n_avail}"
        res.update({k[:-len(tag)] + "_nall": v for k, v in list(res.items()) if k.endswith(tag)})
        return res
    patch("results/latent_ceiling/T9.csv", 1e4, ceil_rows, a.dry_run)


if __name__ == "__main__":
    main()
