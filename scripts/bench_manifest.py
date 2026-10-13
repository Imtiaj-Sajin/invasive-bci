"""Write the fixed benchmark splits (benchmark/manifests/<subject>.csv) from the paper's session pairs.

Pairs: those of the paper's ladders at the five main gaps (one per training session and gap; see select_pairs in
scripts/drift_anatomy.py). ref_r2: R2 of the same-day reference decoder at the tuned penalty (results/decay_alpha);
valid: ref_r2 > 0.1 (docs/decisions/0003).

Usage: python scripts/bench_manifest.py [--ref-dir results/decay_alpha] [--out benchmark/manifests]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.bench import GAPS, MIN_REF_R2, SUBJECTS  # noqa: E402
from ibci.participants import ladder_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="results/decay_alpha")
    ap.add_argument("--out", default="benchmark/manifests")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for k in SUBJECTS:
        pairs = pd.read_csv(ladder_path(k))
        pairs = pairs[pairs.gap_target.isin(GAPS)][["train", "test", "gap_target", "days"]].reset_index(drop=True)
        ref = pd.read_csv(os.path.join(args.ref_dir, f"{k}_alpha10000.csv"))[["train", "test", "gap_target", "own"]]
        m = pairs.merge(ref, on=["train", "test", "gap_target"], how="left", validate="one_to_one")
        assert m.own.notna().all(), f"{k}: {m.own.isna().sum()} pairs without a reference R2"
        m = m.rename(columns={"own": "ref_r2"}).assign(subject=k, valid=lambda x: x.ref_r2 > MIN_REF_R2)
        m = m[["subject", "train", "test", "gap_target", "days", "ref_r2", "valid"]]
        m.to_csv(os.path.join(args.out, f"{k}.csv"), index=False, float_format="%.6f")
        print(f"{k}: {len(m)} pairs, {int(m.valid.sum())} valid, {m.train.nunique()} training sessions")


if __name__ == "__main__":
    main()
