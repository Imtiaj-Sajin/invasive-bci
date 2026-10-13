"""Check that the benchmark API reproduces the paper's ladder on the first pairs of each individual.

Compares each reference method with its ladder column in results/decay_alpha/<subject>_alpha10000.csv, and a
'public' renormalized decoder built only from the public Task arrays (src_x, src_y, cal_x, eval_x) with the
renormalized method.

Usage: python scripts/bench_check.py [--subjects C M T6] [--pairs 5] [--n-labels 300] [--label-free-only]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci import bench  # noqa: E402
from ibci.bench import baselines as B  # noqa: E402
from ibci.anatomy import N_LAGS  # noqa: E402
from ibci.linear import LagDecoder  # noqa: E402


def public_renorm(t):
    """Renormalized day-i decoder using only public arrays."""
    mu_i, sd_i = t.src_x.mean(0), t.src_x.std(0)
    sd_i = np.where(sd_i > 0, sd_i, 1.0)
    dec = LagDecoder.fit(((t.src_x - mu_i) / sd_i).astype(np.float32), t.src_y, N_LAGS, bench.REF_ALPHA)
    mu_j, sd_j = t.cal_x.mean(0), t.cal_x.std(0)
    sd_j = np.where(sd_j > 0, sd_j, 1.0)
    return dec.predict(((t.eval_x - mu_j) / sd_j).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["C", "M", "T6"])
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--n-labels", type=int, default=300)
    ap.add_argument("--label-free-only", action="store_true")
    ap.add_argument("--ref-dir", default="results/decay_alpha")
    args = ap.parse_args()
    worst = 0.0
    for k in args.subjects:
        ref = pd.read_csv(os.path.join(args.ref_dir, f"{k}_alpha10000.csv")).set_index(["train", "test", "gap_target"])
        methods = dict(B.LABEL_FREE)
        if not args.label_free_only:
            methods.update(B.SUPERVISED)
        for t in bench.tasks(k, n_labels=args.n_labels, valid_only=False, limit=args.pairs):
            row = ref.loc[(t.train, t.test, t.gap)]
            out = {"own": abs(t.reference_r2() - row.own)}
            for name, f in methods.items():
                col = B.LADDER_COLUMN[name].format(n=args.n_labels)
                out[name] = abs(t.score(f(t))["r2"] - row[col])
            out["public_renorm"] = abs(t.score(public_renorm(t))["r2"] - row.L2)
            worst = max(worst, max(v for n, v in out.items() if n != "public_renorm"))
            print(k, t.train, t.test, t.gap, " ".join(f"{n}={v:.1e}" for n, v in out.items()), flush=True)
    print(f"largest difference from the paper (reference methods): {worst:.2e}")


if __name__ == "__main__":
    main()
