"""Benchmark baseline results (v1) from the paper's tuned-penalty ladders, which are the benchmark's reference methods on
exactly the benchmark pairs (scripts/bench_check.py reproduces them through the benchmark API to ~1e-16).
Label budgets 0 and 300; budgets 10-100 on all pairs are left for a later version.

Writes results/bench/<method>/<subject>.csv in the format of scripts/bench_run.py and results/bench/summary.csv.
Usage: python scripts/bench_from_paper.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci import bench  # noqa: E402
from ibci.bench import baselines as B  # noqa: E402

OUT = "results/bench"


def main():
    for k in bench.SUBJECTS:
        lad = pd.read_csv(f"results/decay_alpha/{k}_alpha10000.csv")
        m = bench.manifest(k).merge(lad, on=["train", "test", "gap_target"], how="left", validate="one_to_one",
                                    suffixes=("", "_lad"))
        assert m.own.notna().all(), f"{k}: manifest pairs missing from the ladder"
        assert (m.ref_r2 - m.own).abs().max() < 1e-5, f"{k}: manifest reference R2 differs from the ladder"
        for name, col in B.LADDER_COLUMN.items():
            n = 0 if name in B.LABEL_FREE else 300
            col = col.format(n=300)
            d = pd.DataFrame({"method": name, "subject": k, "train": m.train, "test": m.test,
                              "gap_target": m.gap_target, "days": m.days, "n_labels": n, "valid": m.valid,
                              "r2": m[col], "ref_r2": m.own, "retention": m[col] / m.own})
            os.makedirs(f"{OUT}/{name}", exist_ok=True)
            d.to_csv(f"{OUT}/{name}/{k}.csv", index=False, float_format="%.6f")
        print(k, len(m), "pairs", int(m.valid.sum()), "valid", flush=True)
    frames = [pd.read_csv(f"{OUT}/{name}/{k}.csv") for name in B.LADDER_COLUMN for k in bench.SUBJECTS]
    s = bench.summarize(pd.concat(frames).to_dict("records"))
    s.to_csv(f"{OUT}/summary.csv", index=False, float_format="%.4f")
    print(s.pivot_table(index=["method", "n_labels"], columns="gap_target", values="retention", aggfunc="median")
          .round(2).to_string())


if __name__ == "__main__":
    main()
