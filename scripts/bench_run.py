"""Run reference methods (or your own) on the benchmark and write per-pair scores and a summary.

Each method is scored on every valid pair of each individual at each labelled-trial budget it can use (label-free
methods at 0; supervised methods at 10, 20, 50, 100 and 300). Rows are appended as they finish, so a stopped run can be
resumed with --resume.

Usage:
  python scripts/bench_run.py --subjects C M --methods renorm reweight          # reference methods
  python scripts/bench_run.py --subjects C --plugin mypkg.mymodule:predict --name mine --budgets 0 50
Output: results/bench/<name or method>/<subject>.csv and results/bench/summary.csv (python scripts/bench_run.py --summary)
"""
import argparse
import glob
import importlib
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci import bench  # noqa: E402
from ibci.bench import baselines as B  # noqa: E402

DATA_ROOTS = {"T8": "H:/ibci-data", "T9": "G:/ibci-data", "T7": "F:/ibci-data", "T10": "F:/ibci-data"}


def run(subject, name, fn, budgets, out_dir, resume):
    path = os.path.join(out_dir, name, f"{subject}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = pd.read_csv(path) if resume and os.path.exists(path) else pd.DataFrame()
    rows = done.to_dict("records")
    seen = {(r["train"], r["test"], r["gap_target"], r["n_labels"]) for r in rows}
    t0 = time.time()
    for c, t in enumerate(bench.tasks(subject, n_labels=budgets[0])):   # each pair loaded once, all budgets scored
        for n in budgets:
            if (t.train, t.test, t.gap, n) in seen:
                continue
            t.n_labels = n
            rows.append(t.score(fn(t), method=name))
        if (c + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"{name} {subject}: {c + 1} pairs, {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def summary(out_dir):
    frames = [pd.read_csv(f) for f in glob.glob(os.path.join(out_dir, "*", "*.csv"))]
    s = bench.summarize(pd.concat(frames).to_dict("records"))
    s.to_csv(os.path.join(out_dir, "summary.csv"), index=False, float_format="%.4f")
    print(s.pivot_table(index=["method", "n_labels"], columns="gap_target", values="retention", aggfunc="median")
          .round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=list(bench.SUBJECTS))
    ap.add_argument("--methods", nargs="+", default=list(B.LABEL_FREE) + list(B.SUPERVISED))
    ap.add_argument("--plugin", default=None, help="module:function taking a Task and returning predictions")
    ap.add_argument("--name", default=None)
    ap.add_argument("--budgets", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="results/bench")
    ap.add_argument("--data-root", default=None, help="folder holding braingate/, link_cache/, perich_cache/")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        return summary(args.out)
    if args.plugin:
        mod, fn = args.plugin.split(":")
        todo = {args.name or fn: (getattr(importlib.import_module(mod), fn), args.budgets or list(bench.BUDGETS))}
    else:
        todo = {}
        for m in args.methods:
            if m in B.LABEL_FREE:
                todo[m] = (B.LABEL_FREE[m], [0])
            else:
                todo[m] = (B.SUPERVISED[m], args.budgets or [n for n in bench.BUDGETS if n > 0])
    for k in args.subjects:
        os.environ["IBCI_DATA"] = args.data_root or DATA_ROOTS.get(k, "D:/ibci-data")
        for name, (fn, budgets) in todo.items():
            print(run(k, name, fn, budgets, args.out, args.resume), flush=True)


if __name__ == "__main__":
    main()
