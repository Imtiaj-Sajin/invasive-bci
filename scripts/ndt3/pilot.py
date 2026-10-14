"""NDT3-45M (2 kh) pilot on the benchmark pairs exported by export_pilot.py (run in the NDT3 WSL env).

Protocol (fixed 2026-10-14 before any pilot result; decision 0005):
  per training session i:  base_i = pretrained NDT3 fine-tuned on day i's training segment (all labelled trials)
  per pair (i, j):
    zeroshot      base_i applied to day j's evaluation segment (no update)
    fewshot_n50   base_i fine-tuned on the first 50 labelled day-j trials
    fewshot_n300  base_i fine-tuned on the first 300 labelled day-j trials
    unsup         base_i fine-tuned on unlabelled day-j training data with the neural (spike) loss only
  per test session j:  ref_j = pretrained NDT3 fine-tuned on day j's training segment (same-day reference)
Behaviour scale: from day i for base_i and its updates, from day j for ref_j. Supervised fine-tunes early-stop on
NDT3's validation R2 (random 1 s windows of the fine-tuning data; 20%), the unsupervised one on the validation spike
loss. Evaluation: chunked causal prediction (2 s windows, 1 s step, last 1 s kept); R2 on unclamped labels.
Rows are appended to results/ndt3/pilot/<subject>.csv as they finish; a rerun skips finished rows.

Usage: bash scripts/ndt3/run_wsl.sh LOG scripts/ndt3/pilot.py --subjects N T5 T8 [--epochs 100 --patience 20]
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ndt3_lib as L  # noqa: E402

PILOT = "/mnt/h/ibci-data/ndt3_pilot"
OUT = "/mnt/g/codes/invasive bci/results/ndt3/pilot"
WORK = "/mnt/h/ndt3/work/pilot"


def load(subject, key):
    return dict(np.load(f"{PILOT}/{subject}/{key}.npz"))


def done_rows(path):
    if not os.path.exists(path):
        return pd.DataFrame(), set()
    d = pd.read_csv(path)
    return d, set(zip(d.kind, d.train.fillna(""), d.test, d.condition))


def append(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False, float_format="%.6f")


def tune(init, s, n_trials, norm, D, C, job, args, kin_weight=1.0):
    """Fine-tune `init` on session s's training segment, restricted to its first n_trials trials (None = all)."""
    x, y = s["x_tr"], s["y_tr"]
    if n_trials is not None:
        keep = s["trial_tr"] < n_trials
        x, y = x[keep], y[keep]
    L.cleanup(job)
    L.write_split(job, "train", x, y)
    unsup = kin_weight == 0
    cfg = L.make_cfg(L.register(job, C), D, norm, lr=args.lr, epochs=args.unsup_epochs if unsup else args.epochs,
                     patience=args.patience, batch_size=args.batch_size, kin_weight=kin_weight, seed=args.seed)
    wd = f"{WORK}/{job}"
    shutil.rmtree(wd, ignore_errors=True)
    model, info = L.finetune(cfg, init, wd, monitor="val_loss" if unsup else "val_kinematic_r2",
                             mode="min" if unsup else "max")
    info["bins"] = int(len(x))
    return model, info


def evaluate(model, s, norm, cov_max, D, C, job):
    L.cleanup(job)
    pred = L.predict_chunked(model, job, s["x_te"], s["y_te"], C, D, norm, cov_max)
    L.cleanup(job)
    return L.r2(pred, s["y_te"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["N", "T5", "T8"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--unsup-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    json.dump(vars(args), open(f"{OUT}/settings.json", "w"), indent=1)
    pairs = pd.read_csv(f"{PILOT}/pairs.csv")
    for k in args.subjects:
        path = f"{OUT}/{k}.csv"
        _, done = done_rows(path)
        P = pairs[pairs.subject == k]
        # same-day references
        for j in sorted(set(P.test)):
            if ("ref", "", j, "sameday") in done:
                continue
            t0 = time.time()
            s = load(k, j)
            C, D = s["x_tr"].shape[1], s["y_tr"].shape[1]
            norm, cov_max = L.write_norm(f"norm{k}{j}".replace("-", "").replace("_", ""), s["y_tr"])
            model, info = tune(L.CKPT_2KH, s, None, norm, D, C, "ref", args)
            r = evaluate(model, s, norm, cov_max, D, C, "refeval")
            append(path, {"subject": k, "kind": "ref", "train": "", "test": j, "gap_target": 0, "condition": "sameday",
                          "r2": r, "epochs": info["epochs"], "val": info["best_score"], "bins": info["bins"],
                          "seconds": round(time.time() - t0)})
            print(k, "ref", j, round(r, 4), info, f"{time.time() - t0:.0f}s", flush=True)
        # base models and their updates
        for i, Q in P.groupby("train", sort=False):
            todo = [(r.test, c) for r in Q.itertuples() for c in ("zeroshot", "fewshot_n50", "fewshot_n300", "unsup")
                    if ("pair", i, r.test, c) not in done]
            if not todo:
                continue
            si = load(k, i)
            C, D = si["x_tr"].shape[1], si["y_tr"].shape[1]
            norm, cov_max = L.write_norm(f"norm{k}{i}".replace("-", "").replace("_", ""), si["y_tr"])
            t0 = time.time()
            base, binfo = tune(L.CKPT_2KH, si, None, norm, D, C, "base", args)
            print(k, "base", i, binfo, f"{time.time() - t0:.0f}s", flush=True)
            for r in Q.itertuples():
                sj = load(k, r.test)
                for cond in ("zeroshot", "fewshot_n50", "fewshot_n300", "unsup"):
                    if ("pair", i, r.test, cond) in done:
                        continue
                    t1 = time.time()
                    if cond == "zeroshot":
                        model, info = base, {"epochs": 0, "best_score": None, "bins": 0}
                    elif cond == "unsup":
                        model, info = tune(base, sj, None, norm, D, C, "upd", args, kin_weight=0.0)
                    else:
                        model, info = tune(base, sj, int(cond.split("_n")[1]), norm, D, C, "upd", args)
                    res = evaluate(model, sj, norm, cov_max, D, C, "updeval")
                    append(path, {"subject": k, "kind": "pair", "train": i, "test": r.test, "gap_target": r.gap_target,
                                  "condition": cond, "r2": res, "epochs": info["epochs"], "val": info["best_score"],
                                  "bins": info["bins"], "seconds": round(time.time() - t1),
                                  "base_epochs": binfo["epochs"], "base_val": binfo["best_score"]})
                    print(k, i, r.test, r.gap_target, cond, round(res, 4), f"{time.time() - t1:.0f}s", flush=True)
    print("pilot finished")


if __name__ == "__main__":
    main()
