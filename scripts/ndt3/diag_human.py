"""Diagnostic (WSL): why is NDT3's same-day R2 below the linear decoder on T5 when it is twice the linear decoder on
monkey N? One session (T5 day 1519, linear same-day R2 0.489 on threshold crossings), one change at a time:
  A  pilot settings (lr 1e-4, patience 20, max 100 epochs; 2 s windows / keep last 1 s at evaluation)
  B  lr 4e-4 (NDT3's fine-tuning default)
  C  patience 50, max 150 epochs
  D  evaluation with shorter context (1 s windows, keep last 0.5 s), model A
Rows go to results/ndt3/diag_T5.csv.

  bash scripts/ndt3/run_wsl.sh /mnt/h/ndt3/diag.log scripts/ndt3/diag_human.py [A] [B] [C]
Run alone: two NDT3 processes exceed the WSL VM's 6 GB cap (one was OOM-killed on 2026-10-15).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ndt3_lib as L  # noqa: E402

SESSION = ("T5", "T5_day1519")
OUT = "/mnt/g/codes/invasive bci/results/ndt3/diag_T5.csv"


def main():
    k, key = SESSION
    s = dict(np.load(f"/mnt/h/ibci-data/ndt3_pilot/{k}/{key}.npz"))
    C, D = s["x_tr"].shape[1], s["y_tr"].shape[1]
    norm, cov_max = L.write_norm("diagnorm", s["y_tr"])
    want = sys.argv[1:] or ["A", "B", "C"]
    rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
    for name, lr, epochs, patience in (("A", 1e-4, 100, 20), ("B", 4e-4, 100, 20), ("C", 1e-4, 150, 50)):
        if name not in want:
            continue
        t0 = time.time()
        L.cleanup("diag")
        L.write_split("diag", "train", s["x_tr"], s["y_tr"])
        cfg = L.make_cfg(L.register("diag", C), D, norm, lr=lr, epochs=epochs, patience=patience)
        model, info = L.finetune(cfg, L.CKPT_2KH, f"/mnt/h/ndt3/work/diag{name}")
        L.cleanup("diageval")
        pred = L.predict_chunked(model, "diageval", s["x_te"], s["y_te"], C, D, norm, cov_max)
        r = L.r2(pred, s["y_te"])
        rows.append({"variant": name, "lr": lr, "epochs_max": epochs, "patience": patience, "epochs": info["epochs"],
                     "val": info["best_score"], "r2": r, "seconds": round(time.time() - t0)})
        print(rows[-1], flush=True)
        if name == "A":
            L.cleanup("diageval")
            pred = L.predict_chunked(model, "diageval", s["x_te"], s["y_te"], C, D, norm, cov_max, win=50, keep=25)
            rows.append({"variant": "D", "lr": lr, "epochs_max": epochs, "patience": patience, "epochs": info["epochs"],
                         "val": info["best_score"], "r2": L.r2(pred, s["y_te"]), "seconds": 0})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
