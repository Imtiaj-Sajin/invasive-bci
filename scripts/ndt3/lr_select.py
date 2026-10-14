"""Pre-specified learning-rate choice for the NDT3 pilot (rule fixed 2026-10-15, before any result with the chosen rates):
for each pilot individual, fine-tune the pretrained 45M-2kh model on the training segment of its earliest pilot session
with lr 1e-4 and 4e-4 (NDT3's tuning grid without its smallest value; 1e-4 was the pilot default, 4e-4 NDT3's config
default) and keep the rate with the higher NDT3 validation R2 (random 1 s windows of the training data). Test R2 on the
session's held-out segment is recorded for transparency only and is not used. Why: on T5 the pilot's same-day R2 was
below the linear decoder, and a diagnostic (scripts/ndt3/diag_human.py) showed the learning rate matters there.

Run alone (two NDT3 processes exceed the WSL VM's 6 GB cap):
  bash scripts/ndt3/run_wsl.sh /mnt/h/ndt3/lr_select.log scripts/ndt3/lr_select.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ndt3_lib as L  # noqa: E402

PILOT = "/mnt/h/ibci-data/ndt3_pilot"
OUT = "/mnt/g/codes/invasive bci/results/ndt3/lr_select.csv"
RATES = (1e-4, 4e-4)


def session_time(key):
    if "_day" in key:
        return int(key.split("_day")[1])
    return int((np.datetime64(key[:10]) - np.datetime64("2000-01-01")).astype(int))


def main():
    pairs = pd.read_csv(f"{PILOT}/pairs.csv")
    rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
    done = {(r["subject"], r["lr"]) for r in rows}
    for k in ["N", "T5", "T8"]:
        P = pairs[pairs.subject == k]
        key = min(set(P.train) | set(P.test), key=session_time)
        s = dict(np.load(f"{PILOT}/{k}/{key}.npz"))
        C, D = s["x_tr"].shape[1], s["y_tr"].shape[1]
        norm, cov_max = L.write_norm(f"lrnorm{k}", s["y_tr"])
        for lr in RATES:
            if (k, lr) in done:
                continue
            t0 = time.time()
            L.cleanup("lrsel")
            L.write_split("lrsel", "train", s["x_tr"], s["y_tr"])
            cfg = L.make_cfg(L.register("lrsel", C), D, norm, lr=lr, epochs=100, patience=20)
            model, info = L.finetune(cfg, L.CKPT_2KH, f"/mnt/h/ndt3/work/lrsel{k}{lr:g}")
            L.cleanup("lrseleval")
            pred = L.predict_chunked(model, "lrseleval", s["x_te"], s["y_te"], C, D, norm, cov_max)
            rows.append({"subject": k, "session": key, "lr": lr, "epochs": info["epochs"], "val": info["best_score"],
                         "test_r2": L.r2(pred, s["y_te"]), "seconds": round(time.time() - t0)})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.6f")
    d = pd.DataFrame(rows)
    chosen = d.loc[d.groupby("subject").val.idxmax(), ["subject", "lr", "val"]]
    print("chosen by validation R2:\n" + chosen.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
