"""Diagnostic (WSL): NDT3 does not fit T8 (same-day R2 ~0) while a linear decoder gets ~0.3-0.4 on the same
threshold crossings. On T8's earliest pilot session (T8_day45), one change at a time:
  P   pretrained 45M-2kh, all 192 channels (pilot setting, lr 1e-4; known to fail)
  S   same architecture trained from scratch (NDT3's own initialization), lr 4e-4
  A1  pretrained, first array only (channels 0-95)
  A2  pretrained, second array only (channels 96-191)
If S fits and P does not, our data path is fine and the pretrained model does not transfer to T8. Rows go to
results/ndt3/diag_T8.csv. Run alone (one NDT3 process at a time):
  bash scripts/ndt3/run_wsl.sh /mnt/h/ndt3/diag_t8.log scripts/ndt3/diag_t8.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ndt3_lib as L  # noqa: E402

KEY = "T8_day45"
OUT = "/mnt/g/codes/invasive bci/results/ndt3/diag_T8.csv"


def main():
    s = dict(np.load(f"/mnt/h/ibci-data/ndt3_pilot/T8/{KEY}.npz"))
    D = s["y_tr"].shape[1]
    norm, cov_max = L.write_norm("t8norm", s["y_tr"])
    rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
    done = {r["variant"] for r in rows}
    for name, init, lr, ch in (("P", L.CKPT_2KH, 1e-4, slice(0, 192)), ("S", None, 4e-4, slice(0, 192)),
                               ("A1", L.CKPT_2KH, 1e-4, slice(0, 96)), ("A2", L.CKPT_2KH, 1e-4, slice(96, 192))):
        if name in done:
            continue
        t0 = time.time()
        xtr, xte = s["x_tr"][:, ch], s["x_te"][:, ch]
        C = xtr.shape[1]
        L.cleanup("t8diag")
        L.write_split("t8diag", "train", xtr, s["y_tr"])
        cfg = L.make_cfg(L.register("t8diag", C), D, norm, lr=lr, epochs=100, patience=20)
        model, info = L.finetune(cfg, init, f"/mnt/h/ndt3/work/t8{name}")
        L.cleanup("t8eval")
        pred = L.predict_chunked(model, "t8eval", xte, s["y_te"], C, D, norm, cov_max)
        rows.append({"variant": name, "channels": f"{ch.start}-{ch.stop - 1}", "lr": lr, "epochs": info["epochs"],
                     "val": info["best_score"], "r2": L.r2(pred, s["y_te"]), "pred_std": float(pred.std(0).mean()),
                     "seconds": round(time.time() - t0)})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
