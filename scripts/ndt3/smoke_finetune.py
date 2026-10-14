"""Smoke test of ndt3_lib (WSL): short fine-tune on one monkey N session, predict its own held-out segment and a
session 7 days later; checks alignment and timing.

  wsl -d Ubuntu-24.04 -u root -- bash -c "cd /opt/ndt3 && /opt/ndt3-venv/bin/python '/mnt/g/codes/invasive bci/scripts/ndt3/smoke_finetune.py'"
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/g/codes/invasive bci/scripts/ndt3")
import ndt3_lib as L  # noqa: E402

PILOT = "/mnt/h/ibci-data/ndt3_pilot/N"
src = np.load(f"{PILOT}/2020-01-30_CO.npz")
tgt = np.load(f"{PILOT}/2020-02-06_CO.npz")
C, D = src["x_tr"].shape[1], src["y_tr"].shape[1]

t0 = time.time()
L.cleanup("smktrain"); L.cleanup("smkeval0"); L.cleanup("smkeval1")
norm, cov_max = L.write_norm("smktrain", src["y_tr"])
L.write_split("smktrain", "N20200130", src["x_tr"], src["y_tr"])
cfg = L.make_cfg(L.register("smktrain", C), D, norm, epochs=int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 20, patience=50)
if "--reuse" in sys.argv:          # debug: reuse the checkpoint of the previous smoke run
    from context_general_bci.model import BrainBertInterface
    model = BrainBertInterface.load_from_checkpoint("/mnt/h/ndt3/work/smk/best.ckpt")
else:
    model, info = L.finetune(cfg, L.CKPT_2KH, "/mnt/h/ndt3/work/smk")
    print("finetune", info, f"{time.time() - t0:.0f}s", flush=True)

for job, d, label in (() if "--chunked-only" in sys.argv else (("smkeval0", src, "same day"), ("smkeval1", tgt, "7 days later"))):
    t1 = time.time()
    n = len(d["x_te"])
    L.write_split(job, "evalsplit", d["x_te"], d["y_te"], pad_to=50)
    ecfg = L.make_cfg(L.register(job, C), D, norm, chop_ms=1000)
    ynorm = np.clip(d["y_te"] / cov_max, -1, 1)
    pred = L.predict(model, ecfg, n, cov_max, y_true_norm_check=ynorm)
    print(label, "R2", round(L.r2(pred, d["y_te"]), 4), f"{time.time() - t1:.0f}s", flush=True)

for job, d, label in (("smkch0", src, "same day (chunked)"), ("smkch1", tgt, "7 days later (chunked)")):
    t1 = time.time()
    L.cleanup(job)
    pred = L.predict_chunked(model, job, d["x_te"], d["y_te"], C, D, norm, cov_max)
    print(label, "R2", round(L.r2(pred, d["y_te"]), 4), f"{time.time() - t1:.0f}s", flush=True)
