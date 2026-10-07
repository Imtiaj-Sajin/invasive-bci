"""Same-day decoding check: spike-band power vs -4.5 threshold crossings, session-wise vs block-wise z-scoring.

Used on 2026-10-07 to test whether the failure of T2/T3 decoders comes from our preprocessing (it does not).
Usage: python scripts/tools/check_feature_normalization.py
"""
import glob, sys, numpy as np
from scipy.io import loadmat
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from ibci.linear import LagDecoder
from ibci.metrics import r2_per_dof

def build(f, feat="sbp", blocknorm=False):
    d = loadmat(f, simplify_cells=True)
    X = np.asarray(d["neural"][feat], np.float64)
    if blocknorm:
        b = d["blocks"]; s0 = np.atleast_1d(b["sample_start_index"]).astype(int); s1 = np.atleast_1d(b["sample_end_index"]).astype(int)
        for a, e in zip(s0, s1):
            seg = X[a:e]; X[a:e] = (seg - seg.mean(0)) / (seg.std(0) + 1e-6)
    cur, tgt, tr = d["cursor_position"], d["target_position"], d["trials"]
    st, en = np.atleast_1d(tr["trial_start_index"]).astype(int), np.atleast_1d(tr["trial_end_index"]).astype(int)
    xs, ys, ts = [], [], []
    for k, (a, e) in enumerate(zip(st, en)):
        e = a + 2 * ((e - a) // 2)
        if e - a < 20: continue
        x = X[a:e].reshape(-1, 2, X.shape[1]).mean(1); v = (tgt[a:e] - cur[a:e]).reshape(-1, 2, 2).mean(1)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        xs.append(x); ys.append(np.where(n > 1e-6, v / np.maximum(n, 1e-6), 0)); ts.append(np.full(len(x), k))
    X, Y, T = np.concatenate(xs), np.concatenate(ys), np.unique(np.concatenate(ts), return_inverse=True)[1]
    tr = T < int(round(0.8 * (T.max() + 1)))
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    out = {}
    for a in (0.1, 1e4):
        dec = LagDecoder.fit(Z[tr], Y[tr], 8, a)
        out[a] = float(r2_per_dof(dec.predict(Z[~tr]), Y[~tr]).mean())
    return out

for P, root in [("T2", "F:"), ("T7", "F:"), ("T6", "D:")]:
    fs = sorted(glob.glob(f"{root}/ibci-data/braingate/decoding/{P}/*.mat"))
    for f in fs[:: max(1, len(fs) // 4)][:4]:
        r = {k: build(f, *k) for k in [("sbp", False), ("sbp", True), ("tx_4_5", False), ("tx_4_5", True)]}
        print(P, f.split("_day_")[1].split("_")[0], " ".join(f"{k[0]}{'+blk' if k[1] else ''}:{v[1e4]:.2f}" for k, v in r.items()))
