"""Oracle ladder for a nonlinear (LSTM) decoder: does the drift anatomy found with ridge hold for networks?

Decoder (close to the LINK paper's): LSTM, hidden 256, 20-bin input window (400 ms), predicts the 4 kinematics at
the window's last bin, trained on day i's first 300 trials (z-scored SBP, day-own statistics) with Adam.
Rungs on later day j (labels from day j's first 300 trials; evaluation on day j's held-out trials):
  L2     renormalized inputs, frozen network                          (no labels)
  L3     + per-channel gain and offset in front of the frozen network (192 params)
  L5     + full input remap (I + D) z + h in front of the frozen network
  L6p    fine-tune all weights with an L2 penalty toward the day-i network
  own    network trained on day j
Penalty strengths are chosen on the last 20% of the labelled trials, then the rung is refit on all of them.

Usage: python scripts/drift_anatomy_nn.py [--gaps 1 7 30 120 480] [--max-train 25] [--out results/anatomy_nn]
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.anatomy import Sess, r2  # noqa: E402
from ibci.data import link  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from drift_anatomy import select_pairs  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ, HID = 20, 256


class LSTMDec(nn.Module):
    def __init__(self, c_in=96, k_out=4, hid=HID):
        super().__init__()
        self.lstm = nn.LSTM(c_in, hid, batch_first=True)
        self.out = nn.Linear(hid, k_out)

    def forward(self, x):                      # x (B, SEQ, C)
        h, _ = self.lstm(x)
        return self.out(h[:, -1])


def windows(Z):
    """(T, C) -> (T, SEQ, C) causal windows (zero-padded at the start)."""
    Zp = torch.cat([torch.zeros(SEQ - 1, Z.shape[1], device=Z.device), Z])
    return Zp.unfold(0, SEQ, 1).permute(0, 2, 1)


def batched_predict(model, X, front=None, bs=4096):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs]
            out.append(model(front(xb) if front else xb))
    return torch.cat(out)


def train(params, loss_fn, n_iter, lr):
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter)
    for _ in range(n_iter):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        sched.step()


def fit_decoder(X, Y, n_iter=1500, seed=0, init=None, prior_lam=0.0):
    torch.manual_seed(seed)
    model = copy.deepcopy(init) if init is not None else LSTMDec(X.shape[2], Y.shape[1]).to(DEV)
    prior = [p.detach().clone() for p in init.parameters()] if (init is not None and prior_lam > 0) else None
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.train()  # cuDNN LSTM backward needs train mode (no dropout here, so outputs are unchanged)
    for p in model.parameters():
        p.requires_grad_(True)

    def loss_fn():
        idx = torch.randint(0, len(X), (256,), generator=g)
        l = ((model(X[idx]) - Y[idx]) ** 2).mean()
        if prior is not None:
            l = l + prior_lam * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), prior))
        return l

    train(model.parameters(), loss_fn, n_iter, 1e-3 if init is None else 3e-4)
    return model.eval()


def fit_front(model, X, Y, kind, lam, n_iter=400, seed=0):
    """Learn an input transform in front of a frozen network. kind: 'gain' (diag) or 'remap' (full)."""
    C = X.shape[2]
    for p in model.parameters():
        p.requires_grad_(False)
    model.train()  # cuDNN LSTM backward needs train mode (no dropout, outputs unchanged)
    torch.manual_seed(seed)
    D = torch.zeros(C, C, device=DEV, requires_grad=True) if kind == "remap" else torch.zeros(C, device=DEV, requires_grad=True)
    h = torch.zeros(C, device=DEV, requires_grad=True)

    def front(x):
        return (x @ D.T + x if kind == "remap" else x * (1 + D)) + h

    g = torch.Generator(device="cpu").manual_seed(seed)

    def loss_fn():
        idx = torch.randint(0, len(X), (512,), generator=g)
        return ((model(front(X[idx])) - Y[idx]) ** 2).mean() + lam * ((D ** 2).sum() + (h ** 2).sum())

    train([D, h], loss_fn, n_iter, 1e-2)
    model.eval()
    D, h = D.detach(), h.detach()
    return lambda x: (x @ D.T + x if kind == "remap" else x * (1 + D)) + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 7, 30, 120, 480])
    ap.add_argument("--max-train", type=int, default=25)
    ap.add_argument("--out", default="results/anatomy_nn")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    sessions = []
    for k in link.list_sessions():
        try:
            sessions.append(Sess(k))
        except ValueError:
            pass
    pairs = select_pairs(sessions, args.gaps)
    train_idx = sorted({i for i, _, _ in pairs})
    keep = set(rng.choice(train_idx, min(args.max_train, len(train_idx)), replace=False))
    pairs = [p for p in pairs if p[0] in keep]
    print(f"{len(sessions)} sessions, {len(pairs)} pairs on {DEV}", flush=True)

    t = lambda a: torch.tensor(a, device=DEV, dtype=torch.float32)  # noqa: E731
    cache = {}

    def data(s):
        if s.key not in cache:
            cache[s.key] = (windows(t(s.ztr)), t(s.y_tr), windows(t(s.zte)), s.y_te,
                            torch.tensor(s.trial_id, device=DEV))
        return cache[s.key]

    models, rows, t0 = {}, [], time.time()
    for c, (i, j, g) in enumerate(pairs):
        si, sj = sessions[i], sessions[j]
        for s in (si, sj):
            if s.key not in models:
                Xtr, Ytr, *_ = data(s)
                models[s.key] = fit_decoder(Xtr, Ytr)
        Xtr, Ytr, Xte, yte, tid = data(sj)
        mi = models[si.key]
        fit_m, val_m = tid < 240, tid >= 240
        row = dict(train=si.key, test=sj.key, gap_target=g, days=sj.day - si.day,
                   L2=r2(batched_predict(mi, Xte).cpu().numpy(), yte),
                   own=r2(batched_predict(models[sj.key], Xte).cpu().numpy(), yte))
        for kind, name in (("gain", "L3"), ("remap", "L5")):
            errs = {}
            for lam in (1e-4, 1e-3, 1e-2):
                f = fit_front(mi, Xtr[fit_m], Ytr[fit_m], kind, lam)
                errs[lam] = float(((batched_predict(mi, Xtr[val_m], f) - Ytr[val_m]) ** 2).mean())
            lam = min(errs, key=errs.get)
            f = fit_front(mi, Xtr, Ytr, kind, lam)
            row[name] = r2(batched_predict(mi, Xte, f).cpu().numpy(), yte)
            row[f"lam_{name}"] = lam
        errs = {}
        for lam in (1e-4, 1e-2, 1.0):
            m = fit_decoder(Xtr[fit_m], Ytr[fit_m], n_iter=400, init=mi, prior_lam=lam)
            errs[lam] = float(((batched_predict(m, Xtr[val_m]) - Ytr[val_m]) ** 2).mean())
        lam = min(errs, key=errs.get)
        row["L6p"] = r2(batched_predict(fit_decoder(Xtr, Ytr, n_iter=400, init=mi, prior_lam=lam), Xte).cpu().numpy(), yte)
        row["lam_L6p"] = lam
        for p in mi.parameters():
            p.requires_grad_(True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "ladder_nn.csv"), index=False)
        if (c + 1) % 5 == 0 or c + 1 == len(pairs):
            print(f"{c + 1}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby("gap_target")[["L2", "L3", "L5", "L6p", "own"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
