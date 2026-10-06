"""Correction ladder for a recurrent-network decoder in any subject, on the same pairs as the linear ladder.

Tests whether the main findings hold for a nonlinear decoder. For a random subset of training sessions (fixed seed),
every pair at the five main gaps from the linear ladder is evaluated with an LSTM on 20-bin causal windows of z-scored
features. The network is regularized for small closed-loop datasets: 64 hidden units, input dropout 0.5, AdamW with
weight decay 0.01, and early stopping on the last 20% of training trials (checked every 100 of at most 4,000
iterations). Without this, a 256-unit network memorized the human training sessions (training R2 0.97, test R2 -1.1).
  L2     renormalized inputs, frozen network                               (no labels)
  L3     + one gain and offset per channel in front of the frozen network  (labels)
  L6p    fine-tune all weights with an L2 penalty toward the old network   (labels)
  own    network trained on the test day                                   (reference)
Penalty strengths are chosen on the last 20% of the labelled trials, then refitted on all of them. The linear
ridge results for the same pairs are merged in for paired comparison.

Usage: python scripts/nn_ladder.py --subject T5 [--max-train 25] [--out results/nn_ladder]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from decay_alpha import GAPS, LADDER, sessions_for  # noqa: E402
from drift_anatomy_nn import DEV, batched_predict, fit_decoder, fit_front, windows  # noqa: E402
from ibci.anatomy import r2  # noqa: E402

BASE_ALPHA = {"N": 0.1, "C": 1.0, "M": 1.0}
HID, P_DROP, WD, MAX_IT = 64, 0.5, 1e-2, 4000


class RegLSTM(nn.Module):
    def __init__(self, c, k, hid=HID, pdrop=P_DROP):
        super().__init__()
        self.drop = nn.Dropout(pdrop)
        self.lstm = nn.LSTM(c, hid, batch_first=True)
        self.out = nn.Linear(hid, k)

    def forward(self, x):
        h, _ = self.lstm(self.drop(x))
        return self.out(h[:, -1])


def fit_es(X, Y, tid, seed=0):
    """Train RegLSTM with early stopping on the last 20% of training trials."""
    n = int(tid.max()) + 1
    fm = torch.tensor(tid < int(0.8 * n), device=DEV)
    Xf, Yf, Xv, Yv = X[fm], Y[fm], X[~fm], Y[~fm]
    torch.manual_seed(seed)
    m = RegLSTM(X.shape[2], Y.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=WD)
    g = torch.Generator().manual_seed(seed)
    best, state = float("inf"), None
    for it in range(MAX_IT):
        m.train()
        idx = torch.randint(0, len(Xf), (256,), generator=g)
        loss = ((m(Xf[idx]) - Yf[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 100 == 0:
            m.eval()
            v = float(((batched_predict(m, Xv) - Yv) ** 2).mean())
            if v < best:
                best, state = v, {k: x.detach().clone() for k, x in m.state_dict().items()}
    m.load_state_dict(state)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--max-train", type=int, default=25)
    ap.add_argument("--out", default="results/nn_ladder")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    src = LADDER.get(args.subject, f"results/replication_bg/{args.subject}_ladder.csv")
    lin = pd.read_csv(src)
    lin = lin[lin.gap_target.isin(GAPS)].reset_index(drop=True)
    rng = np.random.default_rng(0)
    trains = sorted(lin.train.unique())
    keep = set(rng.choice(trains, min(args.max_train, len(trains)), replace=False))
    pairs = lin[lin.train.isin(keep)].reset_index(drop=True)
    sess = sessions_for(args.subject, BASE_ALPHA.get(args.subject, 0.1), set(pairs.train) | set(pairs.test))
    print(f"{args.subject}: {len(pairs)} pairs from {len(keep)} training sessions on {DEV}", flush=True)

    t = lambda a: torch.tensor(np.asarray(a), device=DEV, dtype=torch.float32)  # noqa: E731
    cache, models, rows, t0 = {}, {}, [], time.time()

    def data(s):
        if s.key not in cache:
            cache[s.key] = (windows(t(s.ztr)), t(s.y_tr), windows(t(s.zte)), s.y_te, np.asarray(s.trial_id))
        return cache[s.key]

    out_csv = os.path.join(args.out, f"{args.subject}.csv")
    for c, r in pairs.iterrows():
        si, sj = sess[r.train], sess[r.test]
        for s in (si, sj):
            if s.key not in models:
                X_, Y_, _, _, tid_ = data(s)
                models[s.key] = fit_es(X_, Y_, tid_)
        Xtr, Ytr, Xte, yte, tid = data(sj)
        n_lab = int(tid.max()) + 1
        fit_m = torch.tensor(tid < int(0.8 * n_lab), device=DEV)
        val_m = ~fit_m
        mi = models[si.key]
        row = dict(subject=args.subject, train=r.train, test=r.test, gap_target=int(r.gap_target), days=int(r.days),
                   L2=r2(batched_predict(mi, Xte).cpu().numpy(), yte),
                   own=r2(batched_predict(models[sj.key], Xte).cpu().numpy(), yte),
                   ridge_L2=r.L2, ridge_L3=r.L3_n300, ridge_L6p=r.L6p_n300, ridge_own=r.own)
        errs = {}
        for lam in (1e-4, 1e-3, 1e-2):
            f = fit_front(mi, Xtr[fit_m], Ytr[fit_m], "gain", lam)
            errs[lam] = float(((batched_predict(mi, Xtr[val_m], f) - Ytr[val_m]) ** 2).mean())
        lam = min(errs, key=errs.get)
        row["L3"] = r2(batched_predict(mi, Xte, fit_front(mi, Xtr, Ytr, "gain", lam)).cpu().numpy(), yte)
        row["lam_L3"] = lam
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
        if (c + 1) % 10 == 0 or c + 1 == len(pairs):
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"{c + 1}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    df["ret_nn"], df["ret_ridge"] = df.L2 / df.own, df.ridge_L2 / df.ridge_own
    df["gain_nn"], df["gain_ridge"] = df.L3 / df.own, df.ridge_L3 / df.ridge_own
    print(df.groupby("gap_target")[["own", "ridge_own", "ret_nn", "ret_ridge", "gain_nn", "gain_ridge"]]
          .median().round(3).to_string())


if __name__ == "__main__":
    main()
