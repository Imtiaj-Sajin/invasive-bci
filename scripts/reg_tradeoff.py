"""Regularization for the future: how does ridge strength trade same-day accuracy against cross-day robustness?

For training sessions sampled across LINK, ridge decoders (8 lags of z-scored SBP) are trained with alpha on a wide
grid and evaluated on the same day's held-out trials and on later same-style sessions at target gaps (daily
renormalization). The LINK-paper default is alpha = 0.1.

Usage: python scripts/reg_tradeoff.py [--max-train 40] [--out results/reg_tradeoff]
       python scripts/reg_tradeoff.py --dataset perich --subject C --out results/reg_tradeoff_C
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci.anatomy import N_LAGS, SessCache, SessMeta, r2  # noqa: E402
from ibci.data import link  # noqa: E402
from ibci.linear import lagged, ridge  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from drift_anatomy import select_pairs  # noqa: E402

ALPHAS = [0.1, 10, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 7, 30, 120, 480])
    ap.add_argument("--max-train", type=int, default=40)
    ap.add_argument("--dataset", choices=["link", "perich"], default="link")
    ap.add_argument("--subject", default="C", help="000688 subject when --dataset perich")
    ap.add_argument("--out", default="results/reg_tradeoff")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.dataset == "link":
        meta = [SessMeta(k) for k in link.list_sessions()]
        cache = SessCache(maxsize=60)
    else:  # DANDI 000688: all sessions of one subject, channels aligned by electrode label
        from ibci.data import perich
        from replicate_perich import PerichSess
        raw = [perich.load_session(k) for k in perich.list_sessions(args.subject)]
        X, _ = perich.aligned_counts(raw)
        day0 = np.datetime64(raw[0].date)
        meta = [PerichSess(s, x, day0) for s, x in zip(raw, X)]
        cache = {s.key: s for s in meta}
    pairs = select_pairs(meta, args.gaps)
    rng = np.random.default_rng(3)
    cand = sorted({i for i, _, _ in pairs})
    train_idx = sorted(rng.choice(cand, min(args.max_train, len(cand)), replace=False))
    rows = []
    for i in train_idx:
        si = cache[meta[i].key]
        H = lagged(si.ztr, N_LAGS)
        decs = {a: ridge(H, si.y_tr, a) for a in ALPHAS}
        Hs = lagged(si.zte, N_LAGS)
        for a, (W, b) in decs.items():
            rows.append(dict(train=si.key, gap_target=0, alpha=a, r2=r2(Hs @ W + b, si.y_te)))
        for (ii, j, g) in pairs:
            if ii != i:
                continue
            sj = cache[meta[j].key]
            Hj = lagged(sj.zte, N_LAGS)
            for a, (W, b) in decs.items():
                rows.append(dict(train=si.key, gap_target=g, alpha=a, r2=r2(Hj @ W + b, sj.y_te)))
        print(si.key, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "tradeoff.csv"), index=False)
    tab = df.groupby(["gap_target", "alpha"]).r2.median().unstack().round(3)
    print(tab.to_string())
    tab.to_csv(os.path.join(args.out, "median_r2.csv"))


if __name__ == "__main__":
    main()
