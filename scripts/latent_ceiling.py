"""Ceiling on latent-subspace alignment: the best rotation of the dominant neural subspace, fitted with labels.

For every pair of the tuned-decoder analysis (results/decay_alpha, alpha = 1e4, five main gaps), the correction ladder
is re-run with the supervised subspace rotation (L4) included. L4 fits the 16 x 16 map Q of the day-j subspace onto
the day-i subspace with labelled day-j trials, shrunk toward the label-free Procrustes map (L4u). Every
rotation-only alignment method is limited by L4 (given enough labels), so

    L4u (label-free rotation)  <=  L4 (best rotation)  <=  L6 (retrain)

separates loss that a better alignment could recover (L4 - L4u) from loss no rotation can recover (own - L4).
Budgets: 300 labelled trials (as in the main ladder) and all labelled trials of the day-j training segment ('nall'),
the second so that the ceiling is not limited by the number of labels. L5 (free input remap) is skipped: it can express
any decoder of the same form and is retraining in disguise (research log, 2026-10-06).

Per pair: L2, L4u, L4s and the reference decoder ('own'), plus L3 and L4 at both budgets, with the code of
ibci.anatomy.ladder (L6 and L6p are not recomputed; the reference decoder is the retrained ceiling). L2, L4u, L3_n300
and own repeat results/decay_alpha exactly and serve as a check.

Usage: python scripts/latent_ceiling.py --subject N|C|M|T6|T5|T9|T7|T10|T8 [--alpha 1e4] [--out results/latent_ceiling]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from ibci.anatomy import (LAMS_CF, cv_shrunk, gain_features, latent_features, pred_latent,  # noqa: E402
                          procrustes_latent_map, r2, stable_procrustes_map)
from decay_alpha import GAPS, LADDER, sessions_for  # noqa: E402


def ceiling(si, sj, n_list, k_folds=5):
    """The L2, L4u, L4s, L3, L4 and own rungs of ibci.anatomy.ladder, line for line, for the budgets in n_list."""
    out = {"L2": r2(si.dec.predict(sj.zte), sj.y_te)}
    Q0 = procrustes_latent_map(sj.V, si.V)
    G_te, B_te = latent_features(sj.zte, sj.V, si.V, si.dec)
    out["L4u"] = r2(pred_latent(G_te, B_te, (Q0.reshape(-1), si.dec.b)), sj.y_te)
    Qs, _ = stable_procrustes_map(sj.V, si.V)
    out["L4s"] = r2(pred_latent(G_te, B_te, (Qs.reshape(-1), si.dec.b)), sj.y_te)
    F_tr, F_te = gain_features(sj.ztr, si.dec), gain_features(sj.zte, si.dec)
    G_tr, B_tr = latent_features(sj.ztr, sj.V, si.V, si.dec)
    Y, tid, C = sj.y_tr, sj.trial_id, sj.ztr.shape[1]
    for n in n_list:
        lam, f = cv_shrunk(F_tr, Y, tid, n, LAMS_CF, np.ones(C), k_folds, intercept_anchor=si.dec.b)
        out[f"L3_n{n}"], out[f"lam_L3_n{n}"] = r2(f(F_te), sj.y_te), float(lam)
        lam, f = cv_shrunk(G_tr, Y - B_tr, tid, n, LAMS_CF, Q0.reshape(-1), k_folds, intercept_anchor=si.dec.b)
        out[f"L4_n{n}"], out[f"lam_L4_n{n}"] = r2(B_te + f(G_te), sj.y_te), float(lam)
    out["own"] = r2(sj.dec.predict(sj.zte), sj.y_te)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--out", default="results/latent_ceiling")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    src = LADDER.get(args.subject, f"results/replication_bg/{args.subject}_ladder.csv")
    pairs = pd.read_csv(src)
    pairs = pairs[pairs.gap_target.isin(GAPS)][["train", "test", "gap_target", "days"]].reset_index(drop=True)
    out_csv = os.path.join(args.out, f"{args.subject}.csv")
    rows, done = [], set()
    if os.path.exists(out_csv):                                   # resume after an interruption
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = set(zip(prev.train, prev.test, prev.gap_target))
        print(f"resuming: {len(done)} of {len(pairs)} pairs done", flush=True)
    todo = pairs[[(a, b, g) not in done for a, b, g in zip(pairs.train, pairs.test, pairs.gap_target)]]
    sess = sessions_for(args.subject, args.alpha, set(todo.train) | set(todo.test)) if len(todo) else {}
    t0 = time.time()
    for c, (_, r) in enumerate(todo.iterrows()):
        si, sj = sess[r.train], sess[r.test]
        n_avail = int(np.max(sj.trial_id)) + 1
        res = ceiling(si, sj, sorted({300, n_avail}))
        tag = f"_n{n_avail}"
        res.update({k[:-len(tag)] + "_nall": v for k, v in list(res.items()) if k.endswith(tag)})
        if n_avail != 300:
            res = {k: v for k, v in res.items() if not k.endswith(tag)}
        rows.append(dict(subject=args.subject, alpha=args.alpha, **r.to_dict(), n_trials_available=n_avail, **res))
        if (c + 1) % 10 == 0 or c + 1 == len(todo):
            pd.DataFrame(rows).to_csv(out_csv + ".tmp", index=False)
            os.replace(out_csv + ".tmp", out_csv)                 # atomic: a power cut cannot leave a broken file
            print(f"{len(rows)}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df = df[df.own > 0.1]
    cols = ["L2", "L4u", "L4s", "L3_n300", "L4_n300", "L3_nall", "L4_nall", "own"]
    print((df.groupby("gap_target")[cols].median().div(df.groupby("gap_target").own.median(), axis=0))
          .round(2).drop(columns="own").to_string())


if __name__ == "__main__":
    main()
