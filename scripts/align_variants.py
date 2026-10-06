"""Label-free alignment with and without per-channel variance normalization.

In the main ladder, Procrustes alignment (L4u) is applied to renormalized features, so it cannot be compared fairly
with renormalization itself. Here, for the same pairs, we also align features whose channel means were updated with
day j's statistics but whose scales stay at day i's ('mean-updated'), with the day-j subspace estimated from those
features. Rungs: mean-updated (L1), mean-updated + alignment, renormalized (L2), renormalized + alignment (L4u).

Usage: python scripts/align_variants.py --subject N|T6|C|M [--out results/align_variants]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from decay_alpha import GAPS, LADDER, sessions_for  # noqa: E402
from ibci.anatomy import K_LAT, latent_features, pred_latent, procrustes_latent_map, r2  # noqa: E402


def aligned(si, sj, Ztr_j, Zte_j):
    Vj = np.linalg.svd(Ztr_j - Ztr_j.mean(0), full_matrices=False)[2][:K_LAT].T
    Q = procrustes_latent_map(Vj, si.V)
    G, B = latent_features(Zte_j, Vj, si.V, si.dec)
    return r2(pred_latent(G, B, (Q.reshape(-1), si.dec.b)), sj.y_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--alpha", type=float, default=None, help="base decoder alpha (default: as in the main ladder)")
    ap.add_argument("--out", default="results/align_variants")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    alpha = args.alpha if args.alpha is not None else (1.0 if args.subject in ("C", "M") else 0.1)
    src = LADDER.get(args.subject, f"results/replication_bg/{args.subject}_ladder.csv")
    pairs = pd.read_csv(src)
    pairs = pairs[pairs.gap_target.isin(GAPS)][["train", "test", "gap_target", "days"]].reset_index(drop=True)
    sess = sessions_for(args.subject, alpha, set(pairs.train) | set(pairs.test))
    rows = []
    for _, r in pairs.iterrows():
        si, sj = sess[r.train], sess[r.test]
        xtr_j = sj.ztr * sj.z.std + sj.z.mean
        m_tr = (xtr_j - sj.z.mean) / si.z.std            # mean-updated: day-j means, day-i scales
        m_te = (sj.x_te - sj.z.mean) / si.z.std
        rows.append(dict(**r.to_dict(),
                         mean_only=r2(si.dec.predict(m_te), sj.y_te),
                         mean_plus_align=aligned(si, sj, m_tr, m_te),
                         renorm=r2(si.dec.predict(sj.zte), sj.y_te),
                         renorm_plus_align=aligned(si, sj, sj.ztr, sj.zte),
                         own=r2(sj.dec.predict(sj.zte), sj.y_te)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"{args.subject}.csv"), index=False)
    print(df.groupby("gap_target")[["mean_only", "mean_plus_align", "renorm", "renorm_plus_align", "own"]]
          .median().round(3).to_string())


if __name__ == "__main__":
    main()
