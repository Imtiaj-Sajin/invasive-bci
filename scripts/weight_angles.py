"""Is apparent rotation of decoder weights across days mostly per-channel scaling?

Earlier work measured drift as the angle between decoders fitted on different days, invariant only to a global scale
(Wilson et al. 2025), and reported large preferred-direction changes (Pun et al. 2024). If each channel keeps its
tuning pattern but changes in strength, such angles would look large although a per-channel scale explains them.

For every pair at the five main gaps, with decoders fitted on each day (ridge, alpha = 1e4 so that weights are well
determined), we compute
  angle_global   angle between vec(W_i) and vec(W_j)                     (invariant to one global scale)
  angle_channel  angle between vec(W_j) and vec(diag(s) W_i), with the per-channel scale s_c fitted by least
                 squares (invariant to one scale per channel, including sign)
  angle_null     angle_channel for W_i with its channels randomly permuted (what per-channel scaling achieves
                 without any shared structure)
where each channel's block of W holds its weights over 8 lags and all outputs.

Usage: python scripts/weight_angles.py --subject N|C|M|T6|T5|T9 [--out results/weight_angles]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from decay_alpha import GAPS, LADDER, sessions_for  # noqa: E402


def angle(a, b):
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def channel_blocks(dec):
    W3 = dec.W3                                       # (L, C, K)
    return np.transpose(W3, (1, 0, 2)).reshape(W3.shape[1], -1)   # (C, L*K)


def scaled_angle(Bi, Bj):
    s = (Bi * Bj).sum(1) / np.maximum((Bi * Bi).sum(1), 1e-12)     # least-squares scale per channel
    return angle((s[:, None] * Bi).ravel(), Bj.ravel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--out", default="results/weight_angles")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pairs = pd.read_csv(LADDER.get(args.subject, f"results/replication_bg/{args.subject}_ladder.csv"))
    pairs = pairs[pairs.gap_target.isin(GAPS)][["train", "test", "gap_target", "days"]].reset_index(drop=True)
    sess = sessions_for(args.subject, args.alpha, set(pairs.train) | set(pairs.test))
    rng = np.random.default_rng(0)
    rows = []
    for _, r in pairs.iterrows():
        Bi, Bj = channel_blocks(sess[r.train].dec), channel_blocks(sess[r.test].dec)
        rows.append(dict(subject=args.subject, **r.to_dict(), angle_global=angle(Bi.ravel(), Bj.ravel()),
                         angle_channel=scaled_angle(Bi, Bj),
                         angle_null=scaled_angle(Bi[rng.permutation(len(Bi))], Bj)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"{args.subject}.csv"), index=False)
    print(df.groupby("gap_target")[["angle_global", "angle_channel", "angle_null"]].median().round(1).to_string())


if __name__ == "__main__":
    main()
