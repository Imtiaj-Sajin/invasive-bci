"""Decay ladder with a different base-decoder ridge penalty, on the same session pairs as the main analysis.

Re-runs the correction ladder (supervised subspace rotation and full input remap skipped) for every pair at the five
main gaps, with day-i and day-j decoders trained with --alpha (default 1e4, the value chosen by the 1% rule in
monkeys N and C and human T6). Used to test whether the decay time course depends on the decoder's regularization.

Usage: python scripts/decay_alpha.py --subject N|C|M|T6|T5|T9 [--alpha 1e4] [--out results/decay_alpha]
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import ibci.anatomy as anatomy_mod  # noqa: E402
from ibci.anatomy import ladder  # noqa: E402

GAPS = [1, 7, 30, 120, 480]
LADDER = {"N": "results/anatomy/ladder.csv", "C": "results/replication/C_ladder.csv",
          "M": "results/replication/M_ladder.csv"}


def sessions_for(subject, alpha, keys):
    if subject == "N":
        anatomy_mod.ALPHA = alpha
        from ibci.anatomy import Sess
        return {k: Sess(k) for k in keys}
    if subject in ("C", "M"):
        from ibci.data import perich
        from replicate_perich import PerichSess
        raw = [perich.load_session(k) for k in perich.list_sessions(subject)]
        X, _ = perich.aligned_counts(raw)
        day0 = np.datetime64(raw[0].date)
        out = {}
        for s, x in zip(raw, X):
            if s.key in keys:
                out[s.key] = PerichSess(s, x, day0, alpha)
        return out
    from replicate_braingate_decoding import BGSess, load_session
    root = os.path.join(os.environ.get("IBCI_DATA", "D:/ibci-data"), "braingate", "decoding", subject)
    out = {}
    for f in glob.glob(os.path.join(root, "*_decoding.mat")):
        day = int(re.search(r"_day_(\d+)_", f).group(1))
        key = f"{subject}_day{day}"
        if key in keys:
            d, X, Y, T = load_session(f)
            out[key] = BGSess(key, d, X, Y, T, alpha)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--out", default="results/decay_alpha")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    src = LADDER.get(args.subject, f"results/replication_bg/{args.subject}_ladder.csv")
    pairs = pd.read_csv(src)
    pairs = pairs[pairs.gap_target.isin(GAPS)][["train", "test", "gap_target", "days"]].reset_index(drop=True)
    sess = sessions_for(args.subject, args.alpha, set(pairs.train) | set(pairs.test))
    rows, t0 = [], time.time()
    out_csv = os.path.join(args.out, f"{args.subject}_alpha{args.alpha:g}.csv")
    for c, r in pairs.iterrows():
        res = ladder(sess[r.train], sess[r.test], [300], skip=("L4", "L5"))
        rows.append(dict(subject=args.subject, alpha=args.alpha, **r.to_dict(), **res))
        if (c + 1) % 25 == 0 or c + 1 == len(pairs):
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"{c + 1}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).assign(ret=lambda x: x.L2 / x.own)
    print(df.groupby("gap_target")[["L1", "L2", "L4u", "L3_n300", "own", "ret"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
