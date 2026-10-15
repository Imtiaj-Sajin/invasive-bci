"""Check whether the cross-validation fix (anchor candidate in cv_shrunk, 2026-10-13) changes the re-weighting rung of
monkey N's default-penalty ladder (results/anatomy/ladder.csv), which calibrates the simulator. Recomputes L3_n300 for
every pair with the current code and compares with the stored value. L2 and L5 targets do not use cv_shrunk.

Usage: python scripts/tools/check_default_ladder_l3.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from ibci import anatomy  # noqa: E402
from ibci.anatomy import LAMS_CF, SessCache, cv_shrunk, gain_features, r2  # noqa: E402


def main():
    lad = pd.read_csv("results/anatomy/ladder.csv")
    assert anatomy.ALPHA == 0.1, "default penalty expected"
    cache = SessCache(maxsize=60)
    diffs, chose_anchor = [], 0
    for c, r in enumerate(lad.itertuples()):
        si, sj = cache[r.train], cache[r.test]
        C = sj.ztr.shape[1]
        lam, f = cv_shrunk(gain_features(sj.ztr, si.dec), sj.y_tr, sj.trial_id, 300, LAMS_CF, np.ones(C), 5,
                           intercept_anchor=si.dec.b)
        diffs.append(r2(f(gain_features(sj.zte, si.dec)), sj.y_te) - r.L3_n300)
        chose_anchor += int(np.isinf(lam))
        if (c + 1) % 100 == 0:
            print(f"{c + 1}/{len(lad)} pairs; max |diff| so far {np.max(np.abs(diffs)):.2e}", flush=True)
    d = np.abs(np.array(diffs))
    print(f"{len(d)} pairs: max |L3 new - stored| = {d.max():.2e}; pairs changed (>1e-9): {(d > 1e-9).sum()}; "
          f"anchor chosen: {chose_anchor}")


if __name__ == "__main__":
    main()
