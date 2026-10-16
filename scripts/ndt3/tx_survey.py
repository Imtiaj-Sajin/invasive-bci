"""Same-day decoding with threshold crossings versus spike-band power, to choose NDT3 pilot individuals.

For 8 sessions spread over each individual's valid benchmark pairs: R2 of the same-day ridge decoder (8 lags, penalty
1e4, the paper's tuned decoder) on spike-band power and on threshold crossings (bench features='tx').
Writes results/ndt3/tx_survey.csv. Usage: python scripts/ndt3/tx_survey.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from ibci import bench  # noqa: E402

ROOTS = {"T8": "H:/ibci-data", "T9": "G:/ibci-data", "T7": "F:/ibci-data", "T10": "F:/ibci-data"}


def same_day_r2(s):
    p, y = s.dec.predict(s.zte), s.y_te
    return float(np.mean(1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)))


def main():
    rows = []
    for k in ["N", "T6", "T5", "T9", "T7", "T10", "T8"]:
        os.environ["IBCI_DATA"] = ROOTS.get(k, "D:/ibci-data")
        m = bench.manifest(k)
        m = m[m.valid]
        idx = np.linspace(0, len(m) - 1, 6).astype(int)
        keys = sorted(set(m.iloc[idx].train) | set(m.iloc[idx].test))[:8]
        sb, tx = bench._sessions(k, set(keys), "sbp"), bench._sessions(k, set(keys), "tx")
        for key in keys:
            rows.append({"subject": k, "session": key, "r2_sbp": same_day_r2(sb[key]), "r2_tx": same_day_r2(tx[key])})
        q = pd.DataFrame(rows)
        q = q[q.subject == k]
        print(f"{k}: same-day R2 median SBP {q.r2_sbp.median():.3f}, TX {q.r2_tx.median():.3f} ({len(q)} sessions)",
              flush=True)
    pd.DataFrame(rows).to_csv("results/ndt3/tx_survey.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
