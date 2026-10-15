"""Pre-specified array rule for NDT3 inputs (fixed 2026-10-15, before any T8 pilot result with it):
an array is left out of NDT3's input if fewer than 25% of its channels are active (mean threshold crossings
> 0.01 per 20 ms bin) on the training segment of the individual's earliest pilot session. The linear reference keeps
all channels. Why: T8's first array is nearly silent (17/96 active; linear R2 0.02 on it alone) and NDT3 failed to fit
T8 whenever that array was included (results/ndt3/diag_T8.csv).

Writes results/ndt3/channel_rule.json: {subject: [first_channel, last_channel_exclusive] ranges kept}.
Usage: python scripts/ndt3/channel_rule.py
"""
import json
import os

import numpy as np
import pandas as pd

PILOT = os.environ.get("NDT3_PILOT", "H:/ibci-data/ndt3_pilot")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ndt3", "channel_rule.json")
MIN_ACTIVE, RATE = 0.25, 0.01


def session_time(key):
    if "_day" in key:
        return int(key.split("_day")[1])
    return int((np.datetime64(key[:10]) - np.datetime64("2000-01-01")).astype(int))


def main():
    pairs = pd.read_csv(f"{PILOT}/pairs.csv")
    rule = {}
    for k in sorted(pairs.subject.unique()):
        P = pairs[pairs.subject == k]
        key = min(set(P.train) | set(P.test), key=session_time)
        x = np.load(f"{PILOT}/{k}/{key}.npz")["x_tr"]
        kept = []
        for a in range(0, x.shape[1], 96):
            frac = float((x[:, a:a + 96].mean(0) > RATE).mean())
            print(f"{k} {key} array {a // 96 + 1} (channels {a}-{a + 95}): {frac:.0%} active", end="")
            if frac >= MIN_ACTIVE:
                kept.append([a, a + 96])
                print(" -> kept")
            else:
                print(" -> left out")
        rule[k] = kept
    json.dump(rule, open(OUT, "w"), indent=1)
    print("written", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
