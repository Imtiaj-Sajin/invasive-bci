"""Individuals in the decoder analyses and the pre-specified inclusion rule for BrainGate participants.

Rule (fixed when the first added participant failed, before the results of T2, T7, T8, T10 and T11 were seen; see
docs/INCLUSION_RULES.md): a human
participant enters the decay and correction summaries only if the median R2 of its same-day reference decoder across
the ladder's session pairs is above MIN_REF_R2. Retention (old decoder R2 / reference R2) is undefined when the
reference decoder does not work. Excluded participants are reported in the Supplement with their numbers.
"""
import os

import pandas as pd

MONKEYS = ["N", "C", "M"]
HUMAN_CANDIDATES = ["T6", "T5", "T9", "T7", "T10", "T8", "T11", "T2", "T3"]   # original three first, then new ones
MIN_REF_R2 = 0.1

LADDER = {"N": "results/anatomy/ladder.csv", "C": "results/replication/C_ladder.csv",
          "M": "results/replication/M_ladder.csv"}


def ladder_path(k):
    return LADDER.get(k, f"results/replication_bg/{k}_ladder.csv")


def reference_r2(k):
    """Median same-day reference R2 across the ladder's pairs (None if the ladder does not exist yet)."""
    p = ladder_path(k)
    if not os.path.exists(p):
        return None
    return float(pd.read_csv(p)["own"].median())


def humans():
    """Included human participants, in fixed order."""
    return [k for k in HUMAN_CANDIDATES if (r := reference_r2(k)) is not None and r > MIN_REF_R2]


def excluded_humans():
    """{participant: median reference R2} for participants with a ladder that fail the rule."""
    out = {}
    for k in HUMAN_CANDIDATES:
        r = reference_r2(k)
        if r is not None and r <= MIN_REF_R2:
            out[k] = r
    return out


ALL_PAIRS = os.environ.get("IBCI_ALL_PAIRS") == "1"   # sensitivity analysis: keep every pair


def valid_pairs(d, *own_cols):
    """Session pairs whose reference decoder(s) work (R2 > MIN_REF_R2); see docs/INCLUSION_RULES.md.
    With IBCI_ALL_PAIRS=1 every pair is kept (sensitivity analysis)."""
    if ALL_PAIRS:
        return d
    keep = pd.Series(True, index=d.index)
    for c in own_cols or ("own",):
        keep &= d[c] > MIN_REF_R2
    return d[keep]


def filter_valid(d, k):
    """Pair rule for result tables without their own reference R2 (e.g. weight angles, gain-electrode link): keep the
    (train, test, gap) pairs whose tuned-penalty reference decoder works. With IBCI_ALL_PAIRS=1 every pair is kept."""
    if ALL_PAIRS:
        return d
    ref = pd.read_csv(f"results/decay_alpha/{k}_alpha10000.csv")
    ok = ref[ref.own > MIN_REF_R2][["train", "test", "gap_target"]]
    return d.merge(ok, on=["train", "test", "gap_target"], how="inner")


def complete_budgets(d, cols):
    """Rows that have a value for every labelled-trial budget in ``cols`` (same pairs at every budget)."""
    return d.dropna(subset=[c for c in cols if c in d.columns])


def individuals():
    return MONKEYS + humans()


def name(k):
    return f"Monkey {k}" if k in MONKEYS else f"Human {k}"
