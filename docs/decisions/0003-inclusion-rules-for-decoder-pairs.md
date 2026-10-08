# 0003 — Inclusion rules for participants and session pairs

Date: 2026-10-08. Status: adopted.

## Context

Retention is the R² of an old (corrected) decoder divided by the R² of a reference decoder trained on the test day.
It is undefined when the reference decoder does not work. Adding six BrainGate participants exposed two such cases:

- whole participants whose decoders never work (T3: median reference R² −0.05 over 12 pairs; T2: −0.02 over 120
  pairs; checks with block-wise normalization and threshold crossings did not help, see RESEARCH_LOG 2026-10-07);
- single broken test sessions inside otherwise good participants (e.g. T7 day 268: reference R² −42.8), which made
  bootstrap intervals explode (T7 one-day retention CI 0.19–54.8).

## Decision

1. **Participants** (fixed 2026-10-07, before the T7, T8, T10 and T11 results were seen): a human participant enters
   the decay and correction summaries only if the median reference R² across its ladder pairs is above 0.1.
   Included: T6, T5, T9, T7, T10, T8 (T11 pending). Excluded: T2, T3, reported in the Supplement.
2. **Pairs** (adopted 2026-10-08, after seeing the T7 artefact): within included individuals, session pairs whose
   reference R² is at most 0.1 are left out of retention summaries. The same rule applies to all nine individuals,
   monkeys included. Implemented in `ibci.participants.valid_pairs`.

## Effect (checked before adopting)

Pairs dropped per individual: N 0%, C 10.3%, M 2.3%, T6 15.7%, T5 3.5%, T9 11.9%, T7 17.9%, T10 15.2%, T8 7.7%.
Median retention changed by less than 0.05 in nearly all individual × gap cells; the largest changes were one-day
retention in T9 (0.51 → 0.56) and T7 (0.59 → 0.48). The all-pairs results are kept as a sensitivity analysis in the
Supplement.

## Rejected alternatives

- Keeping all pairs: intervals become meaningless when the denominator is near zero.
- Differences (old R² − reference R²) instead of ratios: avoids the instability but changes every reported quantity
  and breaks comparability with prior work that reports ratios (e.g. retention in Degenhart et al. 2020).

## Addendum (2026-10-08): T11

T11 (196 sessions, 677 pairs) failed the participant rule: median reference R² −16.2, only 155/677 pairs above 0.1.
`scripts/tools/scan_artifacts.py` showed why: in 107 of 196 sessions the held-out segment contains values more than
10³ s.d. from the session's training segment (T7 2/35, T8 4/58, T10 1/57), consistent with scale jumps between
recording blocks. Block-wise z-scoring, as in the dataset paper, rescued most affected sessions (day 630: R²
−8.5×10⁸ → 0.33; day 596 → 0.11; day 413 stayed −0.23) and left clean sessions unchanged (0.52 → 0.52, 0.52 → 0.55).

Decision: the pre-specified rule stands, so T11 is not in the main analysis (six human participants). T11 is analysed
with block-wise normalization (`IBCI_BLOCKNORM=1`; results/replication_bg_blocknorm, results/decay_alpha_blocknorm) and
reported as a sensitivity analysis. The main pipeline keeps session-wise normalization, under which block-wise
normalization was equal or slightly worse for T6 and T7 (check_feature_normalization.py).
