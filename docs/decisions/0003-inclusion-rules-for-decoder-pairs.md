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
