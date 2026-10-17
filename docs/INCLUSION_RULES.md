# Inclusion rules for participants and session pairs

Retention is the R² of an old (corrected) decoder divided by the R² of a reference decoder trained on the test day.
It is undefined when the reference decoder does not work. Two rules handle this; both are implemented in
`src/ibci/participants.py` and used by every summary script.

## 1. Participants

A human participant enters the decay and correction analyses only if the median R² of its same-day reference decoder
across its session pairs is above 0.1.

The rule was fixed when the first added participant (T3) failed, before the results of the other added participants
(T2, T7, T8, T10, T11) were seen.

| Participant | Median reference R² | Included |
| --- | --- | --- |
| T6, T5, T9, T7, T10, T8 | 0.25 to 0.50 | yes |
| T2 | −0.02 (120 pairs) | no |
| T3 | −0.05 (12 pairs) | no |
| T11 | −16.2 (677 pairs) | no |

T2 and T3 had very long movement periods (median 15.6 s, against 1.6 to 3.9 s in the included participants), and
neither block-wise normalization nor threshold crossings gave a usable decoder for T2. In T11, recordings jumped in
scale between recording blocks. When each block was z-scored separately (`IBCI_BLOCKNORM=1`), most affected T11
sessions decoded normally; T11 is reported as a sensitivity analysis (`results/replication_bg_blocknorm/`), not in the
main analysis.

## 2. Session pairs

Within included individuals, session pairs whose reference R² is at most 0.1 are left out of retention summaries. The
same rule applies to all nine individuals, monkeys included (function `valid_pairs`). It removed 0 to 18% of pairs per
individual:

| N | C | M | T6 | T5 | T9 | T7 | T10 | T8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0% | 10.3% | 2.3% | 15.7% | 3.5% | 11.9% | 17.9% | 15.2% | 7.7% |

Median retention changed by less than 0.05 in nearly all individual and gap cells. Results with every pair kept are
produced with `IBCI_ALL_PAIRS=1 python scripts/v2_summary.py` (`results/v2_summary_allpairs.json`).

## Alternatives considered

- **Keeping all pairs.** Confidence intervals become meaningless when the reference R² is near zero.
- **Differences instead of ratios** (old R² minus reference R²). This avoids the instability but changes every reported
  quantity and breaks comparability with earlier work that reports retention as a ratio. The paper reports absolute
  R² differences as a sensitivity analysis.
