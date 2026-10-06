# Prior-art check for the v2 findings (2026-10-07)

*Research sub-agent; full texts via PMC/Europe PMC, arXiv, publisher PDFs and public code. Supersedes claims C3 and
C6a in `06_prior_art_check.md` (gains "15–50%" and "80% revive" were revised by the 2026-10-06 re-analyses).*

## Verdicts

| Finding | Status | Key prior work | Must address in the paper |
| --- | --- | --- | --- |
| (1) One gain per channel after new-day z-scoring recovers 70–99% of the loss in three humans (linear and LSTM), less in monkeys | Partially reported | Bishop 2014 (monkeys, <2 months, *additive* per-electrode shifts; doi:10.1088/1741-2560/11/2/026001); Jarosiewicz 2015 (baseline subtraction); CORP, Fan 2023 (frozen GRU + full affine day layer, T5, arXiv:2311.03611); Willett 2021/2023 (day-specific input layers); POYO and APST (richer per-unit parameters recover monkey C) | Wilson 2025 (doi:10.1038/s41551-025-01536-z) frames drift as rotation: their weight cosine is invariant only to *global* scale, so per-channel gains look like rotation (testable: `scripts/weight_angles.py`). Pun 2024 reports preferred-direction changes in most tuned features (T11, T5). LINK reports stable tuning angles in monkey N. Boccato 2026 J Neural Eng (doi:10.1088/1741-2552/ae8576) may report near-diagonal day transforms [UNVERIFIED; check before submission] |
| (2) Procrustes adds nothing beyond renormalization; CORAL a little; FA stabilizer worse offline | Partially reported | CORP (FA stabilizer unsuccessful); NoMAD (doi:10.1038/s41467-025-59652-y; aligned FA insignificant on monkey C, static decoder not normalized); Ma 2023 (PAF last); Wilson closed loop (FA variable) | Wilson offline: stabilizer beats *mean-only* recalibration on T5 — our mean-only vs renormalization ablation answers this |
| (3a) Fixed-decoder half-life 1–125 days across individuals | Partially reported | Nuyujukian 2014 (doi:10.1088/1741-2560/11/6/066003); NoMAD half-life; Hahn 2025 tuning-stability curves (repo nptl-stanford/array-paper); LINK double exponential | Flint 2013: offline variability can overstate online decay |
| (3b) Under-regularized ridge exaggerates drift | Apparently new | Sussillo 2016 (augmentation); Shaikh 2020 TNSRE (channel bagging; doi:10.1109/TNSRE.2019.2962708) | Sussillo: linear FIT-KF did not benefit from multi-day training |
| (4) Silence-then-revival mostly matches a shuffled null; only ≥ 30-day silences depart | Apparently new | Barrese 2013; Downey 2018 ("units frequently return"); Chestek 2011; Eleryan 2014; Colachis 2021; Hahn 2025 (array-level anecdotes) | Rule out connector/pedestal events (array-wide-shift removal does this partly) |

## Suggested novelty statements (from the search; adapt to final numbers)

1. Per-electrode drift has been modelled as baseline shifts and absorbed by full day-specific affine input layers, but
   whether a strictly diagonal correction suffices had not been isolated; across 1–480 days in three people it recovers
   most of the loss for linear and recurrent decoders, and markedly less in three macaques.
2. Prior label-free alignment evaluations used unnormalized or mean-only baselines; against full per-channel
   renormalization, Procrustes alignment adds nothing, CORAL little, and the FA stabilizer performs below it offline.
3. Fixed-decoder half-lives span about 1 to 125 days across six individuals, and a weak ridge penalty is itself a
   confound that inflates apparent drift.
4. At the standard yield criterion, most electrode revivals are indistinguishable from a shuffled-session null; only
   silences longer than about 30 days depart from it.

## Additional references to consider citing

Brandman et al. 2018 J Neural Eng 15, 026007 (rapid calibration for people with tetraplegia); Natraj et al. 2025 Cell
188, 1208 (stable representational geometry, ECoG); Deo et al. 2024 Sci Rep (day-specific affine transform); Azabou et al.
2023 NeurIPS (POYO); Hahn et al. 2025 medRxiv 10.1101/2025.07.02.25330310.
