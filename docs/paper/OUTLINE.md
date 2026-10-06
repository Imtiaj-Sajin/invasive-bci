# Paper outline (living document; updated 2026-10-06 morning)

**Working title:** *How intracortical BCIs age: conserved decoder drift, a cross-species electrode failure process, and
a calibrated simulator, with practical recipes for robust decoding*

**Target venues:**

- Primary: Journal of Neural Engineering or IEEE TNSRE (Q1).
- Stretch: Nature Communications or Communications Engineering (multi-species, 20 human arrays, open simulator).
- Companion option: NeurIPS Datasets & Benchmarks (simulator plus benchmark).

## Story in one paragraph

Implanted BCIs degrade as recordings change. Using public multi-year data from 3 monkeys (2 labs) and 20 human Utah arrays
(BrainGate, 14 participants), we show four things:

1. Decoder decay under standard daily renormalization follows a broadly conserved time course. About 25% of performance is
   lost overnight and about half within a week, and the decoder is near useless by about 4 months.
2. Label-free fixes (subspace alignment) and label-free health monitoring do not recover or predict the loss.
3. Electrodes fail through a common process: losses are mostly *transient* (about 80% of silenced electrodes revive), abrupt
   single-electrode changes are frequent, spiking declines faster at array *edges* (consistent with micromotion strain), and
   impedance falls over years.
4. A simulator calibrated to these measurements reproduces held-out decoder decay 3–4× better than a no-drift null, and
   exposes that drift slows later in an implant's life.

We turn the measurements into recipes:

- **regularize for the future:** free in same-day accuracy, sizeable cross-day gains for SBP decoders;
- **history-informed ridge-to-prior recalibration:** about 100 trials recover 80–90%, and harmful recalibrations disappear.

## Results sections and figures

| # | Section | Key result | Figure / file |
| --- | --- | --- | --- |
| 1 | Data and protocol | LINK 3.4 y, 312 sessions; 000688 Chewie and Mihili; BrainGate 20 arrays and 2,289 sessions; FALCON H2 | schematic (to do) |
| 2 | Conserved decoder decay | retention 0.73/0.54/0.19/0.04 (N); similar in C and M; LSTM more accurate, modestly more robust | `fig_anatomy` (+ replication panel, to do) |
| 3 | Label-free fixes and monitoring fail | Procrustes and stable-channel alignment ≤ renormalization in 3 monkeys; 14 label-free features ≤ calendar (split-half 0.84) | pilot and ladder tables |
| 4 | What does fix it, and at what cost | per-channel gains recover 15–50%; ridge-to-prior with history shrinkage; 100 trials → 80–90%; decoder's top-32 channels slightly better at ≤ 50 trials | `fig_efficiency`; recal_policy and targeted_recal tables |
| 5 | Regularize for the future | α = 1e4: same-day +0.003; cross-day +0.02 to +0.09 (N), +0.01 to +0.03 (C), about +0.001 (M) | `fig_reg_tradeoff`; `rule_across_subjects.csv` |
| 6 | Electrode failure across species | revival 0.80 (humans) vs 0.74 (N); edge effect Fisher p = 2e-8, 14/19 arrays; impedance falls in 17/18 arrays | `fig_failure_xspecies`, `fig_health` |
| 7 | Calibrated simulator | fit loss 0.074; held-out error 3–4× below a no-drift null; drift slower late in implant life (v2: age-dependent parameters) | sim figure (to do) |
| 8 | Uses of the simulator | augmentation < regularization control (honest negative); benchmarking and policy evaluation | augment table |

## Pending items

- **Human decoder drift:** BrainGate decoding T6 (124 sessions over 3.1 y). The owner will download it.
  The script `replicate_braingate_decoding.py` is ready.
- **Simulator v2:** late-period calibration (running), then an age-dependent parameterization and validation.
- **Figures:** the replication panel for figure 2, the simulator figure, and the schematic.
- **Literature check before writing:** what Hahn et al. *Nat Med* 2026 already report on edge effects, impedance and
  revivals in the same BrainGate data, so the novelty is stated precisely.
  - Expected new: the cross-species comparison, the Markov switching rates, abruptness, and the link to decoder drift and the simulator.

## Corrections made during analysis (keep in the paper's methods and limitations)

- A free 96×96 input remap is retraining-equivalent, so it is not a structural test of drift.
- k-channel recovery reflects decoder importance, not localized drift.
- The simulator must be scored with exactly the same correction and regularization pipeline as the real data.

## Related work to position against

- **Stability:** Sussillo 2016; Degenhart 2020; Gallego 2020; ADAN and CycleGAN (Ma 2023); NoMAD 2025; PRI-T (Wilson 2025,
  which includes a closed-loop drift simulator); CORP; SPINT; FALCON.
- **Recording longevity:** Sponheim 2021; Hahn 2025/2026; Colachis 2021; Chen 2023; Barrese 2013/2016; Woeppel 2021;
  Bjånes 2025; Forrest 2025 (edge strain); Perge 2013.
- **Simulation:** MEArec; SpikeInterface; Wan 2023 ("relatively arbitrary" parameters).
- **Data:** LINK (Temmar 2025); 000688 (Perich/Miller); BrainGate release (Hahn 2026); FALCON (Karpowicz 2024).
