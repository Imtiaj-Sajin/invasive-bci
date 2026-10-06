# Status (updated 2026-10-06, late morning)

A short, honest summary of the first night of work. Every number, caveat and correction is in
[RESEARCH_LOG.md](RESEARCH_LOG.md). The plan is in [paper/OUTLINE.md](paper/OUTLINE.md).

## What happened

1. **Research (evening).** Four parallel literature and dataset sweeps went into `docs/knowledge/01–05`. The first topic chosen
   was *label-free monitoring of decoder health* (decision 0001).
2. **The pilot killed topic 1.** Day-to-day decoder performance varies reliably (split-half reliability 0.84), but none of
   14 label-free signals predicts it beyond elapsed time. That is a real negative finding, kept for the paper.
3. **Pivot (decision 0002):** the *anatomy of chronic drift*, the *electrode failure process*, and a *calibrated
   simulator* of both. The code survey confirmed that no calibrated chronic-drift simulator exists. The closest is the
   Wilson et al. 2025 closed-loop simulator, which is simpler and uncalibrated on multi-year data.
4. **Data:** LINK (3.4 years, monkey N), DANDI 000688 (monkeys Chewie and Mihili, another lab), and FALCON H2 (human T5),
   about 27 GB in `D:/ibci-data`.

## Results that look solid

- **Conserved decay.** Under daily renormalization, a frozen decoder loses about 25% of R² overnight and about half in a week,
  and is near zero by about 4 months. The pattern is similar in 3 monkeys across 2 labs, tasks and feature types.
- **Label-free fixes fail.** Procrustes and stable-channel alignment never beat plain renormalization (3 monkeys).
- **Per-channel gains explain little.** They recover about 15–50% of the loss.
- **Recalibration recipe.**
  - Method: ridge shrunk toward the old decoder, intercept included, with the shrinkage strength borrowed from history.
  - Result: 100 trials (about 3 minutes) recover 80–90% of own-day R², and harmful recalibrations at 10–20 trials drop from about 10% to about 1%.
- **LSTM vs ridge, on matched pairs:** the LSTM is +0.13 R² more accurate and modestly more drift-robust.
- **Electrode failure.**
  - Impedance falls about 45% over 3 years and active channels halve.
  - Channel loss is mostly *transient* (alive/silent switching, rates fitted).
  - Abrupt single-electrode jumps are common.
  - Spiking declines faster at array *edges*, while broadband power declines uniformly.
  - Silencing and revival also appear in Chewie, Mihili and human T5.
- **Human electrode failure, all 20 BrainGate arrays** (14 participants, 2,289 sessions; downloaded by the owner):
  - 80% of silenced electrodes revive (monkey N: 74%).
  - Edge electrodes lose spiking faster: Fisher p = 2e-8, 14 of 19 arrays across species.
  - Impedance falls in 17 of 18 arrays.
  - Hahn et al. 2026 (same data) note edges only qualitatively and report no electrode-level revival statistics, so these are new.
- **Regularization replicated** in Chewie and Mihili: always helps in direction; large for LINK SBP, smaller for sorted units.
- **Simulator validated** on held-out years: 3–4× better than a no-drift null. Drift is slower later in implant life
  (simulator v2 with age-dependent parameters in progress).
- **Simulator calibrated** on the first 700 days: loss 0.074; it reproduces the real ladder at 1–480 days.
- **Regularize for the future.** Ridge α = 10⁴ instead of the common 0.1 leaves same-day accuracy unchanged, but cross-day R²
  is about 50% higher at 30 days and 2.6× higher at 120 days. Same-day CV cannot see this. Simple rule: choose the largest α that keeps
  same-day R² within about 1% (`results/figures/fig_reg_tradeoff.png`).

## Honest negative result

- **Training on simulated futures** beats ad hoc perturbations (+0.010 R²), but plain stronger regularization beats both
  (+0.042). For linear decoders the simulator is a benchmarking and policy tool, not an augmentation tool. Network decoders are untested.

## Claims I made during the night and then corrected

- "A 96×96 input remap restores about 80%, so drift is channel re-mixing": **withdrawn**. That remap can represent any decoder,
  so it is retraining in disguise.
- "Drift is concentrated on a few electrodes, findable without labels": **not supported** beyond the decoder's own
  reliance on its most-used channels.
- "The LSTM is far more drift-robust": based on one pair. On matched pairs it is only modestly more robust.

## Still running or next

- Simulator v2: late-period calibration running (`results/sim_calib_late`), then an age-dependent parameter set.
- Human decoder drift: waiting for the owner's `decoding_T6.tar.gz`. The script is ready.
- Figures done: `fig_decay_xsubject`, `fig_anatomy`, `fig_efficiency`, `fig_reg_tradeoff`, `fig_health`, `fig_failure_xspecies`.
  Next: the simulator figure, then drafting the results text.

## Needs you

- **qBittorrent, EA Desktop and NVIDIA Broadcast were closed** to free bandwidth and GPU. qBittorrent was choking downloads.
  Reopen them if you need them.
- **BrainGate decoding data (optional):** please download `decoding_T6.tar.gz` (6.9 GB) by clicking its name on the Dryad
  page. The yield files are done, thank you.
- **Git history:** two early commits contain a "Co-Authored-By: Claude" trailer. All later commits are yours only.
  Removing the trailer from those two means rewriting history and force-pushing; say if you want that.
