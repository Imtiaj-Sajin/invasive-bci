# Status: morning of 2026-10-06

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
- Simulator validation on held-out years and untargeted statistics (`results/sim_validation`).
- Then: package the simulator as a reusable tool, write the methods section, add the BrainGate 20-year human data.

## Needs you
- **qBittorrent, EA Desktop and NVIDIA Broadcast were closed** to free bandwidth and GPU. qBittorrent was choking downloads.
  Reopen them if you need them.
- **BrainGate 20-year data (optional, valuable):** Dryad blocks automated downloads. In a browser, open
  https://doi.org/10.5061/dryad.x0k6djj1h, download the `yield_*.tar.gz` files (5.3 GB total) and the README, and put them in
  `D:/ibci-data/braingate/`. They would add human electrode-failure statistics across 14 participants.
- **Git history:** two early commits contain a "Co-Authored-By: Claude" trailer. All later commits are yours only.
  Removing the trailer from those two means rewriting history and force-pushing; say if you want that.
