# Decision 0002: pivot to the anatomy of chronic drift and a data-calibrated drift/failure simulator

- **Date:** 2026-10-05 (late evening)
- **Status:** Accepted. Supersedes 0001.

## Why 0001 was dropped
The LINK pilot (logged in `docs/RESEARCH_LOG.md`, 2026-10-05, "PILOT") showed:
- Under realistic deployment (daily unsupervised renormalization), decoder performance decays smoothly with elapsed time.
- The day-to-day variation left after the time trend is *real*: split-half reliability is 0.84.
- That variation is *invisible* to all 14 label-free signals tried (shift statistics, ensemble and cross-view
  disagreement, output self-consistency, Procrustes alignment, impedance, dead channels):
  - partial correlations given days are about 0.1 or less;
  - in cross-validated prediction, nothing beat a time-only model;
  - unexpected-bad-day detection AUROC was 0.50–0.58.

0001's main claim therefore fails. The negative result is kept, and becomes an explained finding inside the new paper.

## New topic

**Working title:** *The anatomy of chronic neural drift in intracortical BCIs, and a data-calibrated simulator for
building decoders that survive it.*

**Core questions:**
1. **Anatomy (guaranteed descriptive core).** When a fixed decoder degrades over days to years, *which kind of change*
   is responsible, and on what timescale? We use an "oracle ladder" of increasingly powerful corrections on the frozen decoder:
   1. unsupervised recentring;
   2. unsupervised renormalization;
   3. supervised per-channel gain/sign, with 96 parameters;
   4. supervised latent-subspace alignment;
   5. supervised full input remapping;
   6. full retraining.

   The size of each step attributes the loss to one cause:
   - per-channel offset and gain changes;
   - channel-level tuning turnover;
   - cross-channel mixing;
   - changes that no input remapping can fix.

   We also measure how many labelled trials each correction needs. That bears directly on calibration burden.
2. **Electrode failure process.** Using LINK's per-channel trajectories over 3.5 years (with grid positions and impedance), we ask:
   - Are failures abrupt or gradual?
   - Do failed channels recover?
   - Are failures clustered in space, and do edge electrodes fail more than centre ones (the micromotion-strain hypothesis of Forrest 2025)?
   - How do failures relate to impedance?
3. **Simulator.** A generative, feature-level model (binned spike-band power and threshold crossings, plus kinematics) of how
   recordings change with elapsed time, with each process fitted to (1) and (2):
   - offset and gain random walks;
   - tuning drift;
   - mixing;
   - noise changes;
   - channel death, revival and fault events.

   Validation:
   - it should reproduce real decoder survival curves, feature-drift statistics and failure statistics;
   - it is fitted on LINK and *validated on held-out time periods and on other monkeys* (DANDI 000688: Chewie about 3 years, Mihili about 1.5 years).
4. **Uses of the simulator:**
   1. Stress-test decoders and unsupervised stabilizers under controlled drift and failure severity.
   2. "Train on simulated futures": augmentation sampled from the fitted model, compared on real held-out future days
      against no augmentation, ad hoc perturbations (Sussillo 2016) and multi-day training.
   3. Explain the monitoring result: if the damaging changes are channel-level tuning changes that leave input
      statistics intact, label-free monitors cannot see them.
5. **Release:** a pip-installable simulator and benchmark.

## Why this is the best available bet (honest scoring after the pilot)
- **Guaranteed core.** The oracle-ladder anatomy, the failure statistics and the simulator's validation all yield publishable results
  whatever the outcome. The only uncertain piece is the *size* of the augmentation benefit (4b).
- **Novelty.** No feature-level or biophysical simulator of *chronic* drift and failure exists (degradation sweep: MEArec,
  SpikeInterface and Kilosort simulate drift only). Wan 2023 explicitly asks for data-derived degradation models, calling its own
  parameters "relatively arbitrary." LINK's own analyses cover mean spike-band power, active channels, participation ratio,
  tuning, cross-day decoding and continual learning. They include no decomposition of decoder loss and no failure-process model.
- **Impact and reuse.** It speaks to the top two bottlenecks (nonstationarity and recalibration; electrode degradation). Decoder and
  stabilizer developers can stress-test methods without years of data, and hardware groups get quantitative failure statistics.
  Simulators in neuroscience are long-lived citation sinks: MEArec has about 200 citations and SpikeInterface about 600.
- **Feasibility.** Linear and small neural decoders, about 13 GB per dataset, all on DANDI.
- **Fit with the owner's interest.** "Intracortical electrodes / invasive BCI" maps directly to the electrode failure and drift work.

## Alternatives re-scored after the pilot
| Option | Verdict |
|---|---|
| Electrode-failure robustness benchmark plus failure-aware training | Close to Sussillo 2016 (perturbation training) and Vasko 2022 (detection). Kept as a *component* (Q2, Q4a). |
| Year-scale benchmark of unsupervised stabilizers | Wilson et al. 2025 already compares methods long-term on T5, and insiders (Chestek/SNEL, who run FALCON) are best placed. Becomes a *use case* (Q4a). |
| "How much history or recalibration?" | Largely covered by LINK's own continual-learning and multi-day analyses. |
| Monitoring paper with negative results | Kept as an explained finding (Q4c), not the headline. |

## Data
- **Calibration:** LINK (DANDI 001201).
- **Held-out validation:** DANDI 000688 (Chewie, Mihili).
- **Optional:**
  - FALCON H2 (human, about 17 months).
  - BrainGate-20 yield tarballs (5.3 GB, Dryad; the owner must download them manually). These would give human array failure statistics.
