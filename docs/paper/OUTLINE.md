# Paper outline (living document)

**Working title:** *Anatomy of chronic neural drift in intracortical BCIs, and a calibrated simulator for building
decoders that survive it*

**Target venues:**
- Primary: Journal of Neural Engineering or IEEE TNSRE.
- Stretch: Nature Communications or Communications Engineering, if the human replication via BrainGate-20 lands.
- Companion: NeurIPS Datasets & Benchmarks, for the simulator plus benchmark.

## One-paragraph story
Every implanted BCI decoder degrades as recordings drift. The field treats the result with recalibration, alignment and
foundation models, but the *anatomy* of the drift has never been decomposed: which changes in the signal break which
decoders, on which timescale, and how much labelled data each fix needs. We decompose decoder loss over 3.5 years of
daily Utah-array recordings (LINK) with an *oracle ladder* of increasingly powerful corrections, characterize the
electrode failure process, and replicate in two more monkeys from another lab (DANDI 000688). From these measurements
we build the first *calibrated* simulator of chronic drift and electrode failure. It is fitted so that its own oracle
ladder matches the real one, and validated on held-out years and animals. We show what it is good for: stress-testing
decoders and stabilizers, training decoders on simulated futures, and explaining why label-free monitoring of decoder
health fails under natural drift.

## Contributions (target)
1. **Drift anatomy.** A quantitative decomposition of decoder loss into:
   - per-channel offset and scale (fixed by renormalization);
   - per-channel gain;
   - cross-channel remixing;
   - changes that need a new decoder.

   Each is measured as a function of days elapsed, with the labelled-trial budget each correction needs (calibration burden).
2. **Electrode failure process.** Per-channel activity, tuning and impedance trajectories over 3.5 years:
   - death and revival rates;
   - abrupt vs gradual changes (heavy tails);
   - spatial correlation of decline;
   - edge vs interior electrodes (a test of the micromotion-strain hypothesis);
   - links to impedance.
3. **A calibrated simulator** (open source): instant plus slow components of mixing and turnover, death and revival,
   gain jumps, raw offset and scale walks. It is calibrated by matching the ladder (simulation-based calibration) and validated on
   held-out time and held-out animals.
4. **Uses:**
   1. A benchmark of decoders and unsupervised stabilizers under controlled drift and failure severity.
   2. "Train on simulated futures," compared against regularization-only, ad hoc perturbations (Sussillo 2016) and multi-day training.
   3. An explanation of the negative label-free monitoring result. The pilot found reliable day-to-day variation
      (split-half reliability 0.84) that 14 label-free statistics cannot see.
5. **Practical guidance.** How often and with how much data to recalibrate, and which recalibration parameterization to use
   (e.g. ridge shrunk toward the previous decoder).

## Figure plan
1. Data and protocol: LINK timeline, the oracle-ladder schematic, and an example of decay.
2. **Anatomy:** R² per rung vs gap, plus share of loss recovered (`make_figures.py anatomy`).
3. **Calibration burden:** R² vs labelled trials per correction and gap (`make_figures.py efficiency`).
4. **Electrode failure:** activity heatmap, active channels, impedance, spatial and edge analyses (`make_figures.py health`).
5. **Simulator:** schematic plus the calibrated fit, with real vs simulated ladders on held-out years and held-out monkeys.
6. **Uses:** the augmentation results, the stabilizer stress test, and the monitoring explanation.
7. Replication in Chewie and Mihili (000688).

## Related work to position against
- **Stability and alignment:**
  - Sussillo 2016 (Nat Commun; robustness to future variability through perturbations).
  - Degenhart 2020 (Nat BME; stabilizer).
  - Gallego 2020 (Nat Neurosci; stable latent dynamics).
  - Farshchian 2019 and Ma 2023 (ADAN, CycleGAN).
  - NoMAD (Karpowicz 2025).
  - Wilson 2025 (PRI-T, Nat BME).
  - CORP (Fan 2023).
  - SPINT (NeurIPS 2025).
  - FALCON (NeurIPS 2024).
- **Long-term recording quality:**
  - Sponheim 2021.
  - Hahn 2025/2026 (BrainGate 14 participants).
  - Colachis 2021.
  - Chen 2023 (1,024-channel V1/V4).
  - Barrese 2013/2016.
  - Woeppel 2021.
  - Bjånes 2025.
  - Forrest 2025 (micromotion strain).
- **Intra-day instability:** Perge 2013 (J Neural Eng).
- **Simulation:**
  - MEArec.
  - SpikeInterface generation.
  - Kilosort4 drift simulation.
  - Wan 2023 (firing-rate degradation with "relatively arbitrary" parameters; it calls for data-derived models).
  - Stephens 2021 (GAN channel loss).
- **Data:**
  - LINK (Temmar et al., NeurIPS 2025 D&B), the source of the primary data.
  - DANDI 000688 (Perich/Miller).
  - LINK's own analyses cover mean SBP, active channels, participation ratio, tuning, cross-day decoding,
    continual learning and multi-day training. **They include no loss decomposition and no failure-process model.**
- **Label-free accuracy estimation (ML):**
  - ATC (Garg 2022).
  - Agreement-on-the-line.
  - MANO.
  - ProjNorm.
  - Probabilistic Co-Control (Huang et al. 2026, speech calibration).

## Threats to validity (address explicitly)
- **One monkey for the anatomy:** replicate on 000688 (2 monkeys, sorted-unit features, different lab and task).
- **Offline only:** state it. The anatomy is about recorded signals and needs no closed loop. Note the
  closed-loop co-adaptation literature.
- **Target style (CO/RD) confound:** pairs are always same-style.
- **Regularization choices in the supervised rungs:** λ is selected on held-out labelled trials. Report sensitivity.
- **Leakage in simulator validation:** time-split calibration (`--max-day`) and evaluation on later days only.
