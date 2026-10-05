# Decision 0001: research topic

- **Date:** 2026-10-05
- **Status:** Accepted

## Decision

**Working title:** *Knowing when an intracortical BCI decoder fails: label-free reliability monitoring across years of
recordings.*

**One-sentence pitch:** Every implanted BCI decoder degrades as neural signals drift and electrodes age, and today nobody can tell,
without making the user run a labelled calibration session, whether the decoder has silently stopped working. We build and benchmark
methods that estimate a decoder's performance on a new day from *unlabelled* neural data. We test them across up to 3.5 years of
recordings in several subjects, show they can cut the recalibration burden at equal performance, and release them as an
open-source toolkit with a year-scale benchmark.

## Research questions

1. **Characterize.** How does decoder performance decay over days to years, and how much of that decay is visible in
   unlabelled neural data (input statistics, neural subspaces, impedance, dead channels)?
2. **Uncertainty under drift.** Do standard per-sample uncertainty methods stay calibrated across days? We test deep ensembles,
   MC dropout, heteroscedastic heads, and split, adaptive or online conformal prediction.
   *Expected result:* within-day coverage is fine and cross-day coverage collapses, which motivates session-level monitoring.
3. **Label-free performance estimation.** Can a decoder's R² or correlation on a new session be predicted without labels?
   We compare these estimator families under strict temporal validation and cross-subject transfer:
   1. Time elapsed (baseline).
   2. Input-shift statistics.
   3. Model-based signals: ensemble disagreement, dropout variance, and regression adaptations of ATC, agreement-on-the-line,
      DoC and ProjNorm.
   4. Hardware metrics: impedance and channel death.
   5. A learned meta-estimator.
4. **Utility.** Does monitor-triggered recalibration reduce the number of calibration sessions at equal performance,
   compared with fixed schedules? We measure this as a Pareto frontier against an oracle and against "never" and "always" recalibrating.
5. **Hardware faults.** Can the monitor detect, and localize to channels, abrupt electrode failures such as dead channels, noise
   bursts and gain jumps? The fault rates come from published Utah-array yield data. A time-elapsed baseline cannot see these failures by construction.

## Data
All sources are public, and the primary set can be downloaded automatically from DANDI.

| Role | Dataset | Span | Notes |
|---|---|---|---|
| Primary | **LINK**, DANDI 001201, CC-BY | 312 sessions over 1,242 days, Monkey N, finger task | Per-session impedance, threshold crossings and spike-band power. |
| Replication (2nd species and lab) | **DANDI 000688** (Perich/Miller) | Chewie about 3 years, Mihili about 1.5 years, reaching | Different lab, task and array placement. |
| Human | **FALCON H2**, DANDI 000950 | T5 handwriting, 26 sessions over about 17 months | Human, discrete output (characters). |
| Human, optional | **BrainGate 20-year release**, Dryad dryad.x0k6djj1h, CC0 | Up to 7.6 years, 729 closed-loop cursor sessions | Dryad has bot protection, so the owner must download these files manually in a browser. |
| Optional | Brain-to-Text '25 (T15) | 20 months, speech | Only if time allows. Same Dryad caveat; also on Kaggle. |

## Why this topic: the evidence
See `docs/knowledge/01`–`04` and the log entry from 2026-10-05.

- **Impact.** Nonstationarity and the recalibration burden are the most-cited bottlenecks for clinical translation.
  - Wilson et al., *Nat BME* 2025.
  - Card et al., *Nat Med* 2026: home use with continuous background recalibration.
  - Neuralink's Oct 2026 headline result: calibration cut from daily to weekly.
  - FDA and iBCI-CC emphasize "reliability of use."
  - Home use is now arriving at Neuralink, Paradromics (IDE Nov 2025), Synchron (pivotal trial 2026) and Neuracle (NMPA approval Mar 2026),
    so every one of these systems needs a way to know when its decoder is degrading.
- **Novelty, verified 2026-10-05:**
  - Europe PMC finds 0 papers on intracortical uncertainty and 0 on intracortical performance prediction. The only
    conformal-prediction BCI paper is for EEG classification (J Neural Eng 2026).
  - A grep of 603 arXiv BCI abstracts finds only UnSPC (2607.24031), which uses uncertainty internally for pseudo-labels,
    and a perspective that *calls for* uncertainty-paced BCIs (2609.01767).
  - On bioRxiv:
    - Probabilistic Co-Control (2026) studies calibration of speech CTC decoders within a session only.
    - Gontier et al. (2026) detects neural error signals, which is a different mechanism.
  - The LINK paper itself does no uncertainty or monitoring analysis.
  - The closest classic is Perge et al. 2013 (J Neural Eng, about 190 citations): signal instabilities within a day degrade decoding,
    but it does no label-free prediction or monitoring.
  - Label-free accuracy estimation exists in general machine learning (ATC, agreement-on-the-line, MANO, ProjNorm), but
    almost only for classification. Adapting it to continuous neural regression is a methodological contribution in its own right.
- **Feasibility:**
  - Decoders are small (Wiener and Kalman filters, tcFNN, GRU, small Transformers), so 8 GB VRAM is plenty.
  - The datasets are about 13 GB each.
  - There are 312 LINK sessions, which gives about 48k train-day/test-day pairs, so the estimator results will have strong statistics.
- **Certainty of outcome:**
  - RQ1, RQ2 and the benchmark produce publishable results whatever the answer is.
  - RQ5 is guaranteed to show value over a time-elapsed baseline.
  - RQ3/RQ4 only need neural shift to carry *some* information beyond elapsed time, which is very likely given known instabilities
    (Perge 2013) and the impedance and spike-band-power changes already seen in LINK.
  - **Weakest link:** whether label-free estimators clearly beat "days since calibration" on natural drift.
    Mitigation: report this honestly. Fault detection and cross-subject transfer still carry the paper.
- **Reuse and citations:**
  - **Toolkit:** the paper ships a pip-installable monitor that wraps any decoder.
  - **Benchmark:** fixed year-scale splits on three public datasets.
  - **Who would cite it:** recalibration, foundation-model and clinical-trial papers, as a standard reliability evaluation.
  - **Precedent:** comparable "stability" papers have about 235–450 citations each. Sussillo 2016 has 235, Degenhart 2020 has 237,
    and Gallego 2020 has 452 (OpenAlex, 2026-10-05).
- **Venue targets:**
  - Primary: J Neural Eng or IEEE TNSRE (Q1).
  - Stretch, if the BrainGate human results are strong: Nature Communications or Communications Engineering.
  - Optional earlier venue for the benchmark: NeurIPS Datasets & Benchmarks.

## Alternatives considered and rejected (for now)
| Alternative | Why not first |
|---|---|
| Mechanistic simulator of chronic degradation (sweep 4, #1) | Strong novelty, but biophysics calibration lacks public impedance-plus-raw data and is a long, risky project. Good follow-up that can reuse this paper's findings. |
| FALCON-C corruption benchmark (sweep 2, #2) | Folded in as RQ5. Standalone, it is more a NeurIPS D&B paper than a Q1 journal paper. |
| Rate–distortion benchmark for on-implant compression (sweep 4, #2) | Needs raw broadband data (tens of GB per session) on a tight disk. Good follow-up. |
| Shortcut and leakage audit (sweep 2, #3) | Cheap, but the effect sizes are uncertain. Its methodology (strict temporal splits, null controls) is reused here. |
| Speech word-error-rate chasing, new foundation models | Saturated or crowded, and needs more compute. |
| Calibrated uncertainty for speech BCIs (sweep 1, #2) | Overlaps Probabilistic Co-Control (2026); also needs RAM-heavy language models. Possible extension through T15. |

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| Offline-only critique | Monitoring is inherently an offline-evaluable property. BrainGate closed-loop sessions, if downloaded, give closed-loop ground truth. |
| Someone publishes first | Move fast. Post an arXiv/bioRxiv preprint as soon as the core results exist. |
| Session confounds in LINK (CO vs RD target style) | Stratify by style. Report results within each style and across styles. |
| Estimator overfits to one monkey | Train the estimator on LINK and test unchanged on Chewie, Mihili and T5 (cross-subject transfer is a headline result). |
