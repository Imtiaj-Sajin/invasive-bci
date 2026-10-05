# Code survey: simulators, stabilizers, LINK tooling (2026-10-05)

*Compiled by a research sub-agent using the GitHub API (via `gh api`), raw sources, license files, and the PyPI, DANDI,
HuggingFace, arXiv and Europe PMC APIs. Dates are last commits on the default branch.*

## Verdict on novelty
**No public binned-feature simulator of *chronic* drift plus electrode failure, calibrated on multi-year data, exists.**

- **Closest prior art:** the closed-loop simulator in Wilson et al., *Nat BME* 2025
  ([guyhwilson/nonstationarities](https://github.com/guyhwilson/nonstationarities), **no license**). Its `simulateTuningShift` models:
  - preferred-direction shrinkage α (E' = αE + ε), fitted to T5;
  - a mean shift;
  - n_stable;
  - SNR resampling.

  It has no channel death, no per-channel gain, no spatial structure and no SBP features. Our simulator must be positioned against this one.
- **Everything else is weaker:**
  - hand-set perturbations: Degenhart 2020, Wan 2023 (code not public), Stephens 2021 (no code), Sussillo 2016 and Kao 2017 (code on request);
  - augmentation noise: LINK and `neuraldecoding` (white noise, random walk, offset, time warp), and Wei et al. 2026 (arXiv 2607.24023).
- **Positioning:** a *calibrated generative model* (explicit mechanisms, fitted time courses, held-out validation), **not augmentation**.

## Spatial locality of cross-day change
- No paper or code shows signals shifting between neighbouring Utah electrodes.
- At the 400 µm pitch, adjacent electrodes should not share units, so locality is plausible only for SBP or multi-unit
  hash, reference or common-mode effects, or tilt-induced gain fields.
- The spatial evidence that does exist concerns *degradation*:
  - Forrest 2025, JNE ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12624975/)): edge and corner strain correlates with impedance.
  - Hahn 2025: insulation degrades at array edges.
- **Our interim locality test agrees: only a weak local advantage. Do not overclaim.**
- Grid maps: LINK has `row`/`col`. **DANDI 000688 has no row/col** (only bank/pin/label), so spatial analyses are LINK-only.

## Simulators
| Repo | License | Drift/failure? |
|---|---|---|
| [agencyenterprise/neural-data-simulator](https://github.com/agencyenterprise/neural-data-simulator) | Apache-2.0 | No |
| [jtcostello/bcisimulator](https://github.com/jtcostello/bcisimulator) (Chestek lab, closed-loop finger/cursor) | MIT | No; a possible closed-loop harness |
| [brandbci/brand-simulator](https://github.com/brandbci/brand-simulator) | MIT | No (Linux/Redis) |
| [SpikeInterface/MEArec](https://github.com/SpikeInterface/MEArec), SpikeInterface generation | MIT | Minutes-scale motion only |
| Toy 2026 repos (neural-cursor, NeuroDrift-CSA, bci-fault-bench-intracortical, …) | — | Uncalibrated toys |

## Stabilizers usable on Windows with PyTorch and 8 GB
| Method | Code | Status |
|---|---|---|
| Cycle-GAN (Ma 2023) | [limblab/adversarial_BCI](https://github.com/limblab/adversarial_BCI) (PyTorch, no license) | Run for comparison |
| ADAN | [farshchian/ADAN](https://github.com/farshchian/ADAN) (TF1) | Reimplement in PyTorch |
| Degenhart stabilizer (FA + Procrustes) | No official code; `stabilizer_utils` in `nonstationarities` (no license); `PAF` in [neuraldecoding](https://github.com/Neuro-core-hub/neuraldecoding) (MIT) | Reimplement (simple) |
| Gallego CCA | No public code | Reimplement |
| PRI-T | [guyhwilson/PRI-T](https://github.com/guyhwilson/PRI-T) (**Stanford research-only, derivatives belong to Stanford**) | Compare only; 2D cursor |
| RTI | Python version in `nonstationarities` | Compare only |
| SPINT | [shlizee/SPINT](https://github.com/shlizee/SPINT) (BSD-3, PyTorch) | Likely works |
| ERDiff | [yulewang97/ERDiff](https://github.com/yulewang97/ERDiff) (MIT, PyTorch) | Works |
| TCLA | [FAMD-CASIA/TCLA](https://github.com/FAMD-CASIA/TCLA) (PyTorch, no license; uses Chewie/Mihili) | Works |
| NDT2/NDT3 | [context_general_bci](https://github.com/joel99/context_general_bci) (MIT); NDT3 weights CC-BY-NC | 45M model possible |
| NoMAD | [snel-repo/nomad](https://github.com/snel-repo/nomad): TF2 + CUDA 10, **non-commercial, no derivatives, patented** | Not practical; do not port |
| CORP | TF 2.7, ≥12 GB GPU, speech-specific | Not applicable |
| FALCON demos | [snel-repo/falcon-challenge](https://github.com/snel-repo/falcon-challenge) (MIT): Wiener filter, RNN, NDT2 | The paper's NoMAD and CycleGAN baselines are not in the repo |

Also: [ewinapun/MINDFUL](https://github.com/ewinapun/MINDFUL), KL-divergence instability metrics (MATLAB).

## LINK tooling
- [chesteklab/LINK_dataset](https://github.com/chesteklab/LINK_dataset) (no license):
  - pickles at 32 ms bins (NWB files are 20 ms);
  - analyses of average SBP, participation ratio, active channels, tuning stability and mutual information, PCA by year, a cross-day matrix of
    single-day LSTM and ridge models (with and without refreshed normalization), multi-day training and continual learning;
  - **no drift decomposition, no simulator.**
- [Neuro-core-hub/neuraldecoding](https://github.com/Neuro-core-hub/neuraldecoding) (MIT):
  - LINK-dev branch: Kalman, ridge, LSTM, CNN, FA+Procrustes, augmentation noise;
  - main branch: adds **tcFNN**, RNN and transformer. The reference tcFNN is [chesteklab/willseyetal2022](https://github.com/chesteklab/willseyetal2022).
- brainsets / torch_brain:
  - 000688 is supported (`perich_miller_population_2018`).
  - **LINK is not merged.** An open PR ([brainsets#80](https://github.com/neuro-galaxy/brainsets/pull/80)) extracts threshold crossings only, with row/col.
  - The POYO HF weight repos are empty. POSSM code is "coming soon"; its only checkpoint is IBL.

## Licensing caveat
Several key repos have **no license**: LINK_dataset, nonstationarities, adversarial_BCI, ADAN, CORP, TCLA. We may run them for comparison
but must not redistribute modified copies. Reimplement where we need the method inside our package.
