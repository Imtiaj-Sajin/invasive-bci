# Methods (draft, 2026-10-06)

*Working draft. Numbers come from the scripts named in brackets; see `results/README.md`.*

## Datasets
- **LINK** (DANDI 001201; Temmar et al., NeurIPS 2025 D&B).
  - Subject: one rhesus macaque (monkey N) with two Utah arrays in the hand area of M1. 96 channels were recorded: all 64 of
    the medial array and the 32 contacts in columns 0–3 of the lateral array.
  - Recording: 312 sessions on 303 days spanning 1,242 days (2020-01-27 to 2023-06-22), with a two-finger-group flexion task
    using center-out (CO) or random (RD) targets.
  - Features: spike-band power (SBP) and threshold crossings (TC, −4.5 RMS) in 20 ms bins, plus finger position and
    velocity (4 outputs) and per-session electrode impedance (stored in 228 of 303 days).
  - Each session has 375 trials.
- **DANDI 000688** (Perich/Miller lab).
  - Subjects and span: monkey C (Chewie), 68 sessions, 2013-10 to 2016-10; monkey M (Mihili), 28 sessions, 2014-01 to 2015-06.
  - Arrays and task: Utah arrays in M1 and PMd (up to 192 electrodes), center-out and random-target reaching.
  - Features: spike-sorted units summed per electrode in 20 ms bins, plus cursor position and velocity.
- **FALCON H2** (DANDI 000950).
  - Subject: human participant T5 (BrainGate2), 192 channels, attempted handwriting.
  - Features: threshold crossings in 20 ms bins. The 20 full held-in calibration sessions span 2022-05 to 2022-12.

## Decoders and protocol
- **Split.** Each LINK session is split as in the LINK paper: the first 300 trials train and the remaining trials test.
  000688 uses the first 80% of trials for training.
- **Features.** Each channel is z-scored with the statistics of the decoding day's own training segment, used without labels.
  This "daily renormalization" is the deployment baseline.
- **Linear decoder.** Ridge regression on 8 causal lags (160 ms) of z-scored features with an unpenalized intercept. The default
  α = 0.1 matches the LINK paper.
- **Network decoder.** An LSTM (hidden 256) on 20-bin windows, trained with Adam (1,500 iterations, batch 256)
  [`drift_anatomy_nn.py`].
- **Metric.** R² averaged over the 4 kinematic outputs, computed on the decoding day's held-out trials.

## Cross-day pairs
For each training session and each target gap g ∈ {1, 2, 4, 7, 14, 30, 60, 120, 240, 480, 900} days, the later session of the
same target style whose gap is closest to g (within 25%) is used. Analyses use 80 training sessions drawn at random with a
fixed seed (537 pairs) unless stated otherwise [`drift_anatomy.py`].

## Oracle ladder
Corrections are applied to a frozen day-i decoder evaluated on day j:

| Rung | Correction | Labels |
|---|---|---|
| L0 | none (day-i normalization) | none |
| L1 | channel recentring | none |
| L2 | channel renormalization | none |
| L4u | Procrustes alignment of the top-16 PCA loadings, rotation within the dominant subspace, identity elsewhere | none |
| L4s | Same, with the rotation estimated from stable channels only (iterative, keep 60%) | none |
| L3 | Per-channel gain + offset (96 + 4 parameters) | supervised |
| L4 | Supervised rotation within the dominant subspace (16 × 16) | supervised |
| L5 | Full input remap (I + D)z + h | supervised |
| L6p | Ridge refit shrunk toward the old decoder (weights and intercept) | supervised |
| L6 | Ridge refit from scratch | supervised |

- **Fitting the supervised rungs.** They use day j's first n labelled trials. Regularization is chosen by k-fold CV over contiguous trial blocks
  (5 folds; 3 for L5). Closed-form rungs use one eigendecomposition per fold. L5 uses L-BFGS.
- **L5 is not a structural test.** With an 8 × 4 readout of rank ≤ 32 < 96 channels, (I + D) can produce any decoder of the same form,
  so L5 is reported only as an alternative full recalibration.
- **Structural questions** use genuinely constrained corrections instead:
  - L3 and L4;
  - re-learning only k channels' weights [`drift_dof.py`, `targeted_recal.py`].

## Statistics
Pairs that share a training session are not independent. Medians and paired differences therefore get 95% CIs from a cluster bootstrap over training
sessions (2,000 resamples) [`ibci/stats.py`]. Split-half reliability of time-residuals uses alternating *pairs* of
trials, because CO trials alternate between outward and centre targets.

## Electrode failure process
- **Per-channel measures** [`channel_health.py`, `failure_stats_xdata.py`]:
  - TC rate per session;
  - the mean and SD of SBP;
  - impedance;
  - tuning strength, i.e. 5-fold contiguous CV R² of 100 ms-smoothed SBP predicted from kinematics at leads of 0–180 ms.
- **Alive/silent states.**
  - A channel is "active" when its TC rate exceeds 2 Hz.
  - Silencing means 3 or more consecutive sessions below threshold. Revival means 3 or more consecutive sessions above 4 Hz.
  - Two-state Markov switching rates are fitted by maximum likelihood to all within-channel session pairs ≤ 400 days apart,
    using channels active in at least 3 sessions.
- **Session-level filtering.**
  - Session-wide events are separated by removing the median log-rate change across channels.
  - Sessions with session-wide rate inflation (median channel rate > 3× the median over sessions) are dropped.
- **Spatial analyses (LINK only).**
  - Grid distance is the Chebyshev distance.
  - Edge electrodes are those in row or column 0 or 7 of the physical 8 × 8 grid.
  - Edge vs interior is tested within each array (Mann–Whitney and permutation tests).

## Simulator and calibration
See `docs/SIMULATOR.md`.
- **Calibration** [`calibrate_sim.py`] matches the simulated oracle ladder (L2, L3, L5 at n = 300, as ratios to own-day) to the
  real one at gaps of 1, 7, 30, 120 and 480 days, on LINK sessions before day 700.
  - It uses the same correction code and the modal CV-selected regularization (λ_L3 = 1e-5, λ_L5 = 1e-3).
  - Optimizer: a 24-point Sobol search, then Nelder–Mead, with common random numbers.
- **Validation** [`validate_sim.py`] covers held-out time (days ≥ 700), rungs that were not fitted, and data budgets that were not fitted.

## Recalibration policies
Ridge-to-prior shrinkage is selected either per pair by CV, or from history: the grid value maximizing median R² over pairs from *other*
training sessions, leave-one-session-out [`recal_policy.py`, `targeted_recal.py`].
