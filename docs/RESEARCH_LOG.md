# Research log

Newest entries at the bottom. Dates are YYYY-MM-DD.

## 2026-10-05 — Project start, topic search

**Brief from owner:** find an invasive-BCI / intracortical-electrode topic that can be published in a Q1
journal and, above all, be *useful to others in future* (high reuse and citation). Any topic is acceptable.
Owner trusts the agent to choose, then carry out the full work in this repo.

**Constraints found:**
- Repo was empty at start.
- Hardware: RTX 3060 Ti 8 GB, 24 GB RAM, about 85 GB usable disk across G: and D: (see `CLAUDE.md`).
  This rules out training large neural foundation models from scratch and any topic that needs
  terabytes of raw 30 kHz broadband data.
- No wet lab and no patients, so the work must use public data or simulation.

**Research plan:** four parallel literature/data sweeps:
1. Field landscape 2024–2026 and clinical bottlenecks.
2. Inventory of public intracortical datasets (sizes, time spans, licences).
3. ML state of the art for iBCI decoding (foundation models, recalibration, speech, low-power, robustness).
4. Electrode degradation, on-implant compression, simulators, spike sorting.

Results will be saved under `docs/knowledge/`.

**Early findings:**
- DANDI API reachable. Exact sizes:
  - NLB: MC_Maze 000128 (0.69 GB), MC_RTT 000129 (0.05 GB), MC_Maze_Large/Medium/Small 000138/139/140
    (0.15/0.08/0.03 GB), Area2_Bump 000127 (1.82 GB).
  - 000070 "Neural population dynamics during reaching": 53.3 GB (too large for comfort).
  - **000688 "Long-term recordings of motor and premotor cortical spiking activity during reaching in
    monkeys": 13.2 GB, 111 assets.** A candidate for studying signal change over months.
- Stimulation side (ICMS, visual cortical prostheses): mostly psychophysics with little public data.
  Existing tool: `dynaphos`, a differentiable phosphene simulator (eLife 2024,
  https://elifesciences.org/articles/85812). Deprioritised because public data is limited.

## 2026-10-05 (cont.) — Sweeps 1, 3 and 4 done; novelty checks

Reports saved:
- `docs/knowledge/01_landscape_and_bottlenecks.md`
- `docs/knowledge/02_ml_state_of_the_art.md`
- `docs/knowledge/03_electrodes_degradation_compression.md`

**The sweeps agree:** the main unsolved clinical problem that can be attacked computationally is *chronic reliability*.
Signals drift, electrodes degrade, and decoders need recalibration, while nobody can tell when a decoder has
silently stopped working. Two of the three sweeps independently found **no work on calibrated uncertainty or
label-free failure prediction for intracortical decoders.**

**Saturated or crowded (avoid):**
- Chasing Brain-to-Text word error rate.
- Pretraining new foundation models (also impossible on 8 GB).
- Spiking-network leaderboards on Indy/Loco.
- New cross-session alignment methods, unless there is a new angle.

**Novelty checks I ran myself:**
- Europe PMC abstract queries (2026-10-05):
  - intracortical + BCI + uncertainty: 0 hits.
  - conformal + BCI/neural decoding: 1 hit, "From confidence to caution: conformal gating for cross-session
    motor imagery BCIs" (J Neural Eng 2026). That is *EEG classification*, so it is a close cousin but not the same problem.
  - intracortical + "performance prediction/estimation": 0 hits.
- arXiv: downloaded 603 unique BCI and neural-decoding abstracts and grepped them for
  uncertainty, conformal, failure, reliability, out-of-distribution and monitoring terms. Closest hits:
  - UnSPC, arXiv 2607.24031: uses uncertainty *internally* to choose pseudo-labels for drift adaptation.
    It does not monitor or predict failure.
  - "Slow-Fast BCI", arXiv 2609.01767: a perspective that explicitly *calls for* uncertainty-paced BCI assistance.
- bioRxiv API:
  - "Probabilistic Co-Control" (Huang…Gilja, bioRxiv 2026.04.02.715749): CTC speech decoders are
    overconfident within a session. No cross-day drift, no motor decoding, no failure prediction.
  - Gontier…Collinger (bioRxiv 2026.02.25.707999): detects neural *error signals* during cursor
    control. A different mechanism, and not about the decoder's own reliability.
- LINK repo (github.com/chesteklab/LINK_dataset): the paper's analyses are signal quality, tuning drift,
  dimensionality, cross-day decoding, stability and continual learning. **No uncertainty and no label-free monitoring.**

**Key data finding: LINK (DANDI 001201, CC-BY-4.0, 12.56 GB, 312 NWB sessions, 2020-01-27 to 2023-06-22).**
- 96 channels of threshold crossings and spike-band power in 20 ms bins, finger position and velocity
  (2 degrees of freedom), and 375 trials per session.
- The electrode table has a **per-session impedance column (`imp`)** that changes over time. The degradation
  sweep had believed no public long-term dataset included impedance.
- Quick check, first vs last session:
  - Median impedance 426.5 → 161.0 kΩ.
  - Median spike-band power 3.31 → 2.01.
  - Channels firing below 1 Hz: 49 → 67.
  - This matches the human BrainGate trends in Hahn et al. 2025.
- Full download started to `D:/ibci-data/link` via `scripts/download_link.py`.

**Environment:** created `.venv` (with `--system-site-packages`, so the CUDA torch is reused) because
`C:\Python311\Scripts` is not writable. Installed pynwb 4.2.0 and dandi 0.81.0. Pip cache is at `G:\pip-cache`.
Datasets live under `D:/ibci-data` (env var `IBCI_DATA`).

## 2026-10-05 (cont.) — Dataset sweep done; TOPIC FIXED

- Sweep 2 (datasets) saved to `docs/knowledge/04_public_datasets.md`.
  - Biggest find: the **BrainGate 20-year release** (Dryad dryad.x0k6djj1h, 2026-09-09, CC0): 14 humans,
    729 closed-loop cursor sessions, up to 7.6 years each, 84.7 GB split per participant. Not yet benchmarked by anyone.
  - Dryad file downloads are behind bot protection (the API returns 401 without a token; the web endpoint serves a JS
    challenge), so the owner must download these files manually in a browser. File IDs and sizes are listed in the decision record.
- Other long-term data:
  - DANDI 000688: Chewie 68 sessions over about 3 years; Mihili 28 sessions over about 1.5 years.
  - FALCON H2: T5 handwriting, about 17 months.
  - T15 Brain-to-Text '25: 20 months.
- Final novelty checks:
  - No intracortical "when to recalibrate" work found.
  - Closest classic is Perge et al. 2013 J Neural Eng (about 190 citations), which reports intra-day instabilities
    but no label-free prediction.
- Citation precedents (OpenAlex): Sussillo 2016 has 235, Degenhart 2020 has 237, Gallego 2020 has 452.

**DECISION (see `docs/decisions/0001-research-topic.md`):** *Knowing when an intracortical BCI decoder fails:
label-free reliability monitoring across years of recordings.*
- Primary data: LINK.
- Replication: 000688.
- Human: FALCON H2, plus BrainGate-20 if the owner downloads it.

**Next:** finish the LINK download, build the cache, and run a pilot. The pilot trains a ridge/Wiener decoder per day,
tests it on all later days, and checks how simple label-free shift metrics correlate with the R² drop.

## 2026-10-05 (cont.) — PILOT: the label-free monitoring hypothesis FAILS on natural drift

Scripts: `scripts/pilot_link_crossday.py` and `scripts/explore_monitor_features.py`. Data: the first 46–61 LINK
sessions (2020-01-27 to about 2021-03; the full download was still running). Decoder: ridge with 8 lags of
spike-band power, alpha 0.1, trained on the first 300 trials and tested on the rest. This follows the LINK paper.

**Baseline sanity check:** within-day R² median is 0.30 (correlation of about 0.55). The LINK tutorial reports per-output
Pearson r of 0.42–0.64, so our numbers are consistent.

**Fixed decoder** (day-i normalization reused on later days): cross-day R² median is **−1.29**, i.e. immediate
collapse. Baseline shifts dominate, as the LINK paper also reports.

**Renormalized decoder** (day-j z-scoring from day j's own *unlabelled* first 300 trials; the realistic deployment):

| Days since training | 0 | 1 | 7 | 21 | 45–60 | 90–120 | 120–180 | >180 |
|---|---|---|---|---|---|---|---|---|
| Median R² | 0.30 | 0.21 | 0.20 | 0.14 | 0.11 | 0.05 | 0.01 | −0.05 |

- R² drops about 30% overnight.
- Elapsed days predicts R² strongly: Spearman ρ = −0.75 across all pairs and −0.56 within 60 days.

**14 label-free features tested.** The pair-level shift and model features, and the hardware features:
- input mean shift and standard-deviation shift;
- principal-angle subspace shift;
- bootstrap-ensemble disagreement;
- disagreement between decoders built on different views (SBP vs threshold crossings, 8-lag vs 1-lag);
- predicted-output statistics (velocity bias, variance ratio, Wasserstein distance to the training kinematics, smoothness);
- position/velocity self-consistency;
- Procrustes alignment disagreement, Procrustes residual, readout off-subspace energy;
- neural modulation;
- impedance log-ratio and dead-channel count.

**Results:**
- Partial Spearman with R² given log(days) is at most about 0.27 (output variance) and mostly under 0.1.
- In time-blocked cross-validated prediction of R² (283 pairs, at most 60 days apart):
  - days only: MAE 0.064;
  - label-free only: MAE 0.067–0.071;
  - days plus label-free: MAE 0.067–0.070.
  - **Label-free features do not beat elapsed time.**
- Predicting the *recalibration gain* (own-day R² minus cross-day R²) instead: days only MAE 0.045, label-free 0.054.
- Detecting "unexpectedly bad days" (residual below −1 SD from the time curve): AUROC 0.50–0.58 for every feature.

**Is the leftover variation just noise?** No.
- Split-half reliability of the time-residuals, using alternating *pairs* of trials:
  r = 0.72, so Spearman–Brown reliability is **0.84**.
- (Odd/even single-trial splits gave a negative correlation. That is an artifact: center-out trials alternate
  between outward and centre targets.)
- So day-to-day performance variation beyond the time trend is *real and reliable*, but **invisible to every
  label-free signal tried**.
- Test-day difficulty (own-day R²) raises explained R² variance from 0.29 (days alone) to 0.56.

**Conclusion:** the central claim of decision 0001 (label-free signals predict decoder failure beyond elapsed time)
is not supported for natural drift. The pilot did its job before months of work were sunk. The negative result is
worth keeping, for example as a section or short note: "under gradual drift, a calendar-based schedule is a strong
baseline; label-free statistics do not capture the reliable residual." Abrupt *channel-level* faults remain detectable
by construction. **Decision 0001 will be superseded; see decision 0002.**

## 2026-10-05/06 (night) — Decision 0002 executed: first anatomy, locality, channel-health and simulator code

The owner went to sleep and authorised full autonomy. Commits must be authored by Imtiaj Sajin with no AI
co-author trailer; see `CLAUDE.md`.

**Housekeeping (owner-approved):**
- Stopped EA Desktop, NVIDIA Broadcast and qBittorrent. qBittorrent was saturating the uplink and throttling
  downloads: DANDI throughput went from about 0.03–0.18 to about 2.2–2.7 MB/s. *The owner should restart qBittorrent if needed.*
- EABackgroundService needs admin rights, so it was left running.

**Code added:**
- `src/ibci/linear.py`: lag decoders, ridge, ridge-to-prior, CV alpha.
- `src/ibci/anatomy.py`: oracle-ladder rungs.
- `scripts/drift_anatomy.py`: ladder over session pairs at target gaps, and data-efficiency curves.
- `scripts/remap_locality.py`: are learned remaps spatially local?
- `scripts/channel_health.py`: per-channel 3.5-year trajectories and the failure process.
- `src/ibci/sim.py`: drift/failure simulator (mixing, turnover, death/revival, gain jumps, raw offset/scale walks;
  instant plus slow components).
- `scripts/calibrate_sim.py`: Nelder–Mead fit of the simulator so its *ladder* matches the real one; supports time splits.
- `scripts/sim_augment.py`: "train on simulated futures" against regularization-only, ad hoc perturbation and multi-day controls.
- `src/ibci/data/perich.py`: DANDI 000688 loader (spikes summed per electrode, 20 ms bins, cursor position and velocity).
- `scripts/download_dandi.py`: generic DANDI downloader.
- `src/ibci/plotting.py` and `scripts/make_figures.py`: validated palette, figure generation.

**Interim anatomy** (`results/anatomy_partial`; 12 training sessions, 47 pairs, data up to about 2021-03).
Median R² at n = 300 labelled trials:

| gap (days) | L0 fixed | L2 renorm | L3 gains | L5 full remap | L6p ridge-to-prior | own |
|---|---|---|---|---|---|---|
| 1 | −0.62 | 0.24 | 0.26 | 0.30 | 0.33 | 0.33 |
| 14 | −2.94 | 0.19 | 0.22 | 0.28 | 0.30 | 0.29 |
| 120 | −1.13 | 0.06 | 0.12 | 0.19 | 0.28 | 0.28 |
| 480 | −2.46 | −0.15 | 0.04 | 0.12 | 0.26 | 0.25 |

What this shows:
- Per-channel gain changes explain little of the drift loss (about 20–40% of the loss beyond renorm).
- A full linear input remap in front of the frozen decoder recovers about 60–75%.
- At long gaps, a large share needs a *changed decoder*.
- With 300 trials, ridge shrunk to the previous decoder matches the own-day decoder.
- The latent-rotation rung was broken (projection loss), so it was reformulated as a rotation inside the dominant
  subspace with identity elsewhere. Not re-run yet.

**Interim locality** (`results/locality_partial`; 19 pairs). Remaps restricted to grid neighbours vs parameter-matched
random *distant* channels (local1 vs far1, 704 parameters):

| gap (days) | local1 | far1 |
|---|---|---|
| 1 | 0.224 | 0.222 |
| 7 | 0.230 | 0.218 |
| 30 | 0.186 | 0.167 |
| 120 | 0.176 | 0.144 |
| 480 | 0.008 | −0.026 |

So locality has a weak advantage, and parameter count dominates. Needs the full data and paired tests before any claim.

**Interim channel health** (`results/channel_health_partial`; first 470 days, 77 sessions):
- Of 40 initially active channels (>2 Hz), 25 "died" (3 or more consecutive sessions below 2 Hz). Median death day was 151, and 8 died then revived.
- Session-to-session log threshold-crossing changes are heavy-tailed (excess kurtosis 8.6; 11% of changes have |Δlog| > 1), so abrupt events are common.
- Neighbouring electrodes have more similar decline slopes than distant pairs (Mann–Whitney p = 3e-10).
- Dead channels are not significantly clustered (permutation p = 0.21).
- No evidence that edge electrodes decline faster (p = 0.77).
- The medial array declines faster than the lateral one.
- Median impedance falls over time (Spearman −0.37), with a weak within-channel link to activity (median ρ = −0.15).
- Bug fixed: the per-channel tuning model must use *future* kinematics (motor cortex leads movement) and SBP smoothed
  over 100 ms. Before the fix, tuning R² was about 0 for almost all channels.

## 2026-10-06 ~01:15 — LINK fully downloaded (312/312); electrode failure results on all 3.4 years

**Ladder fixes before the full runs:**
1. The Procrustes rotation was applied in the wrong direction (Q must be Rᵀ). This made the latent rungs worse than nothing.
2. λ is now chosen by k-fold CV over contiguous trial blocks, using an eigendecomposition per fold. The selected λ values are logged.
3. Grids were widened; λ was hitting the grid edge.
4. Intercepts and offsets are now **shrunk toward the previous decoder**. Previously a free intercept fitted on only
   10–20 trials could push recalibration below "do nothing."
5. Added L4s: a Degenhart-style stable-channel Procrustes alignment, without labels.

Remap L-BFGS converges in about 0.1 s regardless of the iteration count (15–120 iterations give identical R²).

**Channel health** (`scripts/channel_health.py` → `results/channel_health`; 312 sessions over 1,242 days):

| Measure | Year 0 | Year 1 | Year 2 | Year 3 |
|---|---|---|---|---|
| Median impedance (kΩ) | 302 | 248 | 192 | 171 |
| Active channels (>2 Hz) | 30 | 23 | 18 | 18 |
| Tuned channels (CV R² > 0.05, 100 ms-smoothed SBP) | 12 | 8 | 8 | 7.5 |

- Impedance has Spearman ρ = −0.88 with time, the same direction as in BrainGate humans.
- 34 of 40 initially active channels went silent (3 or more sessions below 2 Hz); **25 of those revived**.
- Spatial: neighbours have more similar decline slopes than distant pairs (p = 4e-25). Silent channels are not clustered (permutation p = 0.49).
- **Edge vs interior, within each array** (log threshold-crossing slope per year; `results/channel_health/channel_slopes.csv`):

| Array | Edge | Interior | Mann–Whitney p |
|---|---|---|---|
| Medial | −0.27 (n = 28) | −0.18 (n = 36) | 0.046 (permutation of the median: 0.12) |
| Lateral | −0.12 (n = 14) | +0.04 (n = 18) | 0.0001 |

  - Medial array, distance from centre vs slope: ρ = −0.24 (p = 0.057).
  - **Spike-band-power slopes show no edge effect** (p = 0.66 and 0.37).
  - Interpretation: isolated spiking is lost faster at array edges, consistent with micromotion strain (Forrest
    2025), while broadband power declines uniformly. One animal, so this is suggestive.

**Failure process** (`scripts/failure_stats_xdata.py` → `results/failure_xdata`, LINK):
- Two-state alive/silent Markov rates: h_off = 0.0031/day and h_on = 0.0008/day. These now set the simulator's failure
  parameters, independently of the ladder calibration.
- Silencing rate: 0.66 per channel-year. Active channels went from 36 to 18 (−4.9 per year).
- Abruptness for sessions ≤ 7 days apart: raw excess kurtosis 7.2 (15.6% of changes have |Δlog| > 1).
- After removing **session-wide common-mode events** (10 of 274 short-gap session pairs, where all channels jump
  together, probably threshold or noise events), channel-level kurtosis is 10.2 (14.0% have |Δ| > 1). Abrupt
  *single-electrode* changes are real.
- Figure: `results/figures/fig_health.png`. Session-wide events are marked, and impedance medians below 50 kΩ (failed
  measurements) are removed.

**Running:** full anatomy (537 pairs, n = 300), data efficiency (86 pairs, n = 10–300), and the 000688 download.

## 2026-10-06 ~02:45 — LSTM ladder, human (T5) failure statistics, 000688 and H2 downloaded

**LSTM oracle ladder** (`scripts/drift_anatomy_nn.py` → `results/anatomy_nn`; LINK; 25 training sessions, 79 pairs,
LSTM with hidden size 256 and a 20-bin window). Median R²:

| gap (days) | L2 renorm | L3 gains | L5 remap | L6p fine-tune to prior | own |
|---|---|---|---|---|---|
| 1 | 0.21 | 0.29 | 0.30 | 0.34 | 0.34 |
| 30 | 0.18 | 0.28 | 0.32 | 0.40 | 0.42 |
| 120 | 0.09 | 0.27 | 0.35 | 0.41 | 0.44 |
| 480 | −0.05 | 0.17 | 0.28 | 0.35 | 0.42 |

- Same anatomy as ridge: renormalization is insufficient, the remap recovers most of the loss, and fine-tuning toward the prior approaches own-day.
- **Per-channel gains recover much more for the LSTM than for ridge**, consistent with network nonlinearities being
  sensitive to input scale.
- The LSTM is more accurate within a day (0.34–0.44 vs 0.29–0.31 for ridge).
- Relative retention on *matched* pairs: only 5 overlapping pairs so far, and they look similar to ridge. **"LSTM is more drift-robust" is NOT
  established.** A matched ridge run on the exact LSTM pairs is queued (`drift_anatomy.py --pairs-from`).

**FALCON H2, human T5** (`results/failure_xdata_h2`):
- The 2023 sessions show session-wide rate inflation, up to a median of 115 Hz per channel on 2023-10-09. That is a
  threshold or preprocessing change, so a dataset-agnostic rule now drops sessions whose median channel rate exceeds 3× the median over sessions.
- The held-out calibration files are about 80 s snippets, so only the 20 full held-in sessions are used (2022-05 to 2022-12, 211 days).
- Results:
  - Active channels: 97 → 93.
  - 21 of 99 initially active channels went silent, 3 revived (0.43 per channel-year).
  - Switching rates: h_off = 0.0099/day, h_on = 0.028/day, i.e. faster turnover than monkey N (the T5 array was about 6 years old).
  - Abruptness: excess kurtosis 2.6, 24% of changes have |Δlog| > 1.
- **Silencing and revival replicate in a human. The span is short, so this is modest evidence.**

**Time-split failure fit** (LINK days < 700; `results/failure_xdata_cal700`): h_off = 0.0038/day, h_on = 0.0018/day.
The simulator calibration (`results/sim_calib_split`) uses only these rates and pairs inside days 0–700.

## 2026-10-06 ~03:50 — FULL drift anatomy (LINK, 537 pairs, 80 training sessions, gaps 1–900 days)

- An out-of-memory crash at pair 351 (too many concurrent jobs holding all sessions in RAM) was fixed by:
  - memory-lean `Sess` objects (only z-scored copies stored);
  - an LRU `SessCache` that loads sessions on demand;
  - `--resume`.
- Results: `results/anatomy/ladder.csv`, `results/anatomy/table_ci.csv` (cluster-bootstrap 95% CIs over training
  sessions), and `results/figures/fig_anatomy.png`.

**Median R²:**

| gap (days) | 1 | 7 | 30 | 120 | 480 | 900 |
|---|---|---|---|---|---|---|
| own-day | 0.29 | 0.29 | 0.28 | 0.29 | 0.29 | 0.29 |
| L2 renorm (no labels) | 0.20 | 0.15 | 0.05 | 0.01 | −0.10 | −0.14 |
| L3 + channel gains (96 parameters) | 0.22 | 0.17 | 0.11 | 0.08 | 0.06 | 0.07 |
| L5 + full input remap | 0.27 | 0.25 | 0.22 | 0.22 | 0.23 | 0.22 |
| L6p ridge-to-prior (300 trials) | 0.31 | 0.30 | 0.29 | 0.29 | 0.29 | 0.29 |

**Retention and recovery** (median [95% CI]):

| | 1 day | 900 days |
|---|---|---|
| L2/own (renorm retention) | 0.73 [0.62, 0.78] | −0.52 |
| L5/own (remap retention) | 0.93 | 0.80 [0.76, 0.83] |

- Share of the drift loss (own − L2) recovered by the full remap: **0.71–0.86 at every gap**.
- Share recovered by per-channel gains: 0.14–0.50.
- L4u (label-free Procrustes) never beats renormalization.
- L4 (supervised rank-16 rotation) sits between L3 and L5, so the remix is not confined to the dominant subspace.

**Headline:** with the decoder frozen, a linear remix of the input channels restores about 80% of own-day performance
whether the decoder is 4 days or 2.5 years old. Chronic drift on this array is predominantly *channel-level linear
re-mixing* onto a decoder-relevant code that stays stable for years. This is the decoder-level counterpart of
"stable latent dynamics" (Gallego 2020), shown here over 900 days. Per-channel gain changes are a minor part, and the
label-free alignments tested do not recover the remix.

## 2026-10-06 ~04:00 — Small-budget recalibration policy; matched LSTM vs ridge

**Recalibration policy** (`scripts/recal_policy.py` → `results/recal_policy`; 86 data-efficiency pairs). Ridge-to-prior
shrinks both weights and intercept toward the previous decoder. The shrinkage α is chosen per pair by CV, *or* from
history (the α maximizing median R² over pairs from other training sessions; leave-one-session-out, so no leakage).

Median R²:

| n trials | renorm | CV | history | history by gap | oracle |
|---|---|---|---|---|---|
| 10 | 0.052 | 0.089 | 0.111 | 0.103 | 0.114 |
| 20 | 0.052 | 0.129 | 0.133 | 0.136 | 0.142 |
| 50 | 0.052 | 0.184 | 0.183 | 0.187 | 0.197 |

Share of pairs where recalibration is **worse than doing nothing** (by more than 0.01):

| n trials | CV | history |
|---|---|---|
| 10 | 10.5% | 1.2% |
| 20 | 8.1% | 0% |

At 1 day with 10 trials, CV gives 0.103, history gives 0.193, and renorm gives 0.190.

**Practical recipe:** borrow the shrinkage strength from past sessions; harmful recalibrations essentially disappear.

**Matched LSTM vs ridge** (same 79 pairs; `results/anatomy_nn/matched_vs_ridge.csv`; cluster-bootstrap CIs):
- Own-day: the LSTM is higher by 0.128 [0.111, 0.141] (higher on 100% of training sessions).
- Renormalized, absolute: the LSTM is higher by 0.056 [0.043, 0.073].
- Relative retention (L2/own): the LSTM is higher by 0.072 [0.010, 0.114], so it is *modestly* more drift-robust.
- The anatomy differs in detail:
  - Per-channel gains recover much more for the LSTM (L3/own 0.74 vs 0.45 at 30 days).
  - The full input remap recovers somewhat less (L5/own lower by 0.051 [−0.079, −0.035]; 0.67 vs 0.83 at 480 days).
  - Both point to input-scale sensitivity of the nonlinear network, and to a harder optimization of a remap in front of it.
