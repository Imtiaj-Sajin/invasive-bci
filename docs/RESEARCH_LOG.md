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
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
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

**Interim anatomy** (`results/archive/anatomy_partial`; 12 training sessions, 47 pairs, data up to about 2021-03).
Median R² at n = 300 labelled trials:

| gap (days) | L0 fixed | L2 renorm | L3 gains | L5 full remap | L6p ridge-to-prior | own |
| --- | --- | --- | --- | --- | --- | --- |
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

**Interim locality** (`results/archive/locality_partial`; 19 pairs). Remaps restricted to grid neighbours vs parameter-matched
random *distant* channels (local1 vs far1, 704 parameters):

| gap (days) | local1 | far1 |
| --- | --- | --- |
| 1 | 0.224 | 0.222 |
| 7 | 0.230 | 0.218 |
| 30 | 0.186 | 0.167 |
| 120 | 0.176 | 0.144 |
| 480 | 0.008 | −0.026 |

So locality has a weak advantage, and parameter count dominates. Needs the full data and paired tests before any claim.

**Interim channel health** (`results/archive/channel_health_partial`; first 470 days, 77 sessions):

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
| --- | --- | --- | --- | --- |
| Median impedance (kΩ) | 302 | 248 | 192 | 171 |
| Active channels (>2 Hz) | 30 | 23 | 18 | 18 |
| Tuned channels (CV R² > 0.05, 100 ms-smoothed SBP) | 12 | 8 | 8 | 7.5 |

- Impedance has Spearman ρ = −0.88 with time, the same direction as in BrainGate humans.
- 34 of 40 initially active channels went silent (3 or more sessions below 2 Hz); **25 of those revived**.
- Spatial: neighbours have more similar decline slopes than distant pairs (p = 4e-25). Silent channels are not clustered (permutation p = 0.49).
- **Edge vs interior, within each array** (log threshold-crossing slope per year; `results/channel_health/channel_slopes.csv`):

| Array | Edge | Interior | Mann–Whitney p |
| --- | --- | --- | --- |
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
| --- | --- | --- | --- | --- | --- |
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
| --- | --- | --- | --- | --- | --- | --- |
| own-day | 0.29 | 0.29 | 0.28 | 0.29 | 0.29 | 0.29 |
| L2 renorm (no labels) | 0.20 | 0.15 | 0.05 | 0.01 | −0.10 | −0.14 |
| L3 + channel gains (96 parameters) | 0.22 | 0.17 | 0.11 | 0.08 | 0.06 | 0.07 |
| L5 + full input remap | 0.27 | 0.25 | 0.22 | 0.22 | 0.23 | 0.22 |
| L6p ridge-to-prior (300 trials) | 0.31 | 0.30 | 0.29 | 0.29 | 0.29 | 0.29 |

**Retention and recovery** (median [95% CI]):

| | 1 day | 900 days |
| --- | --- | --- |
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
| --- | --- | --- | --- | --- | --- |
| 10 | 0.052 | 0.089 | 0.111 | 0.103 | 0.114 |
| 20 | 0.052 | 0.129 | 0.133 | 0.136 | 0.142 |
| 50 | 0.052 | 0.184 | 0.183 | 0.187 | 0.197 |

Share of pairs where recalibration is **worse than doing nothing** (by more than 0.01):

| n trials | CV | history |
| --- | --- | --- |
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

## 2026-10-06 ~04:50 — Simulator calibration: two bugs found and fixed (lessons)

1. **Silencing applied to every channel.** The alive/silent process must act only on channels that are *alive in the base session* (threshold-crossing rate > 2 Hz).
   Most channels are already silent on any given day, so applying it to all of them double-degraded the signal.
   - Fix: `Sess.alive`; `sample_drift(..., alive=)`.
2. **Regularization mismatch between the real and simulated ladders.** On real pairs, CV mostly selects λ_L3 = 1e-5
   and λ_L5 = 1e-3 (or 1e-4). The calibration used fixed λ_L3 = 1e-2 and λ_L5 = 1e-1, i.e. 100–1000× stronger.
   - That handicapped the simulated corrections. The fit (loss 0.663, `results/archive/sim_calib_split_badlam`) could not reproduce
     the real signature: remap recovery stays high while renormalized performance collapses. The optimizer distorted the parameters to
     compensate (high turnover).
   - Fix: the calibration now uses the modal real λ.
   - Probe: with real λ, a mixing-dominated simulator gives, at 480 days, L2/own −0.2 to −0.37 and L5/own 0.58–0.70,
     against real values of −0.32 and 0.78.

**Lesson for the paper:** the simulator must be scored with *exactly* the same correction procedure as the real data
(same regularization, same CV), or calibration silently compensates for pipeline differences.

**Optimizer:** coarse Sobol search (24–32 points) followed by Nelder–Mead with a wide initial simplex. Default
Nelder–Mead with about 5% simplex steps barely moved (loss stuck at about 2.4 after 24 evaluations).

## 2026-10-06 ~05:00 — CORRECTION: the "full input remap" rung (L5) is retraining in disguise

**Retraction of the 03:50 headline.** The L5 rung learns M (96×96) in front of the frozen decoder, so the effective
weights become W'_l = Mᵀ W_l. The 8 lags × 4 outputs give a 96×32 readout matrix A. If rank(A) = 32 ≤ 96, then Mᵀ A can equal
*any* 96×32 target. **L5 can therefore represent any new decoder of the same form.** It is full recalibration with a different
regularizer (shrink toward the identity remap), not a constrained "channel re-mixing" model.

- Its about 80% recovery reflects that regularizer and finite data, *not* evidence that drift is linear channel re-mixing.
- Chewie, where L5 ≈ own, is consistent with this.
- The 03:50 claim "drift is predominantly channel-level linear re-mixing" is **withdrawn**.

**What stands:**

- The renormalization decay curve.
- Label-free Procrustes alignments don't help.
- L3 per-channel gains (96 parameters, genuinely constrained) recover 14–50% of the loss.
- L4 rank-16 rotation (256 parameters, constrained) recovers more than L3.
- Data efficiency, the recalibration policy, and the LSTM-vs-ridge comparison.
- The simulator calibration targets the L5 *statistic*, which is still well defined, but it must not be read as "mixing only."

**New, properly constrained anatomy** (next):

1. Rank-r input remaps M = I + U Vᵀ with r = 1, 2, 4, 8. These confine the decoder change to an r-dimensional channel subspace.
2. Re-learning only k channels' weights (k = 4–96), with the others frozen. Channels are chosen by:
   - supervised weight change (an upper bound);
   - a *label-free* change in each channel's correlation profile;
   - random choice (a control).

The question becomes: **how many degrees of freedom (subspace dimensions, or electrodes) does decoder-relevant drift occupy?**

## 2026-10-06 ~05:20 — Replication in two more monkeys (000688), and constrained drift DOF

**Replication** (`scripts/replicate_perich.py` → `results/replication/{C,M}_ladder.csv`). Another lab, a reaching task,
sorted units summed per electrode (192 channels), cursor kinematics. Chewie: 145 pairs; Mihili: 44 pairs.

| gap | ret L2 (N / C / M) | ret L3 gains (N / C / M) |
| --- | --- | --- |
| 1 d | 0.73 / 0.72 / 0.77 | 0.78 / 0.81 / 0.84 |
| 7 d | 0.54 / 0.56 / 0.61 | 0.63 / 0.72 / 0.70 |
| 30 d | 0.19 / 0.25 / 0.55 | 0.43 / 0.51 / 0.69 |
| 120 d | 0.04 / −0.09 / 0.04 | 0.34 / 0.33 / 0.35 |
| 480 d | −0.36 / −0.27 / 0.02 | 0.22 / 0.11 / 0.20 |

(ret = median rung R² divided by own-day R²)

- L4u (label-free Procrustes) is ≤ L2 in all three monkeys.
- **The drift timescale is broadly conserved across 3 monkeys, 2 labs, 2 tasks and 2 feature types:** about 25% of
  performance lost overnight, about half by one week, and near zero by about 4 months. Per-channel gains recover a similar,
  modest share everywhere.

**Constrained drift degrees of freedom** (`scripts/drift_dof.py` → `results/drift_dof`; LINK; 150 pairs, 30 per gap).
Share of the drift loss recovered, relative to full ridge-to-prior with 300 trials:

| channels re-learned (k of 96) | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- |
| supervised selection (largest weight change) | 0.56 | 0.74 | 0.90 | 0.98 |
| **label-free** (mean shift + log-sd ratio) | 0.31 | 0.46 | 0.68 | 0.86 |
| label-free (correlation-profile change) | 0.17 | 0.29 | 0.40 | 0.78 |
| random | 0.18 | 0.28 | 0.45 | 0.75 |

- The pattern is consistent across gaps (1–480 days).
- Decoder-relevant drift is concentrated on a minority of electrodes, and simple label-free per-channel statistics
  identify part of them (0.46 vs 0.28 for random at k = 16).
- **Caveat being tested:** supervised selection may reflect channel *importance*. An `importance` control (old-decoder
  weight norm) and a decoder-weighted label-free score (`wstats` = change × importance) are added in `scripts/targeted_recal.py`.
- **Rank-r input remaps are not informative:** recovery plateaus at rank 2 (0.64) because the ridge readout itself is
  effectively rank 2 (top-2 singular values carry 94–95% of the energy). Dropped as a drift measure.

## 2026-10-06 ~05:35 — Targeted recalibration; CORRECTION to the "concentrated drift" reading

`scripts/targeted_recal.py` → `results/targeted_recal`. 86 pairs. Shrinkage chosen from history (leave-one-session-out).
All refits come from one Gram matrix per labelled set, verified identical to a direct solve (max difference 1.5e-7).

Median R² by labelled trials n:

| n | renorm | full refit | stats k=32 | importance k=32 | wstats k=32 | random k=32 |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.052 | 0.111 | 0.085 | **0.124** | 0.113 | 0.090 |
| 20 | 0.052 | 0.133 | 0.127 | **0.154** | 0.149 | 0.101 |
| 50 | 0.052 | 0.183 | 0.136 | **0.199** | 0.196 | 0.121 |
| 100 | 0.052 | 0.227 | 0.182 | 0.226 | 0.214 | 0.139 |

- At n = 100, re-learning the old decoder's 16 most-used channels already recovers about 76% of the loss, i.e.
  (0.185 − 0.052) / (0.227 − 0.052). That is as much as the *supervised* top-16 selection in `drift_dof` (74%).
- Label-free change statistics are worse than importance, and "change × importance" is no better than importance.

**Correction:** the 05:20 reading "decoder-relevant drift is concentrated on a minority of electrodes, partly
identifiable without labels" is **not supported** beyond the decoder's own reliance on a subset of channels. The k-channel
recovery mostly reflects *decoder importance*, not where drift happens.

**What survives (practical):** with 10–50 labelled trials, re-learning only the 32 channels the old decoder relies on
most is slightly better than a full refit (+0.013 to +0.021 median R²), because fewer parameters are fitted. At n ≥ 100 the
full refit is as good.

## 2026-10-06 ~06:40 — Simulator calibrated (time split: LINK days < 700)

`results/sim_calib_split/calibration.json`. 24-point Sobol search followed by Nelder–Mead; real-matched λ; alive-only
silencing; failure rates from days < 700 (h_off = 0.0038/day, h_on = 0.0018/day). **Final loss 0.074**, against 0.663 for
the mis-regularized run.

Fit on a fresh random seed (ratio to own-day, L2 / L3 / L5):

| gap | real | simulated |
| --- | --- | --- |
| 1 d | 0.62 / 0.72 / 0.91 | 0.61 / 0.73 / 0.92 |
| 7 d | 0.50 / 0.62 / 0.87 | 0.45 / 0.71 / 0.88 |
| 30 d | 0.02 / 0.30 / 0.83 | 0.28 / 0.40 / 0.82 |
| 120 d | −0.12 / 0.24 / 0.79 | −0.07 / 0.36 / 0.75 |
| 480 d | −0.32 / 0.23 / 0.78 | −0.35 / 0.29 / 0.73 |

The worst point is L2 at 30 d: the simulator decays too slowly around one month.

**Fitted parameters:**

- Instant (session-to-session) mixing s_mix0 = 0.96.
- Slow mixing s_mix = 1.48 with τ_mix = 15 days.
- Turnover rho: 0.04 instant, rising to 0.12 with τ = 360 days.

Reading: most simulated drift is a large session-to-session component plus a fast (about 2-week) accumulating component,
with little slow turnover. (The simulated mechanisms reproduce the measured statistics. Given the L5 caveat, this is a
generative fit, not proof of the mechanism.)

**Running now:**

- `validate_sim.py`: held-out days ≥ 700, untargeted rungs and n = 20/50/100.
- `sim_augment.py`: train on simulated futures; evaluated on days ≥ 700 only.

## 2026-10-06 ~06:50 — "Train on simulated futures" (negative vs a regularization control); regularization trade-off

**Augmentation** (`scripts/sim_augment.py` → `results/augment`). Simulator calibrated on days < 700; 25 training
sessions after day 700; 77 pairs; daily renormalization. Paired gain over base ridge (α = 0.1), cluster-bootstrap 95% CI:

| Method | Gain |
| --- | --- |
| Ad hoc perturbations (Sussillo-style: channel dropout plus gain noise) | +0.000 [−0.003, +0.002] |
| Simulated-futures augmentation | +0.010 [+0.005, +0.013]; +0.012 at gaps ≥ 30 d; beats perturbations by +0.005 [0.001, 0.019] |
| **Ridge with α = 10⁴ (chosen on the calibration period)** | **+0.042 [+0.033, +0.050]**, positive for 100% of sessions |
| Simulated futures + α = 10⁴ | +0.015; worse than α = 10⁴ alone by −0.024 [−0.030, −0.019] |
| Multi-day real training (3 previous sessions; more labels) | +0.056 [+0.041, +0.073] |

**Conclusion:** for linear decoders, simulator augmentation beats ad hoc perturbations but **not** plain stronger
regularization. The simulator's value is in benchmarking and policy evaluation, not as augmentation for ridge. Network
decoders are untested.

**Regularization trade-off** (`scripts/reg_tradeoff.py` → `results/reg_tradeoff`, `results/figures/fig_reg_tradeoff.png`;
40 training sessions). Median R² by α:

| gap | α = 0.1 (LINK default) | α = 1e3 | α = 1e4 | α = 3e4 | α = 1e5 |
| --- | --- | --- | --- | --- | --- |
| same day | 0.304 | 0.307 | **0.305** | 0.268 | 0.178 |
| 1 d | 0.267 | 0.272 | 0.275 | 0.242 | 0.166 |
| 7 d | 0.143 | 0.156 | 0.178 | 0.165 | 0.120 |
| 30 d | 0.098 | 0.112 | 0.145 | 0.149 | 0.106 |
| 120 d | 0.039 | 0.057 | 0.101 | 0.111 | 0.083 |
| 480 d | −0.097 | −0.080 | −0.002 | 0.040 | 0.046 |

- **Free lunch:** α = 10⁴ leaves same-day accuracy unchanged, but cross-day R² is about 50% higher at 30 d and 2.6× higher at 120 d.
- Same-day CV is flat between α = 0.1 and 10⁴, so it gives no reason to choose the larger value; hence small defaults.
- **Simple rule:** choose the largest α that keeps same-day R² within about 1%.
- Note: the anatomy ladder used α = 0.1 base decoders, so its absolute decay numbers describe that common default.

## 2026-10-06 ~07:40 — Simulator validation on held-out time; drift is slower later in the implant's life

**Validation** (`scripts/validate_sim.py` → `results/sim_validation`). 10 base sessions from days ≥ 700; full ladder on
simulated futures compared with the real ladder for pairs with training day ≥ 700.

Median |simulated − real| (ratio to own-day):

| | targeted rungs | untargeted rungs | data budgets n = 20/50/100 |
| --- | --- | --- | --- |
| calibrated simulator | 0.12 | 0.06 | 0.11 |
| no-drift reference | 0.37 | 0.27 | 0.39 |

- **The simulator is 3–4× closer to reality than a no-drift null.**
- It tracks untargeted rungs well. For example, ridge-to-prior L6p at n = 100: real 0.80–0.90 vs simulated 0.65–0.85.

**Systematic miss:**

- In the held-out later period, real decoders decay *more slowly* than in the calibration period.
- Real L2 retention is 0.80 at 1 d, 0.32 at 30 d, 0.17 at 120 d and −0.04 at 480 d.
- The simulator (fitted on days < 700) predicts 0.59, −0.16, −0.10 and −0.29.

**Direct check in the real data** (L2/own by implant age of the training session):

| gap | year 1 | year 2 | year 3 | year 3.5 |
| --- | --- | --- | --- | --- |
| 30 d | 0.38 | −0.02 | 0.48 | 0.15 |
| 120 d | −0.12 | −0.12 | 0.18 | 0.17 |

- Year 3 at 120 d: 0.18, 95% CI [0.03, 0.31]. Year 1: −0.12, CI [−0.87, 0.07].
- Not monotonic (year 2 is worst), but long-gap drift is slower in years 3+ than in years 1–2.
- **The drift process is itself non-stationary over the implant's life.**
- Next simulator version: make the drift rates depend on implant age (e.g. per-year calibration, or a parametric age term).

**α = 10⁴ base decoders in the ladder** (`results/anatomy_alpha1e4`; same 150 pairs as `drift_dof`).
L2/own for α = 0.1 vs α = 1e4:

| gap | 1 d | 7 d | 30 d | 120 d | 480 d |
| --- | --- | --- | --- | --- | --- |
| α = 0.1 | 0.71 | 0.52 | 0.18 | 0.05 | −0.39 |
| α = 1e4 | 0.79 | 0.62 | 0.41 | 0.29 | −0.01 |

This confirms the regularization finding inside the ladder.

**Waiting on the owner:** a browser download of the BrainGate yield files (Dryad zip of README plus 14 `yield_*.tar.gz`, 5.3 GB).

## 2026-10-06 ~08:00 — BrainGate 20-year human release: first 5 participants

- The owner downloaded the Dryad zip in a browser (Dryad blocks automated downloads).
- **The streamed zip was truncated.** It contains only README plus yield A1, S1, S2, S3 and T1 (392 MB) instead of 14 archives
  (5.3 GB). The owner is downloading the remaining 9 yield archives individually.
- Extracted to `D:/ibci-data/braingate/yield/`. The README text confirms the format: 10 ms bins; per-electrode rates at −3 to
  −5.5 RMS; impedance in kΩ (only in some sessions); grid x/y on a 10×10 array; "spiking" means at least 2 Hz at −4.5 RMS, the same definition as ours.
- `scripts/braingate_failure.py` → `results/braingate_failure` (368 sessions):

| Array | Span (d) | Yield first → last (%) | Silenced / revived | Impedance trend (ρ) | Edge vs interior decline p |
| --- | --- | --- | --- | --- | --- |
| A1 | 252 | 81 → 55 | 68 / 44 | −0.27 | 0.32 |
| S1 | 337 | 3 → 1 | 4 / 1 | — | 0.017 |
| S2 | 458 | 1 → 6 | 1 / 0 | — | 0.93 |
| S3 | 1,954 | 84 → 10 | 88 / 68 | −0.91 | 0.040 |
| T1 | 277 | 65 → 86 | 17 / 11 | −0.93 | 0.38 |

**Pooled so far:**

- Revived/silenced = 0.70 (LINK monkey N: 23/31 = 0.74).
- Median channel-level kurtosis 3.4.
- Edge declines faster: Fisher-combined p = 0.041, median edge − interior slope −0.10/yr.
- Impedance declines wherever it was measured.

**Preliminary:** S1 and S2 had almost no active electrodes. The remaining 9 participants (T2, T3, T5–T11) hold most of the long-term data.

## 2026-10-06 ~08:10 — BrainGate humans: 11 arrays (9 participants, 1,027 sessions)

The owner downloaded the remaining files as smaller zips; the large streamed zips stall at 30–40%. Zip 2 (T2, T3, T10,
T11) passed testzip and was extracted. Zip 3 (T5–T9) is still downloading; a watcher will verify and extract it.

`results/braingate_failure` (per array):

| Array | Span (d) | Silenced / revived | Impedance ρ vs day | Edge vs interior p |
| --- | --- | --- | --- | --- |
| T2 | 914 | 67 / 50 | −0.98 (1,394 → 83 kΩ) | 0.97 |
| T3 | 399 | 0 / 0 (no active channels) | −0.98 (607 → 118 kΩ) | 0.13 |
| T10 MFG | 329 | 59 / 56 | −0.70 | 0.88 |
| T10 dPCG | 329 | 31 / 26 | −0.21 | **0.006** |
| T11 lateral | 1,677 | 24 / 16 | −0.95 | **0.025** |
| T11 medial | 1,677 | 35 / 35 | −0.99 | **0.039** |
| (plus A1, S1, S2, S3, T1 as before) | | | | |

**Pooled over 11 arrays:**

- Revived/silenced = **0.78** (monkey N 0.74).
- Median channel-level kurtosis 4.1.
- Edge electrodes decline faster in 5/11 arrays at p < 0.05. **Fisher-combined p = 0.0012**; median edge − interior slope = −0.08 log-rate per year.
- Impedance declines in every array with measurements.

**The electrode failure process replicates across species:**

- Transient silencing.
- Abrupt single-electrode changes.
- Faster loss of spiking at array edges, consistent with micromotion strain (Forrest 2025).
- Falling impedance.

## 2026-10-06 ~08:40 — BrainGate humans: 16 arrays (12 participants)

Zip 3 (T5, T6, T7) passed testzip and was extracted. T8 and T9 are still downloading.
`results/braingate_failure` now covers 16 arrays with yield files from 2,000+ sessions.

**Pooled:**

- Revived/silenced = **0.78**.
- Median channel-level kurtosis 3.6.
- Impedance falls in every array with measurements (ρ mostly −0.82 to −0.99).
- T5 over 7.3 years: lateral yield 60 → 30%, medial 51 → 40%.

**Edge effect:**

- Individually significant in 6/16 arrays. Fisher-combined p = 2.3e-6; median edge − interior slope = −0.064 log-rate per year.
- In arrays with at least 10 initially active channels: the edge declines faster in **10/13** (sign test p = 0.046; Fisher p = 5.9e-6).
- With LINK's 2/2 monkey arrays, that is **12 of 15 arrays across species**.

## 2026-10-06 ~08:50 — BrainGate humans: ALL 20 arrays (14 participants, 2,289 sessions)

Zip 4 (T8, T9) passed testzip and was extracted. Every yield archive is now in `D:/ibci-data/braingate/yield/`.
The `.tar.gz` copies were deleted after extraction; the owner's zips in `D:/Downloads` remain as a backup.

`results/braingate_failure/per_array.csv` and `pooled.json`:

- **Transient silencing:** 584 of 730 silenced electrodes later revived (**0.80**). Monkey N: 0.74.
- **Abrupt changes:** median channel-level kurtosis 3.6.
- **Switching rates:** median h_off = 0.0098/day, h_on = 0.0065/day (monkey N: 0.0031 and 0.0008).
- **Impedance declines in 17/18 arrays with measurements** (median ρ = −0.92). The exception is T9 lateral.
- **Edge effect:**
  - Individually significant in 8/20 arrays. Fisher-combined p = **2.0e-8**; median edge − interior slope −0.064/yr.
  - In arrays with at least 10 initially active channels: 12/17 (sign test p = 0.072; Fisher p = 4.9e-8).
  - With LINK monkey N (2/2 arrays), that is 14/19 arrays across species.
- **Long spans:** T5, 7.3 years (yield 60 → 30% lateral, 51 → 40% medial); S3, 5.4 years (84 → 10%); T11, 4.6 years; T6, 3.2 years.

**Conclusion:** the electrode-failure process measured in one monkey generalizes to 20 human arrays:

- channel loss is mostly transient;
- abrupt single-electrode changes are common;
- spiking is lost faster at array edges (consistent with micromotion strain);
- impedance falls over the years.

## 2026-10-06 ~09:10 — Regularization finding replicated in Chewie and Mihili (smaller effect)

The owner will download `decoding_T6` later. Work continues without it.

`scripts/reg_tradeoff.py --dataset perich` → `results/reg_tradeoff_{C,M}`. The **1% rule** picks the largest α whose
same-day median R² is within 1% of the best (chosen per subject, using same-day data only). Its paired gain over α = 0.1,
from `results/reg_tradeoff/rule_across_subjects.csv`, with cluster-bootstrap CIs:

| Subject | Rule α | Same day | 1 d | 7 d | 30 d | 120 d | 480 d |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N (LINK, SBP) | 1e4 | +0.003 | +0.021 | +0.035 | +0.042 | +0.056 | +0.090 |
| C (sorted units) | 1e4 | +0.002 | +0.008 | +0.012 | +0.001 (n.s.) | +0.031 | +0.032 |
| M (sorted units) | 1e3 | +0.000 | +0.001 | +0.000 | +0.001 | +0.001 | +0.001 |

**Conclusion:** regularizing for the future never costs same-day accuracy and always helps cross-day in direction.
The effect is large for dense SBP features (LINK), moderate for Chewie, and negligible for Mihili. Report it as a
recommended default with a dataset-dependent effect size, not a universal large gain.

## 2026-10-06 ~10:50 — Prior-art alert from the owner: Pun et al. 2024 (MINDFUL); systematic claim-by-claim check started

The owner found **Pun et al., *Communications Biology* 2024**, "Measuring instability in chronic human intracortical neural
recordings towards stable, long-term brain-computer interfaces" (PMC11494208).

**What it does:**

- Data: 2 BrainGate humans (T11: 15 sessions over 142 days; T5: 6 sessions over 28 days), with fixed decoders.
- Method: a label-free instability score (MINDFUL; KL divergence of neural features plus decoder outputs in 60 s windows) that correlates with
  decoder angle error (r = 0.91 for T11, 0.59 for T5).
- Not covered: electrode-level failure, a simulator, regularization or recalibration budgets, monkeys or multiple labs.

**Relevance:**

- It is **prior art for the abandoned topic 1** (label-free monitoring). The evening novelty sweep missed it; the dataset
  sweep listed its data (Dryad dryad.n2z34tn5s) but not its claim. The pivot made it moot for the main paper.
- It **overlaps with one sub-result** (label-free statistics do not predict performance beyond elapsed time), and the two
  apparently conflict. Pun et al. report raw correlations; our pilot found that the predictive information disappears once
  elapsed time is controlled.

**New analysis to add:** compute a MINDFUL-style KL-divergence score on LINK and BrainGate T6, and on MINDFUL's own public
data, then report raw vs time-partialled correlations with performance. This is a clean, publishable test whichever way it comes out.

**Action:** a sub-agent is running a rigorous claim-by-claim prior-art check (C1–C7) before any writing.

## 2026-10-06 ~11:00 — Simulator v2: age-dependent presets

Late-period calibration (`calibrate_sim.py --min-day 700`; failure rates fitted on days ≥ 700) finished with **loss 0.063**.

Early vs late parameters:

| Parameter | Early | Late |
| --- | --- | --- |
| s_mix0 | 0.96 | 0.77 |
| s_mix | 1.48 | 1.32 |
| tau_mix (d) | 15 | 12 |
| rho0 | 0.042 | 0.032 |
| tau_rho (d) | 360 | 527 |
| h_off (/d) | 0.0038 | 0.0034 |
| h_on (/d) | 0.0018 | 0.0017 |

Late fit on a fresh seed, real vs simulated L2/own: 0.80 vs 0.71 at 1 d, 0.32 vs 0.15 at 30 d, 0.17 vs 0.11 at 120 d.

**Conclusion:** drift is smaller and slower in the mature implant, while electrode switching is about the same. Shipped as presets:
`sim.load_preset("early" | "late")` (`src/ibci/sim_presets.json`); see `docs/SIMULATOR.md`.

## 2026-10-06 ~11:20 — MINDFUL-style instability score vs elapsed time (LINK)

`scripts/mindful_test.py` → `results/mindful_test`. For all 537 anatomy pairs we computed Gaussian KL divergences, day j vs day i:

- neural features on day i's top-10 PCs, with per-day z-scoring;
- decoder outputs.

| Predictor | Raw Spearman with R² | Partial given log(days) |
| --- | --- | --- |
| log(days) | −0.67 | — |
| KL neural | −0.34 | −0.16 |
| KL output | −0.22 | −0.16 |
| KL sum | −0.36 | −0.17 |

Time-blocked cross-validated prediction of R², MAE:

| Predictors | MAE |
| --- | --- |
| days only | 0.085 |
| KL only | 0.114 |
| days + KL | 0.090 |

**Conclusion:** a MINDFUL-style score carries some real information (partial ρ about −0.17), but about half of its raw correlation
with performance is elapsed time, and it does not improve prediction beyond the calendar.

**Caveat:** Pun et al. (2024) used closed-loop human sessions with a fixed decoder and 60 s windows. The fair claim is that raw
instability–performance correlations across days are substantially confounded by elapsed time, and should be reported
alongside time-partialled values. A re-test on MINDFUL's own public data (Dryad dryad.n2z34tn5s, 0.41 GB) would be ideal;
it needs a browser download.

## 2026-10-06 ~11:40 — Claim-by-claim prior-art check done (`docs/knowledge/06_prior_art_check.md`)

**Verdicts:**

| Verdict | Claims |
| --- | --- |
| NEW | C2b (no incremental validity of label-free metrics beyond elapsed time; none of 27 citing papers control for time); C4 (regularize for the future, for iBCI); C6a (electrode revival and switching); C7 (calibrated simulator; drift slows later) |
| New quantitatively | C3 (gain-explained fraction; tension with Bishop 2014) |
| Partially known | C1 (decay documented per dataset; conserved cross-lab course and ladder new; Wilson 2025 human −46% at 1–2 weeks corroborates); C5 (estimator known: Kuzborskij & Orabona 2013, SmoothBatch, Bayesian updates; harm-rate analysis new); C6b; C6c (Forrest 2025 strain; Patel 2023 cracks; our 20-array time-to-loss test and the SBP dissociation are new) |
| Already published | C6d (impedance decline): a replication only |
| Contested | C2a (label-free alignment ≤ renormalization) |

**C2a reconciled with our data** (LINK ladder, median R²):

| gap | L1 recentre (mean-only) | L4u Procrustes | L2 full renorm |
| --- | --- | --- | --- |
| 1 d | 0.141 | 0.187 | 0.201 |
| 7 d | 0.070 | 0.116 | 0.148 |
| 30 d | −0.037 | 0.032 | 0.045 |
| 120 d | −0.038 | −0.008 | 0.010 |

- Alignment beats the mean-only baseline used in prior work (L4u > L1 in 58% of pairs), consistent with Wilson 2025 and
  Degenhart 2020. But it does not beat full per-channel z-scoring (L4u > L2 in only 26% of pairs).
- **Reframed claim:** much of the reported benefit of label-free stabilizers over mean recalibration is already obtained
  by per-channel variance normalization.

The MINDFUL public data is being downloaded through the Dryad API with the owner's token (not stored anywhere). The owner
was advised to reset the Dryad API credentials afterwards.

## 2026-10-06 ~12:00 — MINDFUL re-analysis on its own public data (closed loop): a refined, more nuanced C2b

- Data: Dryad dryad.n2z34tn5s, 412 MB, downloaded through the Dryad API with the owner's token (not stored).
- Script: `scripts/mindful_reanalysis.py` → `results/mindful_reanalysis`.
- Setup: fixed online decoder, closed-loop cursor blocks; non-overlapping 60 s windows; KL against day 1 (the reference).
  - Neural features: top-10 PCs of log SBP for T11, or sqrt threshold crossings for T5 (no SBP in its release).
  - Decoder output: online velocity.
  - Performance: median angular error (AE) per window.

**T11** (15 days, 145 windows):

- AE vs elapsed days: r = 0.84.

| | Raw r with AE | Partial given days |
| --- | --- | --- |
| KL neural | 0.82 | 0.37 |
| KL output | 0.81 | 0.73 |
| Sum | 0.60 | −0.09 |

- Leave-one-day-out MAE:

| Model | MAE |
| --- | --- |
| days | 17.1° |
| days + KL neural | 17.5° |
| **days + KL output** | **11.9°** |
| KL output alone | 12.8° |

**T5** (6 days, 73 windows; non-monotonic, with a recovery):

- AE vs days: r = 0.17.

| | Raw r | Partial given days |
| --- | --- | --- |
| KL neural | 0.50 | 0.49 |
| KL output | 0.79 | 0.78 |

- Leave-one-day-out MAE:

| Model | MAE |
| --- | --- |
| days | 32.9° |
| days + KL neural | 38.3° |
| days + KL output | 20.8° |
| **KL output alone** | **15.0°** |

**Refined conclusion** (replaces the blanket "label-free adds nothing beyond time"):

1. **Neural-feature instability metrics mostly track elapsed time.** This holds for our offline LINK analysis (partial ρ ≈ −0.16,
   no CV gain) and for MINDFUL's T11 (partial 0.37, no gain in leave-one-day-out). They are no better than a calendar for prediction.
2. **In closed-loop use, decoder-output statistics carry genuine information beyond elapsed time:** 30–55% lower
   leave-one-day-out error in both participants. This is plausibly because the user's corrective behaviour against a degrading
   decoder shows up in the outputs. It is a symptom detector, which is fine for triggering recalibration.
   Consistent with our pilot, where output-based features were the strongest label-free signals.
3. **Practical guidance:** monitor decoder outputs during closed-loop use, and do not rely on neural-only drift scores.
   Report time-partialled statistics.

**Credit:** MINDFUL's decoder-output component is the useful part. The raw correlation of the neural component is largely a time confound.

## 2026-10-06 ~11:58 — HUMAN decoder drift: BrainGate T6

- Data: `decoding_T6` (124 closed-loop cursor sessions over 1,123 days), obtained through the owner's VPS and verified by gzip CRC.
- Script: `scripts/replicate_braingate_decoding.py` → `results/replication_bg/T6_ladder.csv`; 369 pairs.
- Setup: intended movement direction (unit vector from cursor to target) during go periods; 20 ms SBP; the same ridge decoder and ladder.

| gap | L2/own (renorm) [95% CI] | L3/own (+ gains) [95% CI] | own R² |
| --- | --- | --- | --- |
| 1 d | 0.73 [0.38, 0.88] | 0.90 [0.79, 0.98] | 0.28 |
| 7 d | 0.66 [0.51, 0.83] | 0.96 [0.87, 1.05] | 0.21 |
| 30 d | 0.71 [0.59, 0.83] | 0.94 [0.88, 0.98] | 0.23 |
| 120 d | 0.51 [0.32, 0.61] | 0.84 [0.79, 0.88] | 0.23 |
| 480 d | 0.30 [0.16, 0.41] | 0.81 [0.78, 0.90] | 0.29 |

- **The overnight loss is conserved across species:** a human retains 0.73, against 0.72–0.77 in the three monkeys.
- **Long-term decay is slower in this human:** 0.51 at 4 months and 0.30 at 16 months, against about 0 at 4 months in the monkeys.
  - Possible causes: species or array, closed-loop intended-direction labels vs measured kinematics, or task consistency. This needs care.
- **Per-channel gain changes explain most of the human loss** (L3 retention 0.81–0.96), unlike the monkeys (0.1–0.4 at long gaps).
  This is consistent with Bishop 2014 ("tuning parameters on the same electrode move together between days").
- L4u (Procrustes) ≤ L2 again. L1 (mean-only) is far below L2 (e.g. 0.007 vs 0.111 at 7 d), the same pattern as in the monkeys.
- **Revised C1 and C3:**
  - The overnight drop (about 25–30%) replicates in 3 monkeys and 1 human.
  - The long-term rate and the share explained by gains differ between subjects.
  - Report the conserved overnight drop and the subject-specific long-term course. Per-channel gain recalibration (96 parameters) is
    especially effective in the human.

## 2026-10-06 ~12:15 — "Regularize for the future" holds in a human (T6)

`scripts/reg_tradeoff.py --dataset braingate --subject T6` → `results/reg_tradeoff_T6`; 40 training sessions.

**Median R², α = 0.1 vs 1e4:**

| gap | α = 0.1 | α = 1e4 |
| --- | --- | --- |
| same day | 0.224 | 0.240 |
| 1 d | 0.129 | 0.168 |
| 7 d | 0.105 | 0.137 |
| 30 d | 0.160 | 0.172 |
| 120 d | 0.133 | 0.170 |
| 480 d | 0.106 | 0.123 |

**1% rule** (α = 1e4), paired gain with cluster-bootstrap CI:

| gap | gain | 95% CI |
| --- | --- | --- |
| same day | +0.008 | [0.003, 0.013] |
| 1 d | +0.011 | [0.002, 0.075] |
| 7 d | +0.025 | [0.016, 0.032] |
| 30 d | +0.016 | [0.013, 0.030] |
| 120 d | +0.018 | [0.009, 0.030] |
| 480 d | +0.017 | [0.005, 0.027] |

- All CIs exclude 0, and 89–100% of training sessions improve.
- **Across subjects** (`results/reg_tradeoff/rule_across_subjects.csv`): monkey N gains a lot, human T6 clearly, Chewie moderately, Mihili negligibly.
  Same-day accuracy never drops (T6 even improves).

**Figure:** `fig_decay_xsubject` now includes human T6.

- The overnight retention of about 0.73 is shared by all four subjects.
- The human decays more slowly long-term.
- Per-channel gains recover the human almost fully (0.81–0.96).

## 2026-10-06 ~13:00 — Manuscript preparation: venue, template, decay summary

- **Venue:** Nature Communications, the owner's suggestion, agreed. Brief saved to `docs/paper/STYLE_BRIEF.md`.
- **Template:** Springer Nature LaTeX template (sn-jnl v3.1, `sn-nature`) in `manuscript/`; it compiles with MiKTeX.
- **Packaging:** `pyproject.toml` (distribution `ibci-drift`), MIT licence, `CITATION.cff`, `.zenodo.json`.
- **Figure 1** (`fig1_overview`): 2,833 sessions from 17 individuals, plus the analysis design.

**Decay summary** (`scripts/decay_summary.py` → `results/decay_summary.csv`; cluster-bootstrap CIs):

| Subject | Overnight retention | Time to 50% retention |
| --- | --- | --- |
| Monkey N | 0.73 [0.62, 0.78] | 8.1 d [3.7, 10.1] |
| Monkey C | 0.72 [0.62, 0.78] | 9.2 d [4.6, 19.5] |
| Monkey M | 0.77 [0.71, 0.85] | 33.9 d [9.5, 44.0] |
| Human T6 | 0.73 [0.41, 0.87] | 125 d [56, 240] |

**Headline:** the overnight drop is conserved (overlapping CIs), while the half-time varies from about 1 week to about 4 months.

## 2026-10-06 ~14:00 — Manuscript draft v1 (Nature Communications), formal tests, authorship

**Authors and commits.** Authors (owners' order): Md. Imtiaj Alam Sajin, Esm E Moula Chowdhury Abha, Md Wahiduzzaman
Suva, all American International University-Bangladesh (ORCIDs in `CITATION.cff`). Commits now have one author each,
alternating Wahid and Esme, with no co-author trailers (see `CLAUDE.md`).

**Formal tests** (`scripts/paper_stats.py` → `results/paper_stats.json`; one value per training session or per array):

- Overnight retention does not differ between subjects: Kruskal–Wallis H = 1.98, d.f. 3, P = 0.58 (n = 30, 42, 14, 24 sessions).
- Renormalization beats label-free Procrustes alignment: N +0.019 (P = 1.1e-10, 80 sessions); T6 +0.011 (P = 1.9e-7, 122).
  Alignment beats mean-only updating: N +0.053 (P = 2.4e-8); T6 +0.120 (P = 2.1e-16). Two-sided Wilcoxon.
- 1% rule, session-level cross-day gain: N 0.060 (40/40, P = 1.8e-12); C 0.017 (39/40, P = 7.8e-11);
  M 0.001 (19/22, P = 2.1e-4); T6 0.021 (40/40, P = 1.8e-12).
- Harmful recalibration, CV vs history: 9/86 vs 1/86 at n = 10 (Fisher P = 0.018); 7/86 vs 0/86 at n = 20 (P = 0.014).
- Edge effect: Fisher-combined one-sided P = 2.0e-8 over 20 human arrays (8 individually < 0.05); sign test 14/19
  arrays with ≥ 10 initially active electrodes (17 human + 2 monkey), one-sided P = 0.032.
- Gain-recovered share of the loss, (L3 − L2)/(own − L2), median per gap: N 14–34%, C 28–42%, M 19–35%, T6 64–81%.

**Manuscript** (`manuscript/main.tex`, `manuscript/supplementary.tex`; both compile with pdflatex):

- Title (15 words): "Shared patterns of decoder drift and electrode failure suggest simple safeguards for implanted brain–computer interfaces".
- Abstract 150 words; main text about 3,470 words; Methods about 2,020 words; 7 figures; 54 references.
- Style audit (`manuscript/check_style.py`): mean sentence length about 16 words, no em dashes, no banned words, subheadings ≤ 57 characters.
- References generated in citation order by `manuscript/make_bib.py` from OpenAlex metadata checked against Crossref.
  Hahn et al. *Nat Med* 2026 DOI is not yet registered (404), so it is cited "in the press" with the medRxiv preprint.
- Supplementary Information: Notes 1–3 (monitoring pilot, simulator, corrected analysis choices) and Tables 1–12, all
  generated by `scripts/make_supp_tables.py` from `results/`.
- Corrections found while checking the draft against code: calibration used 12 base sessions (not 16); augmentation
  used 8 simulated futures with Δt ≤ 120 d; the LINK CV "days + divergence" model uses both divergences.
- An AI-use statement is included in Methods, as Nature Portfolio policy requires; the owners should review it.

**Data.** BrainGate `decoding_T5.tar.gz` (11.34 GB) is downloading from the owners' VPS to G:. It will be stream-converted
by `scripts/tools/compact_braingate_decoding.py` (verified identical loader output on a T6 session). The four Dryad yield
zips in Downloads (5.3 GB, already extracted) were deleted to free space.

## 2026-10-06 ~21:00 — Referee review of draft v1; robustness re-analyses; T5 and T9

An independent referee-style review (sub-agent) checked 36 numbers against the result files (all matched apart from a
few listed issues) and raised major points. Each was tested rather than argued:

- **"Overnight" pairs include 2-day gaps** (selection window max(1 day, 25%)). True 1-day retention: N 0.74 (n = 21),
  C 0.70 (31), M 0.80 (12) — holds; **T6 0.41 (n = 5)** vs 0.79 for 2-day pairs (19) — the human "overnight" value was
  carried by 2-day pairs. Report 1- and 2-day pairs separately (`results/revision_stats.json`).
- **Decay depends on decoder regularization** (`scripts/decay_alpha.py`, same pairs, α = 10⁴): monkey N retention
  0.79 / 0.63 / 0.41 / 0.30 / 0.01 at 1 / 7 / 30 / 120 / 480 d, versus 0.73 / 0.54 / 0.19 / 0.04 / −0.36 at α = 0.1.
  Much of the apparent drift loss with the default decoder is over-fitting. Running for C, M, T6, T5, T9.
- **Electrode revival is mostly fluctuation** (`scripts/failure_robustness.py`). Revival survives artefact controls
  (fixed-µV threshold 0.72, waveform ≥ 30 µV 0.71, deep silence < 0.5 Hz 0.71, array-wide shifts removed 0.77,
  −5.5 RMS 0.74), but a shuffled-session-order null gives the same values (0.79 / 0.71 / 0.72). Only silences of
  ≥ 30 days carry signal: 96/168 recover vs 33/81 in the null (Fisher P = 0.021), and such silences are twice as common
  as in the null.
- **Edge effect is modest** (initially active electrodes only, array as the unit): edge minus interior slope negative
  in 13/17 human arrays (sign P = 0.025, Wilcoxon P = 0.036); distance-from-centre correlation 13/17 (Wilcoxon
  P = 0.013); adjusted for initial log rate 12/17 (P = 0.072 / 0.087). The earlier Fisher P = 2e-8 treated dependent
  electrodes as independent and included never-active electrodes.
- **The 1% rule versus ordinary same-day tuning:** per-session best same-day α already equals the rule in C, M and T6
  (no difference); in N the rule adds +0.004 (P = 6.9e-5). The real effect is the small default (α = 0.1, LINK paper)
  versus any same-day tuning (`results/revision_stats.json`).
- **Harmful recalibration** holds with the correct paired test: exact McNemar P = 0.0078 (n = 10), 0.016 (n = 20);
  Holm 0.039 / 0.063. The history policy chose λ = 10⁴ for every pair at n ≤ 50, i.e. a fixed strong shrinkage.
- Monkey C and M base decoders used α = 1 (not 0.1 as Methods stated); fix in v2.

**T5** (BrainGate, 92 closed-loop sessions over 7.2 years, 192 channels; converted with CRC check): retention after
1–2 days 0.62 [0.34, 0.76], half-time 3.7 d [1.0, 9.5] (T6: 125 d). Per-channel gains recover 78–92% at every gap, as
in T6 (both humans), unlike the monkeys (14–42%). 1% rule: +0.008 same day, +0.018 to +0.040 cross-day, all CIs > 0.
**T9** downloaded (11.73 GB) and converted to `G:/ibci-data/braingate/decoding/T9`; pipeline queued.

Decision: pause manuscript writing until all re-analyses finish, then rewrite v2 around what holds.
