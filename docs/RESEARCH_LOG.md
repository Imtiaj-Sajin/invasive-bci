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
