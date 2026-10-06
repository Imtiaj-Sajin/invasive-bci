# Chronic intracortical recording: degradation, simulators, on-implant compression, spike sorting

*Compiled 2026-10-05 by a research sub-agent (about 57 searches and 35 page fetches, plus Europe PMC API novelty checks).
Corrections: the 980-electrode SEM study is Bjånes et al. 2025, not Hughes. Bullard 2020 is a hardware-complications
review with no signal trajectories. The BrainGate 14-participant paper (Hahn et al.) was still a preprint at the time of checking.*

## 1. Chronic signal degradation (evidence)

- **Sponheim et al. 2021, JNE 18:066044** ([link](https://iopscience.iop.org/article/10.1088/1741-2552/ac3eaf))
  - 55 Utah arrays in 17 macaques and 2 humans, about 6,000 datasets over 9 years.
  - Median array lifespan 622 days; 16 arrays exceeded 800 days.
  - Yield fell about 2% per 30 days (per summary).
  - Data on request only.
- **Hahn, …, Hochberg, Willett, medRxiv Jul 2025** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12236888/))
  - 14 BrainGate participants, 20 arrays, 2,319 sessions, implants of 296 to 2,780 days.
  - Mean yield 35.6%, falling from 41% in the first 3 months to 34% in the last 3.
  - Impedance rose sharply after implant, then declined slowly.
  - **Spike amplitude and noise fell together, so SNR stayed roughly constant**; this is unexplained.
  - 3 of 14 arrays failed to decode consistently.
  - Data "to be released upon acceptance."
- **Colachis et al. 2021, JNE** (DOI 10.1088/1741-2552/ac1add): one participant over 5 years. Fast decline in year 1, then slow; still decoding at year 5.
- **Hughes et al. 2021, JNE** ([PubMed](https://pubmed.ncbi.nlm.nih.gov/34320481/)): Pitt participant, more than 1,500 days.
- **Pitt "decade of ICMS"** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12363726/)): stimulation thresholds rose about 3.5 µA/yr; 62 ± 15% of electrodes still evoke sensations (55% at 10 years).
- **T15 home use** ([Nat Med 2026](https://www.nature.com/articles/s41591-026-04414-6)): about 2 years, more than 3,800 h, stable to improving accuracy.
- **Chen et al. 2023, JNE** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7617000/)): 1,024 channels in monkey V1/V4 over about 3 years. SNR and amplitude fell while impedance fell. Phosphene-evoking electrodes dropped from 98% to 24%.

**Failure modes:**

- Barrese 2013 and 2016: array ejection caused about 30% of chronic failures.
- Woeppel 2021: human explants ([doi](https://doi.org/10.3389/fbioe.2021.759711)).
- Bjånes 2025 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11419230/)): 980 electrodes; for SIROF tips, 1 kHz impedance tracks damage, and SIROF was 2× more likely than platinum to record units.
- Szymanski 2021: human histology, neuron loss between 8 and 16 weeks.

**Biophysical models:**

- Lempka 2011: noise model ([JNE](https://iopscience.iop.org/article/10.1088/1741-2560/8/4/045006)).
- Malaga 2016: a thin tip layer, not the glial scar, explains impedance ([JNE](https://iopscience.iop.org/article/10.1088/1741-2560/13/1/016010)).
- Forrest 2025, JNE (DOI 10.1088/1741-2552/ae1bda): micromotion strain is highest at edge and corner shanks.

**Neuralink 2024:**

- Thread retraction, with about 85% of threads lost ([New Atlas](https://newatlas.com/technology/first-human-neuralink-implant-fails/)).
- Fix: the "recording algorithm [made] more sensitive to neural population signals."
- Second participant: about 400 of 1,024 electrodes working ([PopSci](https://www.popsci.com/technology/second-neuralink-implant/)).

**Gaps:**

- No model explains the human multi-year signature (impedance, amplitude and noise all falling).
- No per-electrode mechanism inference.
- No yield forecasting.

## 2. Public long-term datasets

| Dataset | Span | Content |
| --- | --- | --- |
| **LINK** (Temmar…Chestek, NeurIPS 2025 D&B), **DANDI 001201, 12.6 GB** | **312 sessions over 1,242 days** | Threshold crossings and spike-band power, 20 ms bins, 96 channels, finger task ([site](https://chesteklab.github.io/LINK_dataset/), [OpenReview](https://openreview.net/forum?id=TAdeh1dLzu)) |
| Indy/Loco (O'Doherty/Sabes), Zenodo 583331 | 37 Indy sessions over about 10 months | 29 sessions have raw broadband at 24.4 kHz ([Zenodo](https://zenodo.org/records/583331)) |
| FALCON | Days to weeks | H1, H2, M1, M2, B1 ([datasets](https://snel-repo.github.io/falcon/datasets.html)) |
| Pun et al. 2024 (Dryad) | T5 28 days; T11 142 days | Threshold crossings, plus spike-band power for T11 ([Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.n2z34tn5s)) |
| Brain-to-text '25 (T15) | 45 sessions over 20 months | 256 channels, threshold crossings and spike-band power ([Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85)) |

No public dataset combines raw broadband with impedance over years.

## 3. Simulators: no chronic "digital twin" found

- Existing tools and what they cover:
  - [MEArec](https://link.springer.com/article/10.1007/s12021-020-09467-7) and [SpikeInterface generation](https://spikeinterface.readthedocs.io/en/stable/modules/generation.html): drift and noise, but no amplitude decay, unit loss or electrode failure.
  - Kilosort4: drift only.
  - [DREDge](https://www.nature.com/articles/s41592-025-02614-5): motion correction.
  - [SimSort](https://arxiv.org/abs/2502.03198): training spike sorters on simulated data.
  - [Laquitaine 2024](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v1): sorters isolate about 15% of neurons within 50 µm.
  - [neural-data-simulator](https://github.com/agencyenterprise/neural-data-simulator) and [bcisimulator](https://github.com/jtcostello/bcisimulator).
- Closest prior work:
  - [Wan 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10213332): firing-rate degradation with "relatively arbitrary" parameters; the authors ask for data-derived degradation models.
  - [Stephens 2021](https://www.biorxiv.org/content/10.1101/2021.05.18.444641v1): GAN for channel loss.
  - [Sussillo 2016](https://www.nature.com/articles/ncomms13749): ad hoc perturbations.

## 4. On-implant data reduction

- **Neuralink N1:** about 200 Mbps raw against about 1 Mbps of radio.
  - The challenge asked for 200× lossless compression; best results were 3.35 to 3.5×, because thermal noise dominates ([brainwire](https://github.com/phoboslab/neuralink_brainwire), [HN](https://news.ycombinator.com/item?id=40531157)).
  - **No peer-reviewed information-theoretic analysis of the challenge exists.**
- **Paradromics:** 421 electrodes per module, 100 Mbit/s optical link. SONIC: over 200 bps (sheep, [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.30.679683v1)).
- **Decoding-evaluated reduction studies:**
  - Nason 2020 (spike-band power).
  - Even-Chen 2020 (about 10× power savings).
  - [Savolainen 2022](https://www.biorxiv.org/content/10.1101/2022.03.25.485863v2.full): about 27 bps per channel with under 1% decoding loss.
  - [Karpowicz 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11528220/): 4-bit LFP, 96.8% power saved.
  - [Mohan/Basu](https://arxiv.org/abs/2312.09503).
  - NeuroZip.
  - [Shi 2026](https://www.nature.com/articles/s44172-026-00735-z): send-on-delta, 11.4×.
- **Benchmarks:** Meyer & Zamani 2026 call the rate-versus-throughput power law "heuristic" ([JNE](https://iopscience.iop.org/article/10.1088/1741-2552/ae6dfd)). NeuroBench and BioCAS score decoding against operations, not transmitted bits.
  - **Gap:** no standardized benchmark of decoding versus bits per second per channel, and none tracking how this changes as the implant ages.

## 5. Spike sorting

- **Does sorting help decoding?**
  - Todorova 2014: sorting helps only if noise waveforms are kept.
  - Christie 2015: mixed results.
  - Trautmann 2019: sorting is not needed to recover population dynamics.
- **Gaps:** no ground-truth sorting benchmark for Utah arrays, and no study of how sorting's value changes over implant lifetime.

## The agent's ranked topics

1. **A calibrated mechanistic simulator of chronic degradation, plus a benchmark suite.** Novelty verified. Closest prior work: Wan 2023, MEArec, Malaga, Lempka.
2. **A rate–distortion benchmark for on-implant data reduction across implant age.** Built on the Indy broadband sessions plus the Neuralink challenge data.
3. **Physics-informed degradation augmentation for decoders.** Tested on LINK and FALCON.
4. **Per-electrode failure-mechanism inference using simulation-based inference.** Blocked by the lack of public impedance data.
5. **Sorting vs threshold crossings vs spike-band power across implant lifetime.**

## Addendum 2026-10-06: what Hahn et al. (*Nat Med* 2026; preprint PMC12236888) report vs what we add

Source: open-access preprint, read 2026-10-06. Same BrainGate data as our `results/braingate_failure`.

| Topic | Hahn et al. | This project |
| --- | --- | --- |
| Edge vs interior | Qualitative: "insulation degradation… particularly around the edges of the planar array"; low impedances in T2, T3, T5, T6 (supplementary). No quantitative edge vs interior yield or spiking comparison | Quantitative per-electrode activity-decline slopes. Edge faster in 14/19 arrays (humans plus monkey N); Fisher p = 2e-8 |
| Electrode revival / switching | Not analysed (array-level dips only, e.g. S2 after repair) | 80% of silenced electrodes revive; two-state Markov switching rates per array |
| Impedance | "increased sharply post-implant, followed by a consistent and gradual decline" | Replicated (17/18 arrays; monkey N too). Not new |
| Abrupt vs gradual | Described (e.g. S3 pedestal trauma), not modelled | Heavy-tailed channel-level changes, with session-wide events separated |
| Monkey comparison | Better longevity than prior NHP studies | Same failure-process statistics in 3 monkeys and 20 human arrays |
| Simulation | None | Calibrated simulator |
| Decoding | dSNR > 1 in 11/14 arrays; tuning correlation > 0.6 within 1 month | Decoder-level decay timescale, recalibration recipes, regularization |

