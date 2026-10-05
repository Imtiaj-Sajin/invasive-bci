# Intracortical BCI landscape 2024–2026, bottlenecks, and candidate computational directions

*Compiled 2026-10-05 by a research sub-agent (about 60 web searches). Claims marked [UNVERIFIED] come from
fan or aggregator sites only. Some Nature, bioRxiv and PMC pages could not be fetched directly, so a few
claims rest on abstracts or secondary reports.*

## 1. Milestones

| Org | Key facts 2024–2026 | Regulatory status |
|---|---|---|
| **Neuralink** | N1 has 1,024 electrodes. In the first patient, 85% of threads retracted and about 870 electrodes were lost; software recovered performance ([PopSci](https://www.popsci.com/health/neuralink-wire-detachment/)). 21 participants by Feb 2026 ([Debrief](https://thedebrief.org/neuralink-reaches-21-patients-as-elon-musk-continues-push-for-high-volume-brain-chip-production/)). Roadmap: 3k channels in 2026, 10k in 2027, more than 25k in 2028 ([Digitimes](https://www.digitimes.com/news/a20250701PD221/roadmap-2028-elon-musk.html)). Oct 1, 2026 blog: self-supervised pretraining on 50k+ hours of unlabeled data, an 11.32 bps cursor record, and calibration cut from about 10 min/day to 10 min/week ([Interesting Engineering](https://interestingengineering.com/innovation/neuralink-brain-chip-cursor-control)). Not peer-reviewed. | PRIME, CONVOY, CAN-PRIME ([NCT06700304](https://clinicaltrials.gov/study/NCT06700304)), GB-PRIME ([NCT07127172](https://clinicaltrials.gov/study/NCT07127172)), VOICE (speech) with Breakthrough designation Apr 2025. Blindsight had no human implant as of Sep 2026 (secondary source). |
| **Synchron** (endovascular) | COMMAND early feasibility study: 6/6 met the primary safety endpoint. About 10 patients and a $200M Series D by Nov 2025. "Chiral" foundation model with NVIDIA. The "100 patients" claim is probably wrong. | Pivotal trial filing planned for end of 2026. |
| **Paradromics** | Connexus, 421 electrodes. First chronic implant June 2026 (U. Michigan). Real-time speech announced Sept 2026, no WPM/WER disclosed. SONIC benchmark claims 200+ bps (preclinical) ([blog](https://paradromics.com/blog/bci-benchmarking/)). | FDA IDE Nov 2025 (Connect-One) for speech. |
| **Precision Neuroscience** | Layer 7 surface film, 1,024 electrodes, more than 68 patients. | 510(k) April 2025 for implants of 30 days or less only. |
| **BrainGate / UC Davis / Stanford** | Card et al. NEJM 2024 (97.5% accuracy, 125k-word vocabulary). Card et al. *Nat Med* 2026 ([link](https://www.nature.com/articles/s41591-026-04414-6)): 19 months and 3,800+ hours of home use, 56 wpm, continuous background recalibration. Wairagkar et al. *Nature* 2025: voice synthesis within 10 ms, data on Dryad. Kunz et al. *Cell* 2025: inner speech decoding. Wilson et al. *Nat BME* 2025: HMM-based unsupervised recalibration ([link](https://www.nature.com/articles/s41551-025-01536-z)). | Academic feasibility studies (BrainGate2). |
| **Science Corp** | PRIMA (retinal) in NEJM 2025; CE mark July 2026. | Retinal, not cortical. |
| **CorTec** | First Brain Interchange implant July 2025 (stroke). | NIH feasibility study. |
| **Axoft** | 11 first-in-human patients; $55M Series A Apr 2026. | FDA trial targeted for 2027. |
| **Motif** | DOT stimulator for depression; feasibility study approved Apr 2026. | |
| **Merge Labs** | $252M seed (OpenAI), ultrasound-based, non-implanted. | Research stage. |
| **China** | Neuracle NEO (epidural) got NMPA approval Mar 13, 2026, the first commercial implanted BCI anywhere ([Bloomberg](https://www.bloomberg.com/news/articles/2026-03-13/china-approves-first-brain-implant-for-commercial-use)). NeuCyber Beinao-1: 7 patients. StairMed: 64/256 channels. | Landscape paper [arXiv 2607.07185](https://arxiv.org/abs/2607.07185): obstacles are durability, standardization, generalizability. |

## 2. Bottlenecks, ranked by evidence, computational tractability and number of future users

1. **Nonstationarity and the recalibration burden.** This is the most cited bottleneck.
   - Static decoders degrade within days ([Nat Rev Bioeng 2025](https://www.nature.com/articles/s44222-024-00239-5)).
   - Wilson et al. 2025: distribution-matching recalibration "accumulates compounding errors over time"; target-inference methods hold up.
   - ALIGN (2026, [arXiv 2603.18299](https://arxiv.org/html/2603.18299v1)): about 45% test word error rate at 71 days out on T15.
   - Neuralink's headline 2026 result is about exactly this problem.
2. **No standardized, clinically meaningful metrics or benchmarks.**
   - FDA's Sept 2024 workshop ([transcript](https://www.fda.gov/media/182567/download)).
   - Only 17.9% of iBCI studies assess clinical outcomes ([medRxiv](https://www.medrxiv.org/content/10.1101/2024.10.15.24315534v1.full)).
   - BCI Society workshop paper ([JNE 2026](https://iopscience.iop.org/article/10.1088/1741-2552/ae9eee)).
   - Speech BCI metrics are inconsistent ([He et al. 2026](https://arxiv.org/html/2603.12279v1)).
   - Meta's NeuralBench: invasive decoding has only FALCON, which is "limited scope (5 datasets)" ([arXiv 2605.08495](https://arxiv.org/html/2605.08495v1)).
3. **Cross-user generalization and data scarcity.**
   - NDT3 gains disappear above about 1.5 hours of target data; scaling "seems unlikely to resolve limitations from sensor variability" ([bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.02.634313v1.full.pdf)).
   - Cross-brain transfer helps only below about 200 sentences ([SNEL](https://snel.ai/publication/cross-brain-transfer-of-high-performance-intracort/)).
4. **Electrode degradation and channel loss.**
   - BrainGate, 14 participants: mean electrode yield 35.6%, 7% decline, fastest drop in year 1 ([Hahn et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12236888/)).
   - Neuralink lost 85% of threads in its first patient.
5. **Bandwidth, power and thermal limits.**
   - "Engineering trilemma of bandwidth, power, and latency" ([JNE 2026](https://iopscience.iop.org/article/10.1088/1741-2552/ae6dfd)).
   - Neuralink's 200x compression challenge runs into an entropy bound of about 3.4x ([HN](https://news.ycombinator.com/item?id=40531157)).
   - Spike-band power about 6.6 µW/ch, LFP about 2.8 µW/ch ([JNE 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11528220/)).
6. **Output reliability, errors and privacy.**
   - Home use: 92% of sentences rated accurate or mostly correct.
   - Inner speech leakage creates a need for gating ([Cell 2025](https://www.cell.com/cell/fulltext/S0092-8674(25)00681-6)).
   - **No intracortical work on calibrated uncertainty was found.**
7. **Offline–closed-loop gap** ([Sci Rep 2019](https://www.nature.com/articles/s41598-019-44166-7)). Limits what offline-only papers can claim.
8. **Spike sorting vs threshold crossings.** Largely settled: threshold crossings and spike-band power work well. Low priority.
9. **Reimbursement.** CMS/FDA RAPID pathway (Apr 2026) needs FDA+CMS-agreed outcomes, which raises the value of #2.

## 3. Explicit calls for work
- FDA 2021 implanted-BCI guidance ([PDF](https://www.fda.gov/media/120362/download)) and the 2024 outcome workshop.
- iBCI-CC ([site](https://www.ibci-cc.org/)): workgroups on Clinical Study Endpoints, Interoperability/ISO, and Neural Data Privacy.
- IEEE P2731 (BCI terminology and metadata) and IEEE P2794 (reporting standard).
- NIH BRAIN Initiative: FY2026 budget $429M; Cures Act funding ends after FY2026.

## 4. Public resources noted
- Brain-to-Text '24: T12, 24 sessions.
- Brain-to-Text '25: T15, 10,948 sentences over 45 sessions and 20 months ([Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85)).
- Wairagkar voice data: 26.9 GB.
- FALCON: H1, H2, M1, M2, B1.
- NLB.
- NDT3 and POYO weights.

**Brain-to-Text word error rate chasing is saturated.** Ensembles plus LLM merging cut error from 9.7% to 5.8%, and
"architectural upgrades did not help" ([arXiv 2412.17227](https://arxiv.org/pdf/2412.17227)).

## 5. The agent's top 5 candidate directions
1. **A longitudinal, label-free stability benchmark and method for communication iBCIs** (a "speech-FALCON"). Built from T12/T15 and FALCON H2. Compare against ALIGN and CORP ([arXiv 2311.03611](https://arxiv.org/abs/2311.03611)). Note: the full n-gram LM plus 6.7B LLM rescoring will not fit locally, so use a 3-gram LM or a 4-bit LLM of 3B parameters or fewer.
2. **Calibrated uncertainty, error detection and intent gating for speech BCIs.** Essentially untouched for intracortical BCIs. Only EEG uncertainty work exists, plus one 2026 cursor preprint ([bioRxiv](https://www.biorxiv.org/content/10.64898/2026.02.25.707999v1.full.pdf)).
3. **Degradation-robust decoding with a calibrated degradation simulator.** Fit to Hahn et al. Compare against SPINT ([arXiv 2507.08402](https://arxiv.org/html/2507.08402)), which covers motor tasks only.
4. **Task-aware compression and power-aware decoding.** Risk: public human data is already binned. Distillation is getting crowded ([BrainDistill](https://arxiv.org/html/2601.17625v1), [LightBeam](https://arxiv.org/pdf/2603.14002)).
5. **Device-agnostic effective communication rate metric.** Partly taken by Jayalath et al. Sept 2026 ([arXiv 2609.02887](https://arxiv.org/pdf/2609.02887)).

Recommendation from this sweep: combine 1+3+5 into a benchmark paper with a reference method, and write 2 as a fast standalone paper.
