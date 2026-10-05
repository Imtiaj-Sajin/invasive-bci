# ML state of the art for intracortical decoding (2023–2026) and research gaps

*Compiled 2026-10-05 by a research sub-agent (about 55 searches; full PDF text was read for NDT3, NDT2, FALCON and POYO+;
other items were read from abstracts). Corrections: Wilson et al. is PRI-T (HMM-based, Nat BME 2025), which is
separate from CORP (Fan et al. NeurIPS 2023, handwriting, stable for 57 weeks without manual recalibration). POYO+ is
calcium imaging and BrainLM is fMRI; neither is intracortical.*

## 1. Neural foundation models

| Model | Data | Compute | Code | Results and limitations |
|---|---|---|---|---|
| **NDT3** ([bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.02.634313v1), NeurIPS 2025) | 2 kh of spiking data from 30+ monkeys and humans; 45M and 350M params | 45M/200 h: 480 A100-h. 350M/2 kh: 20K A100-h | [joel99/ndt3](https://github.com/joel99/ndt3), weights on HF; needs bf16 and FlashAttention-2 (Ampere, so a 3060 Ti works) | Scaling to 2 kh *hurt* the 45M model. Cross-subject data gave no gain over 1.5 h of task data. **Channel shuffle or a half-token shift destroys transfer.** Fails on held-out reach angles, where a Wiener filter wins. |
| **NDT2** (NeurIPS 2023) | Multi-session and multi-subject motor data | 2–72 GPU-h | [context_general_bci](https://github.com/joel99/context_general_bci) | Shallow transfer curves. Monkey data gave no benefit for human decoding. |
| **POYO** ([arXiv 2310.16046](https://arxiv.org/abs/2310.16046)) | 7 monkeys, 158 sessions, 27k units | n/a | torch_brain | Spike tokenization with PerceiverIO. |
| **POSSM** ([arXiv 2506.05320](https://arxiv.org/abs/2506.05320)) | Multi-dataset monkey data | About 8M params, 2–6 ms latency | [site](https://possm-brain.github.io/) | Up to 9x faster than Transformers. Monkey pretraining helps human handwriting. |
| **MtM** ([arXiv 2407.14668](https://arxiv.org/abs/2407.14668)) | IBL Neuropixels | Moderate | Yes | Multi-task masking. |
| **NEDS** ([arXiv 2504.08201](https://arxiv.org/html/2504.08201v3)) | 73 IBL mice | 16 GPUs × 2 days | Yes | Beats POYO+ and NDT2 on IBL. |
| **SPINT** ([arXiv 2507.08402](https://arxiv.org/abs/2507.08402), NeurIPS 2025) | FALCON M1, M2, H1 | Under 2 GB VRAM | Release not confirmed | Few-shot unsupervised R²: M1 0.66, M2 0.26, H1 0.29. |
| CEBRA, AutoLFADS, STNDT, EIT | Single or few sessions | 8 GB is fine | Yes | Mature baselines. |

**2026 models (abstracts only):**
- RPNT: [2601.17641](https://arxiv.org/abs/2601.17641)
- UniBCI: [2605.00061](https://arxiv.org/abs/2605.00061)
- iBrain, 7 kh of iEEG plus spikes: [2609.06960](https://arxiv.org/abs/2609.06960)
- MOJO: [2607.14086](https://arxiv.org/abs/2607.14086)
- NeuroPB: [2608.04389](https://arxiv.org/abs/2608.04389)
- BrainVLA: [2609.34561](https://arxiv.org/abs/2609.34561)
- APST, few-shot FALCON R² of M1 0.65, M2 0.42, H1 0.44: [2609.39080](https://arxiv.org/abs/2609.39080)
- GRAFT: [2606.11066](https://arxiv.org/abs/2606.11066)

**Scaling skepticism:**
- Jiang et al.: 5 selected sessions beat pretraining on all 84 ([bioRxiv](https://www.biorxiv.org/content/10.1101/2025.05.12.653551v3)).
- OmniMouse: models saturate around 80M params ([2604.18827](https://arxiv.org/abs/2604.18827)).
- BrainWideBench: transfer gains depend on how well the pretraining objective matches the task ([2609.22064](https://arxiv.org/abs/2609.22064)).

**Verdict:** building a new foundation model is crowded and impossible on 8 GB. Analyzing and adapting the *released* models is open. NDT3-45M fine-tuning fits on 8 GB; the 350M model needs LoRA.

## 2. Stability and recalibration

**Prior methods:**
- NoMAD ([Nat Comms 2025](https://www.nature.com/articles/s41467-025-59652-y))
- CycleGAN and ADAN ([eLife 2023](https://elifesciences.org/articles/84296))
- Degenhart stabilizer ([Nat BME 2020](https://www.nature.com/articles/s41551-020-0542-9))
- Gallego latent alignment ([Nat Neuro 2020](https://www.nature.com/articles/s41593-019-0555-4))
- Sussillo 2016 perturbation augmentation ([Nat Comms](https://www.nature.com/articles/ncomms13749))

**FALCON baselines** (held-out R²; [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Paper-Datasets_and_Benchmarks_Track.pdf), [EvalAI](https://eval.ai/web/challenges/challenge-page/2319/overview)):

| Method | M1 | M2 | H1 |
|---|---|---|---|
| NoMAD + Wiener filter | 0.49 | 0.20 | 0.13 |
| CycleGAN + Wiener filter | 0.43 | 0.22 | 0.12 |
| NDT2 Multi (few-shot supervised) | 0.59 | 0.43 | 0.52 |

- On H2 (handwriting), CORP reaches a word error rate of 0.11.
- **The motor test-time-adaptation track has no baseline.**

**Test-time adaptation:**
- MPA ([2606.14866](https://arxiv.org/abs/2606.14866))
- Time-masked Transformer with test-time adaptation for speech ([2507.02800](https://arxiv.org/abs/2507.02800))
- TCLA ([2609.27441](https://arxiv.org/abs/2609.27441))

**Verdict:** alignment is crowded. Label-free *estimation* of when a decoder has degraded is open.

## 3. Speech decoding
- Brain-to-Text '24: word error rate 9.7% → 5.8% from ensembles merged by an LLM ([2412.17227](https://arxiv.org/abs/2412.17227)).
- Brain-to-Text '25: 466 teams, baseline 6.7%. The winner's roughly 1.5% comes only from a sponsored article; unverified.
- End-to-end decoding: BIT ([2511.21740](https://arxiv.org/abs/2511.21740)).
- CTC decoders are overconfident: "Probabilistic Co-Control" ([bioRxiv 2026](https://www.biorxiv.org/content/10.64898/2026.04.02.715749v1)).
- OVMI metric ([2609.02887](https://arxiv.org/abs/2609.02887)).

**Verdict: saturated and crowded. Do not target word error rate.**

## 4. Low-power decoding
- NeuroBench primate-reaching leaderboard: frozen since Aug 2024, best R² about 0.71 ([leaderboard](https://github.com/NeuroBench/neurobench/blob/main/leaderboard.rst)).
- Spiking-network decoders: [2409.04428](https://arxiv.org/abs/2409.04428), [2504.11568](https://arxiv.org/abs/2504.11568), Spikachu [2510.20683](https://arxiv.org/abs/2510.20683).
- BrainDistill ([2601.17625](https://arxiv.org/abs/2601.17625)).

**Verdict:** saturated within a session. Efficient decoders tested *across sessions* remain open.

## 5. Robustness, uncertainty and safety: least explored
- arXiv searches for BCI uncertainty quantification return only EEG papers ([2507.07511](https://arxiv.org/abs/2507.07511), [2403.09228](https://arxiv.org/abs/2403.09228)).
- Conformal prediction in neural decoding: only ConformalHDC (hippocampal classification, [2602.21446](https://arxiv.org/abs/2602.21446)).
- Error detection from neural signals: Even-Chen 2017 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/29130452/)).
- Corrupted-channel detection: Vasko 2022 ([Frontiers](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.858377/full)).
- BrainGate: 3 of 14 arrays failed to decode consistently ([medRxiv](https://www.medrxiv.org/content/10.1101/2025.07.02.25330310v1.full)).
- **No intracortical corruption benchmark exists.** EEG has one ([2609.32743](https://arxiv.org/abs/2609.32743)); the image-domain template is ImageNet-C.

## 6. Evaluation methodology
- Leakage: Wong et al. 2026 ([bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.26.701583v2.full)).
- Timing shortcuts: synthetic data decoded at 22.0% vs 22.3% for real data in non-invasive brain-to-text ([2609.40359](https://arxiv.org/abs/2609.40359)).
- RNNs overlearn trial structure (Deo 2024, [Sci Rep](https://www.nature.com/articles/s41598-024-51617-3)).
- **No one has audited intracortical benchmarks for shortcuts.**

## The agent's ranked gaps
1. **Calibrated uncertainty and label-free failure prediction for motor decoders under drift** (scores 5×5×5×5). Data: FALCON M1, M2, H1 and NLB.
2. **FALCON-C: a corruption and electrode-failure benchmark** for decoders and foundation models (5×5×4×5).
3. **Shortcut and leakage audit** of intracortical benchmarks (5×5×5×4).
4. **Plug-in channel re-identification** for frozen pretrained decoders (5×3×4×4).
5. **Uncertainty-gated self-training** for the FALCON motor test-time-adaptation track (4×5×3×4).

Not recommended: new foundation-model pretraining, speech word-error-rate chasing, and spiking-network leaderboard work.
