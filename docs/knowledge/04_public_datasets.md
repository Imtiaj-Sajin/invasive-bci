# Public intracortical and invasive datasets: inventory (as of 2026-10-05)

*Compiled by a research sub-agent from about 20 page fetches and about 70 direct queries to the DANDI, Dryad, Zenodo,
EvalAI, OpenAlex, S2 and GitHub APIs. Sizes marked (API) are exact decimal GB. Citation counts are OpenAlex (OA) or
Semantic Scholar (S2) as of 2026-10-05.*

## Headline findings
1. **BrainGate 20-year Utah-array release.** Dryad [doi:10.5061/dryad.x0k6djj1h](https://doi.org/10.5061/dryad.x0k6djj1h),
   released 2026-09-09; Hahn et al., *Nat Med* 2026, doi 10.1038/s41591-026-04530-3.
   - 14 human participants and 20 arrays; 2,289 array-yield sessions and **729 closed-loop cursor decoding sessions**.
   - Up to **7.6 years** per participant (mean 2.8).
   - 84.7 GB in total, split per participant: yield tarballs 5.30 GB; decoding data T11 21.2, T8 14.2,
     T9 11.7, T5 11.3, T6 6.9, T10 4.6, T7 4.4, T2 4.4, T3 0.6 GB. Format .mat, licence CC0.
   - Code: github.com/nptl-stanford/array-paper. **Not yet benchmarked by anyone.**
2. **FALCON** (NeurIPS 2024 D&B, S2 about 20 citations). Leaderboard open with no end date, with new teams in Sep 2026.
   M2, H2 and B1 remain far from solved.
3. **NLB'21 is saturated.** MC_Maze top co-bps 0.386 (Apr 2022); EvalAI shows the challenge as closed (2026-01-31).

## A. Benchmarks
| Dataset | Subj | Array | Task | Sessions / span | Host | Size GB | Lic |
|---|---|---|---|---|---|---|---|
| NLB MC_Maze (+L/M/S) | NHP | 2× Utah | Maze reach | 1 each | DANDI 000128/138/139/140 | 0.69/0.15/0.08/0.03 | CC-BY |
| NLB MC_RTT | NHP Indy | Utah M1 | Random-target reach | 1 | DANDI 000129 | 0.05 | CC-BY |
| NLB Area2_Bump | NHP | Utah area 2 | Reach + bump | 1 | DANDI 000127 | 1.82 | CC-BY |
| FALCON M1-A/B | 2 NHP | Utah 64/96 + EMG | Reach-to-grasp, EMG | A: 7 sessions / 30 d | DANDI 000941 / 001209 | 0.31/0.23 | CC-BY |
| FALCON M2 | NHP Monkey N (same monkey as LINK) | Utah 96 | 2-DoF finger | 8 sessions / 36 d | DANDI 000953 | 15.9 | CC-BY |
| FALCON H1 | Human (Pitt) | 2× Utah, 172 ch | 7-DoF open-loop reach | 13 sessions | DANDI 000954 | 0.10 | CC-BY |
| FALCON H2 | Human T5 | 192 ch | Handwriting | 26 sessions over about 17 months **LONG** | DANDI 000950 | 1.23 | CC-BY |
| FALCON B1 | Zebra finch | Neuropixels | Song | 6 sessions / 9 d | DANDI 001046 | 1.29 | CC-BY |
| Brain-to-Text '24 | Human T12 | 4×64 Utah | Speech | about 24 sessions over about 4 months | Dryad dryad.x69p8czpq | 3.67 competition subset (80 total; LMs 14–38) | CC0 |
| **Brain-to-Text '25** | Human T15 | 256 electrodes, 512 features | Speech | **45 sessions over 20 months** **LONG** | Dryad dryad.dncjsxm85 / Kaggle | 11.6 | CC0 |
| Willett 2021 handwriting | T5 | 2×96 | Handwriting | about 10 sessions | Dryad dryad.wh70rxwmv | 1.41 | CC0 |
| CORP handwriting | T5 | 192 ch | Handwriting | 21 sessions over about 1 year **LONG** | Dryad dryad.hqbzkh1p6 | 3.57 | CC0 |

## B. Long-term and other NHP datasets
| Dataset | Span | Host | Size GB |
|---|---|---|---|
| **LINK** (Monkey N, 96 ch, threshold crossings + spike-band power, finger, **per-session impedance**) | **312 sessions on 303 days over 1,242 days** (2020-01-27 to 2023-06-22) | DANDI 001201 | 12.56 |
| **Perich/Miller long-term** | **Chewie: 68 sessions, 2013-10 to 2016-10 (about 3 years, 9.74 GB)**. Mihili: 28 sessions over about 1.5 years (2.92 GB). MrT: 12 sessions over 3 weeks. | DANDI 000688 | 13.18 |
| O'Doherty/Sabes (Indy/Loco) | Indy: 37 sessions over about 10 months (10.7 GB). Loco: 10 sessions over 3 weeks. | Zenodo 3854034 | 24.0 |
| O'Doherty broadband (24.4 kHz) | About 30 sessions | One Zenodo record per session | 1.1–8.9 each |
| Ma/Miller ADAN | Multi-day, 6 monkeys, 2012–2021 | Dryad dryad.cvdncjt7n | 15.4 |
| Churchland/Shenoy | Jenkins and Nitschke | DANDI 000070 | 53.3 |
| Even-Chen | 12 sessions | DANDI 000121 | 39.5 |
| NeuroTask (repackaged) | 19 subjects | DANDI 001055–001060, 001078 | about 10.2 |
| NoMAD | 2 sessions | Dryad dryad.q83bk3jtp | 0.25 |

## C. Human intracortical datasets
| Dataset | Span | Host | Size GB |
|---|---|---|---|
| **BrainGate 20-year** | Up to 7.6 years, 14 participants | Dryad dryad.x0k6djj1h | 84.7 (per participant) |
| MINDFUL (Pun 2024) | 142 d and 28 d, fixed decoder | Dryad dryad.n2z34tn5s | 0.41 |
| Wilson unsupervised recalibration | T5, multi-year | Dryad dryad.1jwstqk6g | 1.69 |
| Inner speech (Kunz 2025) | 4 participants | Dryad dryad.gf1vhhn1j | 10.8 |
| Voice synthesis (Wairagkar 2025) | T15 | Dryad dryad.2280gb64f | 26.9 |
| Others (typing, drag-and-drop, gestures, bimanual cursors) | — | Dryad | 0.1–38 |

## D. ECoG/sEEG (brief)
- Natraj/Ganguly long-term ECoG: DANDI 001535, 4.8 GB.
- AJILE12: DANDI 000055, 846 GB.
- Brain Treebank: about 130 GB.
- Du-IN: 93 GB.
- Podcast ECoG: OpenNeuro ds005574.

## E. Other
- **Neuralink compression challenge data:** the official zip now returns 404.
- **brainsets** ([GitHub](https://github.com/neuro-galaxy/brainsets)) wraps perich_miller, odoherty_sabes, churchland_shenoy, flint and NLB.

## Leaderboards (2026-10-05)
- **FALCON**
  - H1 best 0.677 (beats the NDT2 oracle at 0.629).
  - M1 best non-oracle 0.664 (SPINT).
  - M2 best 0.504 against an oracle of 0.585.
  - H2 best WER 7.4% against an oracle of 3.4%.
- **Brain-to-Text '24:** best 3.34% (LENS DeepEn, Sep 2026).
- **Brain-to-Text '25:** winner about 1.53% [UNVERIFIED].
- **NLB:** frozen since 2022.

## Analysis
- **Longest spans:**
  1. BrainGate 20-year (7.6 years, human).
  2. LINK (3.4 years, densest sampling).
  3. 000688 Chewie (about 3 years).
  4. T15 (20 months).
  5. FALCON H2/CORP (about 17 months).
  6. Indy (10 months).
- **Open problems:**
  - FALCON M2/H2/B1, and unsupervised adaptation on all FALCON tasks.
  - Any year-scale benchmark (LINK and BrainGate-20 have none).
  - Predicting yield or decoding SNR.
- **Gaps the field complains about:**
  - No year-scale benchmark.
  - Tiny human datasets.
  - Raw broadband rarely released.
  - Home-use data withheld.
  - Evaluation mostly offline.
  - Language-model RAM needs.

## Machine constraints
- The Brain-to-Text 3-gram LM needs about 60 GB RAM and the 5-gram about 300 GB, so neither fits. Use the 1-gram model or LightBeam.
- OPT-6.7B needs 4-bit quantization.
- Dryad file downloads returned 401 through the API without a token; use the web interface or a direct link.
