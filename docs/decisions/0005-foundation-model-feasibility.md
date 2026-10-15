# 0005 Feasibility of testing pretrained foundation-model decoders over years

Status: **proposed, waiting for the owners' go/stop** (2026-10-13). Step 3 of the agreed order (see RESEARCH_LOG).

## Question

Can we test pretrained intracortical "foundation" decoders on our year-scale benchmark (`benchmark/`) and ask how
they decay and how they are best adapted (zero-shot, 10-300 labelled trials, unsupervised), on the owner's PC
(RTX 3060 Ti, 8 GB) or PC2 (RTX 5080, 16 GB)?

## Findings (web and code review, 2026-10-13)

**NDT3** (Ye et al., NeurIPS 2025; https://doi.org/10.52202/085713-3692)
- Public weights: https://huggingface.co/joel99/ndt3 (45M and 350M parameters; licence CC-BY-NC-4.0, verified on the
  Hugging Face API). Code MIT: https://github.com/joel99/ndt3
- Input: threshold-crossing counts in 20 ms bins, 32-channel tokens, at most 320 channels and 2 arrays
  (`config/exp/v5/_default.yaml`; https://github.com/joel99/ndt3/blob/main/model_card.md). No spike-band power input.
  Our data have what it needs: LINK stores threshold crossings, and our compact BrainGate files keep the -4.5 RMS
  crossings (`scripts/tools/compact_braingate_decoding.py`, KEEP_NEURAL).
- Pretraining overlap: the registry loads DANDI 000688 sub-C and the Miller Chewie/Mihili sets
  (`contexts/context_registry.py`, verified that 000688 sub-C is listed). Monkeys C and M are therefore inside
  pretraining and can serve only as an in-pretraining control. Monkey N (FALCON M2 is excluded from pretraining) and
  BrainGate (human pretraining data are Pitt only) look clean [inclusion lists read by the review agent; to be
  re-checked against the exact 200 h and 2000 h configs before any run].
- Compute: 45M fine-tuning fits 8 GB [estimate]; 350M needs the 5080 or low-rank adapters. Code assumes Linux,
  FlashAttention 2, Weights & Biases and Hydra; Windows support [UNVERIFIED], so plan WSL2 or PyTorch attention.
- Relevant: the paper reports strong sensitivity to channel order and permutation (Fig. 4), which suggests electrode
  turnover over years may hurt it more than a linear decoder. Its longest tested shift is about 1 hour (Fig. 5A).

**NDT2** (Ye et al. 2023; https://github.com/joel99/context_general_bci): 6M parameters, human (~100 h) and monkey
checkpoints on Google Drive; FALCON baselines at https://drive.google.com/drive/folders/1ijvmPcbyjHlLEoWWSSZrNm1bBojGgrZG.
NDT3 reports pretrained NDT2 to be weak and fragile. FALCON M2 checkpoints were trained on monkey N; overlap with
LINK dates [UNVERIFIED].

**POYO / POYO+**: official repos (https://huggingface.co/nerdslab/POYO, .../POYO_plus) hold no weights. An unofficial
reproduction (https://huggingface.co/mhough/poyo-mp-reproduction) was trained on 000688 (contaminated for C, M). Input
is spike times with unit identity. **POCO** is for calcium imaging; **POSSM**, **UniBCI**, **iBrain**: no public
weights found.

**Novelty**: no published study found that tracks a pretrained model's decoding over months to years across many
session pairs (closest: NDT3 up to 1 hour; LINK paper with linear and network decoders but no pretrained models;
FALCON held-out sessions of days to months). Search not exhaustive.

## Options

| Model | Verdict | Blockers | Effort |
| --- | --- | --- | --- |
| NDT3-45M | GO (monkey N and six humans; C, M as in-pretraining control) | Threshold crossings only; Linux toolchain | 8-12 days |
| NDT3-350M | Conditional | Needs PC2 (16 GB) or adapters | +3 days |
| NDT2 | Conditional, secondary | Weak per NDT3; M2 checkpoint overlaps monkey N | 3-5 days |
| POYO family, others | Stop | No usable clean weights or wrong modality | - |

## What a GO would involve

1. Add a threshold-crossing feature option to the benchmark (`Task`) and run the ridge reference methods on it, so
   NDT3 is compared with a linear decoder on the same features (and with our spike-band-power results).
2. Environment for NDT3 (WSL2 or PyTorch attention), loaders for LINK and BrainGate.
3. Fine-tune NDT3-45M on day i; test zero-shot on day j; few-shot fine-tuning at 10-300 trials; unsupervised
   adaptation with the neural-only objective on unlabelled day-j data; with and without renormalized inputs.
4. A subset of pairs first (e.g. all gaps, 10 training sessions per individual), then the full set if informative.

## Decision

Pending. Recommendation: GO for NDT3-45M, limited first to a pilot on monkey N and two humans (about 3-4 days) with a
stop rule: continue to the full benchmark only if the pilot runs end to end and the fine-tuned NDT3 matches or beats
the threshold-crossing ridge decoder on same-day data.

## Addendum 2026-10-14: owners' go for the pilot; environment and checkpoints checked

- Owners approved the NDT3-45M pilot ("do the best") and an Ubuntu-24.04 WSL distro on H: for it.
- Environment: `scripts/ndt3/setup_wsl.sh` (Python 3.11, torch 2.1.0+cu118, official FlashAttention 2.6.3 wheel,
  NDT3 @ 877f9f5, mid-2024 pins for packages whose latest releases need torch >= 2.5). FlashAttention runs on the
  3060 Ti and NDT3 imports.
- Pretraining contents read from the checkpoints' own stored context (`hyper_parameters.data_attrs.context`):
  - 45M 200 h (`753jmg4u`, sha256 74e39356...8745): 485 sessions, 15 monkeys, no humans. Includes Chewie, Mihi and
    Perich C/J/M/T.
  - 45M 2 kh (`l2ulo75e`, sha256 b6b8eef2...a2e): 26,298 sessions, 23 subjects; 25,181 Pitt human sessions
    (CRS02b, CRS07, CRS08, BCI02, BCI03, BMI01) plus the monkeys above. No BrainGate (Stanford/MGH/Brown) or Chestek
    monkey-N sessions.
  - Monkeys C and M overlap directly (e.g. Perich_C CO 2013-10-03 is one of our training sessions): excluded from
    the NDT3 comparison except as an explicit in-pretraining control.
- Pilot model: 45M 2 kh (human data in pretraining, clean for monkey N and BrainGate). Pilot individuals: monkey N,
  T5, T8 (threshold crossings decode well; T6 does not).

## Addendum 2026-10-15: pilot result and recommendation

Pilot (`scripts/ndt3/pilot.py`, NDT3-45M-2kh, threshold crossings, learning rate chosen per individual by NDT3's own
validation R2 with a rule fixed beforehand; results/ndt3/pilot/*_lr4e-4.csv). Linear = the paper's ridge decoder on the
same threshold crossings. Medians over 12 pairs (3 training days x up to 4 gaps of 1-480 days).

| | Monkey N | Human T5 |
| --- | --- | --- |
| Same-day R2, NDT3 vs linear | 0.42 vs 0.20 (NDT3 better in 12/12 sessions) | 0.33 vs 0.43 (better in 0/11) |
| No update, R2 (NDT3 vs renormalized linear) | -0.06 vs -0.04 | -0.24 vs -0.19 |
| 50 labelled trials (vs linear refit toward prior) | 0.16 vs 0.08 (NDT3 better 10/12) | 0.06 vs 0.30 (2/12) |
| 300 labelled trials | 0.41 vs 0.18 (12/12); retention 0.97 vs 0.94 | 0.26 vs 0.41 (1/12); retention 0.72 vs 0.91 |
| Neural-loss-only (unsupervised) adaptation | harmful (median retention -0.43) | harmful (-0.16) |

Monkey N gave the same pattern at lr 1e-4 (results/ndt3/pilot/N_lr1e-4.csv). T8: NDT3 does not fit when both arrays are
used (R2 ~0, pretrained or from scratch); T8's first array is nearly silent (17/96 channels active, linear R2 0.02 on
it alone). With the second array only, NDT3 reaches 0.31 on T8 day 45, against 0.38 for the linear decoder on all
channels (results/ndt3/diag_T8.csv). Not a pipeline fault: a near-silent array prevents NDT3 from learning here.

Stop rule (fixed beforehand): continue only if NDT3 matches or beats the linear decoder on same-day data in at least
2 of 3 individuals. Met in monkey N only (T5 no; T8 no, also with the silent array removed). **Recommendation: STOP the
full NDT3 benchmark.** The pilot itself is informative and can be reported as a supplementary analysis: a pretrained
foundation decoder did not decay more slowly than a linear decoder without recalibration, neural-loss adaptation did
not help, and its same-day advantage held in a monkey but not in two human participants. Decision for the owners.
