# 0004 — FALCON head-to-head test: not pursued

Date: 2026-10-09. Status: rejected after a feasibility check.

## Question

Could the paper's "simple recipe" (per-channel or per-block normalization, strong ridge regularization, a short refit
shrunk toward the old decoder) be tested against published drift-robust methods on the FALCON benchmark, to support a
claim that simple corrections match complex methods?

## What the check found

Source: Karpowicz et al., FALCON, NeurIPS 2024 Datasets and Benchmarks, Table 1
(https://proceedings.neurips.cc/paper_files/paper/2024/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Paper-Datasets_and_Benchmarks_Track.pdf);
repository https://github.com/snel-repo/falcon-challenge.

- Held-out calibration data are labelled ("only a small amount of supervised data is released from held-out
  sessions"), and few-shot supervised methods are an official category, so the recipe would be admissible.
- Submissions are Docker containers pushed to EvalAI (6-hour limit), implementing a causal, timestep-by-timestep
  decoder interface (`falcon_challenge.interface`).
- Held-out R² (Table 1): Wiener filter oracle (trained on all held-out data, including unreleased) M1 0.53, M2 0.26,
  H1 0.21; zero-shot Wiener filter 0.34, 0.06, 0.16; NoMAD + WF (few-shot unsupervised) 0.49, 0.20, 0.13; CycleGAN + WF
  0.43, 0.22, 0.12; NDT2 Multi (few-shot supervised) 0.59, 0.43, 0.52. Leaderboard bests noted on 2026-10-05:
  M1 0.664, M2 0.504, H1 0.677 (docs/knowledge/04_public_datasets.md).

## Decision

Not pursued. The recipe is a linear decoder, and the linear oracle bounds what any correction of a linear decoder can
reach (H1 0.21 versus 0.52 for NDT2 few-shot and 0.68 on the leaderboard). The recipe could beat the unsupervised
NoMAD and CycleGAN baselines, but only because it uses labels, which is not a fair comparison. In the fair category,
nonlinear models win by a wide margin, so the test would contradict a "simple beats complex" claim.

## Consequence for the paper

Claims stay scoped to corrections *for a given decoder*: normalization and regularization matter more than the
label-free alignments tested, and a short regularized refit restores most accuracy. The paper does not claim that
simple decoders match complex ones. Its distinct contribution is year-scale, cross-individual measurement of decay and
correction (FALCON spans days to weeks).
