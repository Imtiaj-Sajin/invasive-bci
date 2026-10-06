# Results (draft, 2026-10-06)

*Working draft. Every number comes from `results/` (see `results/README.md`) and is logged in `docs/RESEARCH_LOG.md`.
Medians, with 95% cluster-bootstrap CIs over training sessions unless stated.*

## 1. A conserved overnight drop, then subject-specific decay

We trained ridge decoders on one session and tested them on later sessions after the standard deployment correction:
each channel is re-normalized with the new day's own unlabelled statistics.

**Monkey N** (LINK; 537 session pairs, gaps of 1–900 days):

- The frozen decoder kept 0.73 [0.62, 0.78] of the own-day R² after one day and 0.54 after a week.
- It kept 0.19 after a month and 0.04 after four months.
- It fell below zero beyond a year (−0.36 at 480 days; −0.52 at 900 days).
- Own-day R² stayed constant at 0.29 across the 3.4 years, so the decay reflects a mismatch between the decoder and the day,
  not worse sessions.

**Replication in two monkeys from another laboratory** (DANDI 000688; sorted units summed per electrode; a reaching task):

| Gap | Chewie | Mihili |
| --- | --- | --- |
| 1 day | 0.72 | 0.77 |
| 1 week | 0.56 | 0.61 |
| 1 month | 0.25 | 0.55 |
| 4 months | −0.09 | 0.04 |

**Human participant T6** (BrainGate, 124 closed-loop cursor sessions over 3.1 years; decoding intended movement direction):

| Gap | Retention [95% CI] |
| --- | --- |
| 1 day | **0.73** [0.38, 0.88] |
| 1 week | 0.66 |
| 1 month | 0.71 |
| 4 months | 0.51 |
| 16 months | 0.30 |

- **The overnight drop is conserved across all four subjects** (0.72–0.77).
- **The long-term course is subject-specific.** It is much slower in the human, possibly because of species or array differences, or
  closed-loop intended-direction labels versus measured kinematics (Figure `fig_decay_xsubject` A).

**Nonlinear decoders.** An LSTM decoder was more accurate within a day (+0.128 R² [0.111, 0.141] on matched pairs, higher on every
training session) and modestly more robust in relative terms: retention was +0.07 [0.01, 0.11] higher.

## 2. Corrections that need no labels do not recover the loss

- **Recentring and renormalization** removed the catastrophic offset errors of the uncorrected decoder (median R² −1.6 to −4.7).
- **Alignment beats mean-only recalibration, but not full renormalization.** Label-free subspace alignment improved on mean-only recalibration
  (the baseline in prior work; consistent with Degenhart 2020 and Wilson 2025). It did not improve on per-channel variance
  normalization. For example, at 7 days: mean-only 0.070, Procrustes 0.116, full renormalization 0.148 (monkey N). The same ordering held in the human.
- **Procrustes alignment** of the dominant neural subspace never exceeded plain renormalization in any monkey. Neither did the
  stable-channel stabilizer (Degenhart et al. 2020).
- **Monitoring, offline.** In a pilot, 14 label-free statistics computed on the new day's unlabelled data failed to predict a decoder's
  performance beyond elapsed time. These covered input shift, subspace angles, ensemble and cross-view disagreement, and output
  self-consistency. The failure is not a noise floor: the day-to-day performance left over after the time trend was reliable
  (split-half reliability 0.84). A MINDFUL-style KL score (Pun et al. 2024) behaved the same way: about half of its raw correlation
  with performance was elapsed time, and it gave no cross-validated gain over the calendar.
- **Monitoring, closed loop.** We re-analysed MINDFUL's public closed-loop data.
  - **Neural-feature divergence was largely explained by elapsed time** in T11 (partial r 0.37 vs raw 0.82) and gave no leave-one-day-out gain.
  - **Divergence of the decoder's outputs carried real information beyond time.** Leave-one-day-out error fell from 17.1° to 11.9° (T11)
    and from 32.9° to 15.0° (T5). In closed loop, the user's corrections against a degrading decoder appear in its outputs.
  - **Recommendation:** monitor decoder outputs during use, and report time-partialled statistics.

## 3. Per-channel gain changes explain a minority of the loss; recalibration recipes

- **Per-channel gains.** Re-learning one gain and offset per channel (96 parameters) recovered 14–50% of the drift loss in monkey N.
  Retention with gains was 0.78, 0.43 and 0.22 at 1, 30 and 480 days, with similar values in Chewie and Mihili.
- **In the human, gains recovered most of the loss:** retention 0.81–0.96 at all gaps (`fig_decay_xsubject` B). This is consistent with
  Bishop et al. (2014), who found that same-electrode tuning parameters move together between days. So a 96-parameter gain recalibration
  is an especially efficient fix in this human.
- **The free input remap is not a structural test.** A free 96×96 input remap in front of the frozen decoder can represent any new
  decoder, so it is not reported as evidence for a particular drift structure.
- **Ridge-to-prior recalibration.** We shrank all weights *and the intercept* toward the old decoder. With 100 labelled trials
  (about 3 minutes), it recovered 79–89% of the own-day R² at every gap. With 300 trials it matched the own-day decoder (`fig_efficiency`).
- **Choosing shrinkage from history.** Picking the shrinkage strength from previous sessions (leave-one-session-out) made small-budget
  recalibration safe. Recalibrations worse than doing nothing fell from 10.5% to 1.2% at 10 trials, and from 8.1% to 0% at 20 trials.
- **Re-learning only the top channels.** With ≤ 50 trials, re-learning only the 32 channels the old decoder relies on most was slightly better than a full refit
  (+0.013 to +0.021 R²). Label-free change statistics did not identify better channels than decoder importance did.

## 4. Regularize for the future

- **Same-day vs cross-day.** For ridge decoders, regularization strength barely affected same-day accuracy over four orders of magnitude.
  It strongly affected cross-day accuracy (`fig_reg_tradeoff`).
- **The 1% rule.** We chose the largest α whose same-day R² stayed within 1% of the best, using same-day data only. Gains over the common default
  (α = 0.1) were:
  - monkey N: +0.021 at 1 day, rising to +0.090 at 480 days (all CIs exclude 0);
  - Chewie: +0.008 to +0.032;
  - Mihili: about +0.001;
  - **human T6: +0.008 same-day and +0.011 to +0.025 cross-day, all CIs excluding 0.**

  Same-day accuracy never fell.
- **Why the usual tuning misses it.** Because same-day cross-validation is flat over this range, it gives no reason to choose the more robust
  setting. This explains why small defaults persist.

## 5. A common electrode failure process across species

Data: 3.4 years of monkey N (LINK, with grid positions and impedance), and all 20 arrays of the BrainGate 20-year release (14 participants,
2,289 sessions). An electrode is "active" at ≥ 2 Hz threshold crossings (−4.5 RMS), which is BrainGate's definition.

- **Losses are mostly transient.**
  - 584 of 730 silenced electrodes in humans later revived (0.80).
  - In monkey N, 23 of 31 (0.74).
  - Alive/silent switching rates: median 0.0098 and 0.0065 per day in humans; 0.0031 and 0.0008 in monkey N.
- **Single-electrode changes are abrupt.** Session-to-session log-rate changes were heavy-tailed even after removing session-wide events (excess kurtosis 10.2 in monkey N;
  median 3.6 across human arrays).
- **Edge electrodes lose spiking faster.**
  - Electrodes on the array perimeter showed steeper declines in threshold-crossing activity than interior electrodes:
    14 of 19 arrays (humans plus monkey N) at the point estimate; Fisher-combined p = 2 × 10⁻⁸ across the 20 human arrays.
  - In monkey N the effect held within each array, and was absent for spike-band power. That is consistent with micromotion strain on
    nearby tissue at the edges (Forrest et al. 2025), while broadband changes are array-wide.
- **Impedance falls over years** in 17 of 18 human arrays with measurements (median ρ = −0.92) and in monkey N
  (302 → 171 kΩ over three years) (`fig_failure_xspecies`, `fig_health`).
- **Earlier work.** Hahn et al. (2026) noted edge-related impedance changes qualitatively in the same human data. The electrode-level revival
  statistics and the quantitative edge effect are new.

## 6. A calibrated simulator of drift and electrode failure

*(Draft. Section to be completed with simulator v2.)*

- **Model.** We fitted a feature-level generative model of chronic change. It has three parts:
  - session-to-session and slowly accumulating mixing of channel signals;
  - turnover of channel tuning;
  - alive/silent electrode switching, with rates measured from activity.
- **Calibration.** The model was calibrated so that its own correction ladder matched the real one on monkey N's first 700 days (loss 0.074).
- **Validation on held-out years.** On days ≥ 700 it reproduced rungs and data budgets it was not fitted to, 3–4× more closely than a no-drift
  null (median absolute error 0.06–0.12 vs 0.27–0.39).
- **What validation revealed.** Real decoders decayed more slowly later in the implant's life: at 120 days, retention was 0.18 in years 3+ vs −0.12 in years 1–2.
  The drift process itself depends on implant age.
- **Augmentation.** Training decoders on simulated futures beat ad hoc perturbations (+0.010 R²) but not simple stronger regularization (+0.042). For
  linear decoders, the simulator is therefore a tool for benchmarking and policy evaluation rather than augmentation.
