# Results

Result tables, summaries, run logs and figures, produced by the scripts in `../scripts`. The order of the analyses
and the exact command behind each folder are in `../scripts/reproduce.py` and
[../docs/REPRODUCE.md](../docs/REPRODUCE.md). Raw data are never stored here. Individuals: monkeys N, C, M; BrainGate
participants T6, T5, T9, T7, T10, T8 (included) and T2, T3, T11 (excluded by the inclusion rule,
[../docs/INCLUSION_RULES.md](../docs/INCLUSION_RULES.md)).

## Decoder decay and corrections

| Folder | Script | Contents |
| --- | --- | --- |
| `anatomy/` | `drift_anatomy.py` | Correction ladder for monkey N, default ridge penalty (gaps 1 to 900 days). |
| `replication/` | `replicate_perich.py` | Correction ladder for monkeys C and M. |
| `replication_bg/` | `replicate_braingate_decoding.py` | Correction ladder per BrainGate participant (`<P>_ladder.csv`). |
| `replication_bg_blocknorm/`, `decay_alpha_blocknorm/` | same, `IBCI_BLOCKNORM=1` | T11 with block-wise normalization (sensitivity analysis). |
| `decay_alpha/` | `decay_alpha.py` | Correction ladder with the tuned ridge penalty (alpha = 10^4), all nine individuals; the main decoder results. |
| `anatomy_alpha1e4/` | `drift_anatomy.py --base-alpha 1e4` | Monkey N ladder with the tuned penalty, earlier run. |
| `latent_ceiling/` | `latent_ceiling.py`, `ceiling_summary.py` | Best linear alignment of the dominant subspace fitted with labels (alignment ceiling); `summary.json`. |
| `heterogeneity/` | `heterogeneity_model.py` | Mixed model and rank-based tests of differences in decay between individuals. |
| `align_variants/`, `label_free_plus/` | `align_variants.py`, `label_free_plus.py` | Label-free corrections: mean updating, Procrustes rotation, CORAL, factor-analysis stabilizer. |
| `reweight_controls/`, `reweight_controls_pilot/` | `reweight_controls.py` | Re-weighting with scrambled channels and non-negative weights. |
| `gain_mechanism/` | `gain_mechanism.py` | Link between fitted channel weights and changes in firing rate and impedance. |
| `weight_angles/` | `weight_angles.py` | Angles between decoders fitted on different days. |
| `nn_ladder/`, `anatomy_nn/`, `anatomy_matched_nn/` | `nn_ladder.py`, `drift_anatomy_nn.py` | Recurrent-network decoder and its comparison with ridge. |
| `matched_output/`, `matched_output_tuned/` | `matched_output_control.py` | Monkeys C and M decoding a two-dimensional movement direction. |
| `reg_tradeoff*/` | `reg_tradeoff.py`, `rule_summary.py` | Ridge penalty trade-off on the training day and later days; the 1% rule (`reg_tradeoff/rule_across_subjects.csv`). |
| `gain_efficiency/` | `gain_efficiency.py` | Recalibration with 10 to 300 labeled trials in the human participants. |
| `data_efficiency/`, `recal_policy/`, `targeted_recal/` | `drift_anatomy.py --n-trials`, `recal_policy.py`, `targeted_recal.py` | Recalibration with few trials in monkey N; harmful updates; targeted refits. |
| `drift_dof/` | `drift_dof.py` | Corrections with a limited number of free parameters (exploratory). |
| `new_participants/`, `audit_reruns/` | per-participant runs | Console logs of the runs that added T2, T3, T7, T8, T10, T11, and of the reruns after the code review. |

## Electrodes and monitoring

| Folder | Script | Contents |
| --- | --- | --- |
| `braingate_failure/` | `braingate_failure.py` | Yield, switching rates, impedance and edge effect per human array. |
| `failure_robustness/` | `failure_robustness.py`, `failure_null.py` | Silence and recovery under different definitions, with the shuffled-session null. |
| `failure_xdata*/` | `failure_stats_xdata.py` | Electrode failure statistics for LINK (all days; before and after day 700), monkeys C and M, and FALCON H2. |
| `channel_health/` | `channel_health.py` | Monkey N per-channel trajectories over 3.4 years. |
| `mindful_test/`, `mindful_reanalysis/` | `mindful_test.py`, `mindful_reanalysis.py` | Label-free monitoring, offline in monkey N and in the public MINDFUL closed-loop data. |

## Simulator, benchmark and foundation decoder

| Folder | Script | Contents |
| --- | --- | --- |
| `sim_calib_split/`, `sim_calib_late/` | `calibrate_sim.py` | Simulator calibrated on monkey N days before and from day 700. |
| `sim_validation/`, `augment/` | `validate_sim.py`, `sim_augment.py` | Validation on held-out days; training decoders on simulated futures. |
| `bench/` | `bench_from_paper.py`, `bench_run.py` | Benchmark baselines per method and individual; `summary.csv`. |
| `ndt3/` | `scripts/ndt3/` | Pilot of the pretrained NDT3 decoder against the ridge decoder. |

## Summaries and figures

- `v2_summary.json`, `v2_summary_allpairs.json`, `paper_stats.json`, `revision_stats.json`, `decay_summary.csv`,
  `decay_summary_tuned.csv`: summary statistics used in the paper.
- `numbers.tex`: every number quoted in the paper, as LaTeX macros (`make_numbers_tex.py`).
- `supp_tables/`: the supplementary tables (`make_supp_tables.py`).
- `figures/v2/`: the paper figures (`make_v2_figures.py`); other files in `figures/` are earlier versions.
- `archive/`: superseded interim results, kept for the record.
- `*.log`: console output of the corresponding runs.
