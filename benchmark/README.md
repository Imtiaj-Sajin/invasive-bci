# Year-scale decoder transfer benchmark for intracortical BCIs

This benchmark asks one question: **how well does a decoder trained on one day still work days to years later, and
how much does a given update method restore?** It uses public recordings from three monkeys and six BrainGate
participants. The splits are the session pairs of our paper (Sajin, Abha and Suva, "Decoder decay and its correction
over years in intracortical brain-computer interfaces", in preparation).

## The task

For each pair of sessions (day *i*, day *j*), with *j* 1 to about 480 days after *i*:

| Allowed input | Arrays in `Task` |
| --- | --- |
| Day *i*: neural features and kinematics of the training segment | `src_x`, `src_y`, `src_trial` |
| Day *j*: neural features of the training segment, without labels | `cal_x`, `cal_trial` |
| Day *j*: kinematics of the first *n* trials only (*n* = 0, 10, 20, 50, 100 or 300) | `cal_y_lab` on the bins `cal_lab` |
| Day *j*: neural features of the evaluation segment (causal use only) | `eval_x` |

A method returns predicted kinematics for the day-*j* evaluation segment. The evaluation labels are never exposed;
`Task.score` computes the result.

*n* = 0 is the label-free setting. The training segment is the first 80% of trials (BrainGate, monkeys C and M) or the
first 300 trials (monkey N). The rest of the session is the evaluation segment.

## Metrics

- **R²** on the evaluation segment, averaged over output dimensions.
- **Retention** = R² / reference R². The reference is a same-day decoder trained on all day-*j* training trials (ridge
  regression, 8 causal lags of 20 ms, penalty 10⁴). Retention 1 means the update is as good as a fresh decoder.
- Summaries report, per individual and gap, the median retention with a 95% bootstrap interval that resamples
  **training sessions**. Pairs that share a training session are not independent.

A pair counts only if its reference R² is above 0.1 (column `valid`). Below that, retention is not meaningful.

## Splits

`manifests/<subject>.csv` lists the fixed pairs: one per training session and target gap (1, 7, 30, 120 and 480 days;
the closest later session of the same task within 25% of the gap; 1-day pairs may be 1 or 2 days apart).

| Individual | Recording | Outputs | Pairs | Valid | Training sessions |
| --- | --- | --- | --- | --- | --- |
| Monkey N | 96-channel Utah array, motor cortex | finger position and velocity (4) | 259 | 259 | 80 |
| Monkey C | 96-channel Utah array, motor cortex | cursor position and velocity (4) | 145 | 130 | 64 |
| Monkey M | 2 × 96-channel Utah arrays, motor and premotor cortex | cursor position and velocity (4) | 44 | 43 | 22 |
| Human T6 | 96-channel Utah array | cursor-to-target direction (2) | 369 | 333 | 122 |
| Human T5 | 2 × 96-channel Utah arrays | cursor-to-target direction (2) | 256 | 247 | 89 |
| Human T9 | 2 × 96-channel Utah arrays | cursor-to-target direction (2) | 361 | 320 | 114 |
| Human T7 | 2 × 96-channel Utah arrays | cursor-to-target direction (2) | 84 | 73 | 33 |
| Human T10 | 96-channel Utah array (the dPCG array in the release) | cursor-to-target direction (2) | 138 | 124 | 53 |
| Human T8 | 2 × 96-channel Utah arrays | cursor-to-target direction (2) | 168 | 155 | 55 |

Features are 20 ms bins of spike-band power (monkey N and the BrainGate participants) or of spike counts summed over
the sorted units of each electrode (monkeys C and M; the release has no spike-band power). Three other BrainGate participants (T2, T3, T11) fail the reference rule
and are left out; see [docs/INCLUSION_RULES.md](../docs/INCLUSION_RULES.md).

## Getting the data

| Data | Source | Licence | Download |
| --- | --- | --- | --- |
| Monkey N | LINK, DANDI [001201](https://dandiarchive.org/dandiset/001201) | CC-BY-4.0 | `python scripts/download_link.py` |
| Monkeys C and M | Perich and Miller, DANDI [000688](https://dandiarchive.org/dandiset/000688) | CC-BY-4.0 | `python scripts/download_dandi.py --dandiset 000688 --subjects C M --dest perich` |
| BrainGate T5, T6, T7, T8, T9, T10 | Hahn et al. 2026, Dryad [10.5061/dryad.x0k6djj1h](https://doi.org/10.5061/dryad.x0k6djj1h) | CC0 | Download `decoding_<P>.tar.gz` in a browser (Dryad blocks automated downloads), then `python scripts/tools/compact_braingate_decoding.py decoding_<P>.tar.gz <data>/braingate/decoding/<P>` |

Data go to `data/` in the repository unless `IBCI_DATA` or `data_roots.json` says otherwise (see
[docs/REPRODUCE.md](../docs/REPRODUCE.md)).

## Using it

```python
from ibci import bench

rows = []
for t in bench.tasks("T6", n_labels=50):
    pred = my_method(t)            # array of shape (len(t.eval_x), t.n_outputs)
    rows.append(t.score(pred, method="mine"))
print(bench.summarize(rows))
```

Or run a method from the command line:

```
python scripts/bench_run.py --plugin mypackage.mymodule:predict --name mine --budgets 0 50 300
python scripts/bench_run.py --summary
```

## Reference methods

`src/ibci/bench/baselines.py` holds the paper's correction ladder:

| Method | Labels | What it does |
| --- | --- | --- |
| `none` | 0 | Day-*i* decoder, unchanged |
| `mean_update` | 0 | Feature means re-estimated on day *j* |
| `renorm` | 0 | Feature means and scales re-estimated on day *j* |
| `procrustes` | 0 | `renorm` plus an unsupervised rotation of the top-16 neural subspace |
| `reweight` | *n* | `renorm` plus one gain per channel, shrunk toward "no change" by cross-validation |
| `prior_refit` | *n* | Ridge decoder refitted on the labelled trials, shrunk toward the day-*i* decoder |
| `retrain` | *n* | New ridge decoder on the labelled trials only |

`python scripts/bench_check.py` confirms that these methods reproduce the paper's numbers (largest difference below
1e-14 on every individual checked).

## Baseline results (v1)

Median across the nine individuals of each individual's median retention on its valid pairs, by gap (per-individual medians with training-session bootstrap
intervals are in `results/bench/summary.csv`; per-pair scores in `results/bench/<method>/<subject>.csv`). Built with
`python scripts/bench_from_paper.py` from the paper's tuned-penalty ladders, which are these methods on these pairs.
Budgets of 10-100 labelled trials on all pairs are planned for the next version.

| Method | Labels | 1 d | 7 d | 30 d | 120 d | 480 d |
| --- | --- | --- | --- | --- | --- | --- |
| `none` | 0 | -2.06 | -2.73 | -5.49 | -8.21 | -4.38 |
| `mean_update` | 0 | 0.18 | -0.33 | -0.98 | -1.03 | -0.97 |
| `renorm` | 0 | 0.73 | 0.52 | 0.29 | 0.09 | -0.06 |
| `procrustes` | 0 | 0.68 | 0.52 | 0.28 | -0.04 | -0.25 |
| `reweight` | 300 | 0.83 | 0.80 | 0.73 | 0.72 | 0.64 |
| `prior_refit` | 300 | 0.97 | 0.98 | 0.98 | 0.96 | 0.98 |
| `retrain` | 300 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 |

## Rules for fair comparison

1. Use only the inputs listed above. Do not use labels beyond the first *n* day-*j* trials, or any day-*j* evaluation
   labels.
2. Process `eval_x` causally (no future bins) if the method is meant for online use; say so if it is not.
3. Report all individuals and all gaps, valid pairs only, with the training-session bootstrap.
4. Report results at every label budget your method can use.
