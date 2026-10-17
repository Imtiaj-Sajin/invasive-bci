# Reproducing the study

This guide rebuilds every result, figure and table from the public datasets. All steps are scripted; the commands
below use `./reproduce.sh` (Linux, macOS, Git Bash). On Windows, use `reproduce.bat` with the same arguments.

## 1. Requirements

| | Tested | Notes |
| --- | --- | --- |
| Operating system | Windows 11 | Linux and macOS work with `reproduce.sh`. |
| Python | 3.11.3 | Package versions are pinned in `requirements.txt`. |
| Memory | 24 GB RAM | Monkeys C and M and participant T8 need the most memory; run one analysis at a time. |
| GPU | NVIDIA, 8 GB | Only for the recurrent-network decoder (`nn_ladder.py`); everything else runs on the CPU. |
| Disk | about 150 GB free | Roughly 100 GB of downloads; the converted BrainGate files are much smaller than the archives. |

## 2. Install

```bash
git clone https://github.com/Imtiaj-Sajin/invasive-bci.git
cd invasive-bci
./reproduce.sh setup
```

This creates a virtual environment in `.venv/`, installs the pinned packages and installs the `ibci` package in
editable mode. Install the PyTorch build that matches your GPU from <https://pytorch.org> if the default does not.

## 3. Data

### Where the data go

By default every dataset is stored under `data/` in the repository. Two options change this (see `src/ibci/paths.py`):

- set the environment variable `IBCI_DATA` to another folder, or
- create `data_roots.json` in the repository root with one folder per dataset or participant, for example
  `{"default": "D:/ibci-data", "T8": "E:/ibci-data"}` when the data are spread over several drives.

Expected layout inside each data folder:

```text
link/                       LINK NWB files (monkey N)            link_cache/      converted sessions (created)
perich/                     DANDI 000688 NWB files (monkeys C, M) perich_cache/    converted sessions (created)
braingate/decoding/<P>/     compact decoding sessions per participant (T2, T3, T5-T11)
braingate/yield/<P>/        yield files (threshold-crossing rates, impedance) per participant
mindful/MINDFUL_Data/       MINDFUL closed-loop data
```

### Downloads

| Dataset | Source and licence | Size | How |
| --- | --- | --- | --- |
| LINK, monkey N | DANDI 001201, CC-BY-4.0 | 12.6 GB | automatic (`./reproduce.sh data`) |
| Long-term reaching, monkeys C and M | DANDI 000688, CC-BY-4.0 | about 13 GB | automatic (`./reproduce.sh data`) |
| BrainGate yield data, 14 participants | Dryad [10.5061/dryad.x0k6djj1h](https://doi.org/10.5061/dryad.x0k6djj1h), CC0 | 5.3 GB | browser download |
| BrainGate decoding data, 9 participants | same Dryad record | 0.6 to 21 GB per participant | browser download, then conversion |
| MINDFUL closed-loop data | Dryad [10.5061/dryad.n2z34tn5s](https://doi.org/10.5061/dryad.n2z34tn5s), CC0 | 0.4 GB | browser download |

Dryad files need a browser. Save them in `<data>/downloads/`, then convert each decoding archive to compact session
files (the archive is read as a stream, so it can be deleted afterwards):

```bash
python scripts/tools/compact_braingate_decoding.py <data>/downloads/decoding_T6.tar.gz <data>/braingate/decoding/T6
```

Extract the yield archives to `<data>/braingate/yield/<participant>/` and the MINDFUL archive to
`<data>/mindful/MINDFUL_Data/`.

## 4. Run the analyses

Each stage is one command and writes its results to `results/`. Console output goes to `logs/<stage>/`.
`./reproduce.sh <stage> --dry-run` lists the commands without running them, `--skip-existing` resumes after an
interruption, and `--subjects T6 T5` limits per-individual steps.

| Stage | Command | What it computes | Approximate time (6-core PC) |
| --- | --- | --- | --- |
| Correction ladders | `./reproduce.sh ladders` | Decoders trained on one day and tested on later days after each correction, default ridge penalty, all individuals (also the excluded T2, T3, T11) | 1 to 2 days |
| Main analyses | `./reproduce.sh analyses` | Tuned penalty, alignment ceiling, channel controls, label-free alignments, decoder angles, recurrent network, regularization trade-off, recalibration with few trials, matched-output control | 2 to 3 days |
| Electrodes and monitoring | `./reproduce.sh electrodes` | Electrode silence and recovery with a permutation null, yield, impedance, edge effect, spike amplitude, MINDFUL re-analysis | several hours |
| Simulator | `./reproduce.sh simulator` | Electrode failure rates, simulator calibration on early and late implant periods, validation, augmentation | several hours |
| Summaries and figures | `./reproduce.sh summaries` | All statistics, the mixed model, benchmark baselines, `results/numbers.tex` (every number quoted in the paper), supplementary tables and figures | a few minutes |
| Checks | `./reproduce.sh check` | Unit tests; the benchmark API reproduces the paper's values to numerical precision | about 30 minutes |

`./reproduce.sh all` runs everything in order. The results of the full run are already in `results/`, so
`./reproduce.sh summaries` rebuilds all statistics and figures without any data.

All random choices use fixed seeds, and the pinned package versions matter: on numerically unstable sessions a
different numpy version can change results in the last digits.

## 5. Where each figure and table comes from

Figures are written to `results/figures/v2/` by `scripts/make_v2_figures.py` (`--only figN` rebuilds one figure).

| Figure | File | Main inputs |
| --- | --- | --- |
| 1 Overview | `fig1_overview` | BrainGate yield and decoding data of T5, brain rendering (`scripts/render_brain.py`, optional packages) |
| 2 Datasets and correction ladder | `fig2_design` | session lists of all datasets |
| 3 Decoder decay | `fig3_decay` | `results/replication*`, `results/anatomy`, `results/decay_alpha`, `results/decay_summary*.csv` |
| 4 Label-free corrections and monitoring | `fig4_labelfree` | `results/align_variants`, `results/label_free_plus`, `results/mindful_test` (offline), `results/mindful_reanalysis` (closed loop) |
| 5 Regularization and recalibration | `fig5_recipes` | `results/reg_tradeoff*`, `results/gain_efficiency`, `results/recal_policy` |
| 6 Channel tuning | `fig6_channels` | `results/decay_alpha`, `results/reweight_controls`, `results/gain_mechanism`, `results/nn_ladder` |
| 7 Electrodes | `fig7_electrodes` | `results/braingate_failure`, `results/failure_robustness`, `results/channel_health` |
| 8 Simulator | `fig8_simulator` | `results/sim_calib_split`, `results/sim_calib_late`, `results/sim_validation` |

Supplementary tables are written to `results/supp_tables/` by `scripts/make_supp_tables.py`, and every number quoted
in the paper to `results/numbers.tex` by `scripts/make_numbers_tex.py`. The alignment ceiling is summarized in
`results/latent_ceiling/summary.json` and the mixed model in `results/heterogeneity/summary.json`.

## 6. Benchmark

The benchmark is defined by the session pairs in `benchmark/manifests/` and the API in `src/ibci/bench/`. To score a
new method:

```bash
python scripts/bench_run.py --plugin my_module:my_method --subjects T6 T5
```

Rules, metrics and baseline scores are in [benchmark/README.md](../benchmark/README.md).

## 7. Foundation-decoder pilot (optional)

The NDT3 pilot needs Linux (or WSL on Windows), an NVIDIA GPU and NDT3's own environment. `scripts/ndt3/setup_wsl.sh`
creates that environment, `scripts/ndt3/export_pilot.py` exports the pilot sessions, `scripts/ndt3/pilot.py` runs
fine-tuning and evaluation, and `scripts/ndt3/summarize_pilot.py` writes `results/ndt3/pilot_summary.json`. Its
results are already included in `results/ndt3/`.

## 8. Troubleshooting

- **A dataset is not found.** Check `python -c "from ibci.paths import data_root; print(data_root('T6'))"` and the
  folder layout in section 3.
- **Out of memory.** Run per individual (`--subjects C`), and close other programs; monkeys C and M and T8 are the
  largest.
- **A step failed or was interrupted.** Rerun the same stage with `--skip-existing`; the ladder scripts also resume
  from their partial output.
