# Knowing when an intracortical BCI decoder fails

Label-free reliability monitoring of intracortical brain-computer interface (iBCI) decoders across years of recordings.

**Status:** research in progress, started 2026-10-05. The topic and its rationale are in
[docs/decisions/0001-research-topic.md](docs/decisions/0001-research-topic.md).

## The problem
Implanted BCI decoders degrade as neural signals drift and electrodes age. Today the only way to know whether a decoder still
works is to make the user run a labelled calibration session. This project builds and benchmarks methods that estimate a
decoder's performance on a new day from **unlabelled** neural data. It tests them on up to 3.5 years of public intracortical
recordings, asks whether they can reduce the recalibration burden and detect electrode faults, and releases them as an
open toolkit with a year-scale benchmark.

## Repository map
- [CLAUDE.md](CLAUDE.md): orientation for AI agents and new contributors (conventions, compute environment).
- [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md): dated log of everything done and found.
- [docs/knowledge/](docs/knowledge/): literature and dataset notes with sources.
- [docs/decisions/](docs/decisions/): decision records.
- `src/ibci/`: library code. `scripts/`: runnable entry points. `results/`: small result artefacts.

## Setup (Windows, as used here)
```bash
python -m venv --system-site-packages .venv      # reuses the system CUDA torch
.venv/Scripts/python -m pip install pynwb dandi
export IBCI_DATA=D:/ibci-data                     # where datasets are stored (outside the repo)
.venv/Scripts/python scripts/download_link.py     # LINK, DANDI 001201, about 12.6 GB
```

## Data
- **LINK** (DANDI 001201, CC-BY-4.0): Temmar et al., NeurIPS 2025 Datasets & Benchmarks.
- **DANDI 000688**: Perich/Miller long-term monkey reaching.
- **FALCON H2** (DANDI 000950): Karpowicz et al., NeurIPS 2024 Datasets & Benchmarks.

Please cite the original dataset papers when you use them.
