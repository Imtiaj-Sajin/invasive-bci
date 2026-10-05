# Anatomy of chronic neural drift in intracortical BCIs

What changes in chronic Utah-array recordings over days to years, which of those changes break BCI decoders, and a
**data-calibrated simulator** of drift and electrode failure for building decoders that survive it.

**Status:** research in progress (started 2026-10-05).

- Topic and rationale: [docs/decisions/0002](docs/decisions/0002-pivot-chronic-drift-anatomy-and-simulator.md),
  which supersedes [0001](docs/decisions/0001-research-topic.md).
- Paper plan: [docs/paper/OUTLINE.md](docs/paper/OUTLINE.md).
- Everything done and found, with dates: [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).

## Findings so far (LINK, monkey N, 312 sessions over 3.4 years; details and caveats in the log)

- **Label-free monitoring does not beat the calendar under natural drift.**
  - Day-to-day decoder performance varies reliably beyond the time trend (split-half reliability 0.84).
  - None of 14 label-free statistics captures that variation (pilot, 2026-10-05).
- **Drift anatomy (interim):**
  - With daily renormalization, a fixed ridge decoder loses about 30% of R² overnight.
  - Per-channel gain changes explain little of the loss.
  - A full linear input remap in front of the frozen decoder recovers about 60–75%.
  - At gaps over a year, the decoder itself must change.
  - Ridge shrunk toward the previous decoder is the most data-efficient recalibration.
- **Electrode failure process:**
  - Impedance falls by about 45% over 3 years (302 → 171 kΩ). Active channels halve (36 → 18).
  - Most channel losses are **transient**. Alive/silent switching rates are 0.0031/day off and 0.0008/day on.
  - Abrupt single-electrode changes are common (heavy tails survive removal of session-wide events).
  - Spiking activity declines faster at **array edges** in both arrays, while spike-band power declines uniformly.

## Repository map

- [CLAUDE.md](CLAUDE.md): orientation for AI agents and contributors (conventions, compute environment, commit rules).
- [docs/knowledge/](docs/knowledge/): source-backed notes on the field, ML methods, electrodes, datasets and existing code.
- `src/ibci/`: the library.
  - `data/` holds loaders for LINK, DANDI 000688 and FALCON H2.
  - `linear.py`: decoders.
  - `anatomy.py`: the oracle ladder.
  - `sim.py`: the simulator.
  - `stats.py` and `plotting.py`: statistics and figure style.
- `scripts/`: entry points. `scripts/run_all.sh` lists the full pipeline in order.
- `results/`: small result tables and figures. `tests/`: unit tests (`python -m pytest tests -q`).

## Setup (Windows, as used here)

```bash
python -m venv --system-site-packages .venv      # reuses the system CUDA torch
.venv/Scripts/python -m pip install pynwb dandi pytest
export IBCI_DATA=D:/ibci-data                     # datasets live outside the repo
.venv/Scripts/python scripts/download_link.py     # LINK, DANDI 001201, about 12.6 GB
```

## Data (please cite the original papers)

- **LINK**, DANDI 001201 (CC-BY-4.0): Temmar et al., NeurIPS 2025 Datasets & Benchmarks.
- **DANDI 000688**: Perich/Miller long-term monkey reaching (Chewie, Mihili).
- **FALCON H2**, DANDI 000950: Karpowicz et al., NeurIPS 2024 Datasets & Benchmarks (human T5).
