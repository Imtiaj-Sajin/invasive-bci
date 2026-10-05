#!/usr/bin/env bash
# Full pipeline for the drift-anatomy / simulator study. Run from the repo root with the project venv.
# Datasets are stored under $IBCI_DATA (default D:/ibci-data). Expect several hours end to end on one GPU.
set -euo pipefail
PY=${PY:-.venv/Scripts/python.exe}

# 1. data
$PY scripts/download_link.py --workers 8                                    # LINK, DANDI 001201 (12.6 GB)
$PY scripts/download_dandi.py --dandiset 000688 --subjects C M --dest perich  # Chewie + Mihili (12.7 GB)
$PY scripts/download_dandi.py --dandiset 000950 --dest falcon_h2              # FALCON H2, human T5 (1.2 GB)

# 2. pilot (label-free monitoring; negative result)
$PY scripts/pilot_link_crossday.py --out results/pilot
$PY scripts/explore_monitor_features.py --out results/explore

# 3. drift anatomy (oracle ladder) and calibration burden
$PY scripts/drift_anatomy.py --max-train 110 --out results/anatomy
$PY scripts/drift_anatomy.py --gaps 1 7 30 120 480 --n-trials 10 20 50 100 300 --max-train 30 --out results/data_efficiency
$PY scripts/remap_locality.py --max-train 40 --out results/locality

# 4. electrode failure process
$PY scripts/channel_health.py --out results/channel_health

# 5. simulator: calibrate on the first part of LINK, evaluate on later days
$PY scripts/calibrate_sim.py --ladder results/anatomy/ladder.csv --max-day 700 --out results/sim_calib_split
$PY scripts/sim_augment.py --calib results/sim_calib_split/calibration.json --split-day 700 --out results/augment

# 6. figures
$PY scripts/make_figures.py anatomy --ladder results/anatomy/ladder.csv
$PY scripts/make_figures.py efficiency --ladder results/data_efficiency/ladder.csv
$PY scripts/make_figures.py health --table results/channel_health/channel_day.csv

# 7. tests
$PY -m pytest tests -q
