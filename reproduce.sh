#!/usr/bin/env bash
# Reproduce the study on Linux or macOS. Usage:
#   ./reproduce.sh [setup | data | ladders | analyses | electrodes | simulator | summaries | check | all | --list]
# The first run of "./reproduce.sh setup" creates the virtual environment .venv and installs the pinned packages.
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=""; fi

if [ "${1:-}" = "setup" ]; then
  if [ -z "$PY" ]; then python3 -m venv .venv; PY=.venv/bin/python; fi
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt
  "$PY" -m pip install -e .
  echo "Setup finished. Next: ./reproduce.sh data"
  exit 0
fi

if [ -z "$PY" ]; then echo 'The environment is missing. Run "./reproduce.sh setup" first.'; exit 1; fi
if [ $# -eq 0 ]; then "$PY" scripts/reproduce.py --list; else "$PY" scripts/reproduce.py "$@"; fi
