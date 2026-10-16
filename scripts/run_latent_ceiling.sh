#!/usr/bin/env bash
# Latent-alignment ceiling (scripts/latent_ceiling.py) for all nine individuals, one at a time.
# Restartable: a finished individual leaves <log>.ok and is skipped; an interrupted one resumes from its CSV.
# Usage: bash scripts/run_latent_ceiling.sh [SUBJECT ...]
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=results/latent_ceiling
mkdir -p "$OUT"
root() { case "$1" in T8) echo H:/ibci-data ;; T9) echo G:/ibci-data ;; T7|T10) echo F:/ibci-data ;; *) echo D:/ibci-data ;; esac; }
SUBJ="${*:-T7 T10 M N T6 T5 T8 T9 C}"
for k in $SUBJ; do
  log="$OUT/$k.log"
  if [ -f "$log.ok" ]; then echo "skip  $k"; continue; fi
  echo "$(date +%T) start $k"
  if IBCI_DATA="$(root "$k")" "$PY" scripts/latent_ceiling.py --subject "$k" >> "$log" 2>&1; then
    touch "$log.ok"; echo "$(date +%T) done  $k"
  else echo "$(date +%T) FAILED $k"; fi
done
echo "$(date +%T) ==== latent ceiling finished"
