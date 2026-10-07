#!/usr/bin/env bash
# One analysis lane for new BrainGate participants (same steps, logs and .ok markers as run_new_participants.sh
# analyse). Two lanes can run side by side; before each step a lane waits until at least MIN_FREE_GB of RAM is free,
# so parallel 192-channel jobs cannot exhaust the 24 GB of memory.
#
# Usage: bash scripts/analyse_lane.sh P [P ...] >> results/new_participants/analyse.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=results/new_participants
MIN_FREE_GB=${MIN_FREE_GB:-6}

data_root() { case "$1" in T8) echo G:/ibci-data ;; T11) echo D:/ibci-data ;; *) echo F:/ibci-data ;; esac; }

free_gb() { powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" | tr -dc 0-9; }

step() {   # step LOG ENV COMMAND...
  local log="$1" env="$2"; shift 2
  if [ -f "$log.ok" ]; then echo "skip  $log"; return; fi
  while [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do sleep 60; done
  echo "$(date +%T) start $log"
  if env "$env" "$PY" "$@" > "$log" 2>&1; then touch "$log.ok"; echo "$(date +%T) done  $log"
  else echo "$(date +%T) FAILED $log"; fi
}

for P in "$@"; do
  while [ ! -f "$OUT/convert_$P.done" ]; do sleep 60; done
  R=$(data_root "$P"); E="IBCI_DATA=$R"
  echo "$(date +%T) ==== $P ($(ls "$R/braingate/decoding/$P" | wc -l) sessions)"
  step "$OUT/${P}_ladder.log" "$E" scripts/replicate_braingate_decoding.py --participant "$P" \
       --root "$R/braingate" --skip L4 L5 --resume
  [ -f "results/replication_bg/${P}_ladder.csv" ] || { echo "$(date +%T) no session pairs for $P, skipped"; continue; }
  step "$OUT/${P}_decay_alpha.log" "$E" scripts/decay_alpha.py --subject "$P"
  step "$OUT/${P}_reg_tradeoff.log" "$E" scripts/reg_tradeoff.py --dataset braingate --subject "$P" \
       --out "results/reg_tradeoff_$P"
  step "$OUT/${P}_align_variants.log" "$E" scripts/align_variants.py --subject "$P"
  step "$OUT/${P}_label_free_plus.log" "$E" scripts/label_free_plus.py --subject "$P"
  step "$OUT/${P}_gain_efficiency.log" "$E" scripts/gain_efficiency.py --subject "$P"
  step "$OUT/${P}_reweight_controls.log" "$E" scripts/reweight_controls.py --subject "$P"
  step "$OUT/${P}_gain_mechanism.log" "$E" scripts/gain_mechanism.py --subject "$P"
  step "$OUT/${P}_weight_angles.log" "$E" scripts/weight_angles.py --subject "$P"
  echo "$(date +%T) ==== $P finished"
done
