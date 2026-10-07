#!/usr/bin/env bash
# Run a chosen list of analysis steps for one participant (same logs, .ok markers and data roots as analyse_lane.sh).
# Two calls with disjoint step lists can run side by side once the participant's ladder exists.
#
# Usage: bash scripts/analyse_steps.sh P step [step ...] >> results/new_participants/analyse.log 2>&1
#   steps: decay_alpha reg_tradeoff align_variants label_free_plus gain_efficiency reweight_controls gain_mechanism
#          weight_angles nn_ladder
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=results/new_participants
MIN_FREE_GB=${MIN_FREE_GB:-6}
P="$1"; shift
case "$P" in T8|T9) R=G:/ibci-data ;; T11|T5|T6) R=D:/ibci-data ;; *) R=F:/ibci-data ;; esac

free_gb() { powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" | tr -dc 0-9; }

for s in "$@"; do
  log="$OUT/${P}_$s.log"
  if [ -f "$log.ok" ]; then echo "skip  $log"; continue; fi
  while [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do sleep 60; done
  case "$s" in
    reg_tradeoff) args=(scripts/reg_tradeoff.py --dataset braingate --subject "$P" --out "results/reg_tradeoff_$P") ;;
    *) args=("scripts/$s.py" --subject "$P") ;;
  esac
  echo "$(date +%T) start $log"
  if IBCI_DATA="$R" "$PY" "${args[@]}" > "$log" 2>&1; then touch "$log.ok"; echo "$(date +%T) done  $log"
  else echo "$(date +%T) FAILED $log"; fi
done
echo "$(date +%T) ==== $P steps finished: $*"
