#!/usr/bin/env bash
# Lane B of scripts/run_audit_reruns.sh as a shared queue, so several workers can run it at once. Each step is claimed
# with an atomic lock directory (<log>.lock); finished steps have <log>.ok (same markers as run_audit_reruns.sh, so
# finished work is never repeated). A failed step releases its lock and is reported in the queue log.
#
# Usage (one line per worker): MIN_FREE_GB=3 bash scripts/run_audit_queue.sh W1 >> results/audit_reruns/queue.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=results/audit_reruns
MIN_FREE_GB=${MIN_FREE_GB:-6}
WORKER=${1:-W}
mkdir -p "$OUT"
ALL="N C M T6 T5 T9 T7 T10 T8"
HUM="T6 T5 T9 T7 T10 T8"

root() { case "$1" in T8) echo H:/ibci-data ;; T9) echo G:/ibci-data ;; T7|T10) echo F:/ibci-data ;; *) echo D:/ibci-data ;; esac; }
free_gb() { powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" | tr -dc 0-9; }

STEPS=()
for k in $ALL; do STEPS+=("reweight_controls|$k|scripts/reweight_controls.py --subject $k"); done
for k in $HUM; do STEPS+=("gain_mechanism|$k|scripts/gain_mechanism.py --subject $k"); done
for k in $ALL; do STEPS+=("align_variants|$k|scripts/align_variants.py --subject $k"); done
for k in $ALL; do STEPS+=("label_free_plus|$k|scripts/label_free_plus.py --subject $k"); done
for k in $HUM; do STEPS+=("gain_efficiency|$k|scripts/gain_efficiency.py --subject $k"); done
for k in C M; do STEPS+=("matched_output|$k|scripts/matched_output_control.py --subject $k --alpha 1e4 --out results/matched_output_tuned"); done
STEPS+=("recal_policy|N|scripts/recal_policy.py")

for s in "${STEPS[@]}"; do
  IFS='|' read -r name k cmd <<< "$s"
  log="$OUT/${name}_$k.log"
  [ -f "$log.ok" ] && continue
  mkdir "$log.lock" 2>/dev/null || continue          # another worker has it
  while [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do sleep 60; done
  echo "$(date +%T) $WORKER start $log"
  # shellcheck disable=SC2086
  if IBCI_DATA="$(root "$k")" "$PY" $cmd > "$log" 2>&1; then touch "$log.ok"; echo "$(date +%T) $WORKER done  $log"
  else echo "$(date +%T) $WORKER FAILED $log"; fi
  rmdir "$log.lock"
done
echo "$(date +%T) ==== queue worker $WORKER finished"
