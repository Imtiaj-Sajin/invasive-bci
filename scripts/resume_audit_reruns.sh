#!/usr/bin/env bash
# Restart the audit reruns after a power cut or reboot. Finished steps (<log>.ok) are skipped; unfinished steps start
# again from the beginning (their partial outputs are overwritten). Refuses to run while any rerun is still active.
#
# Usage: bash scripts/resume_audit_reruns.sh
set -u
cd "$(dirname "$0")/.."
OUT=results/audit_reruns
if powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'run_audit_(reruns|queue)|retry_failed_reruns\.sh' }" | grep -q .; then
  echo "reruns are still running; nothing to do"; exit 1
fi
rmdir "$OUT"/*.lock 2>/dev/null          # stale claims from workers that died
stamp=$(date +%Y%m%d_%H%M%S)
for log in "$OUT"/*_*.log; do            # logs of interrupted steps: set aside so the lanes (not the sweeper) rerun them
  [ -f "$log.ok" ] || mv "$log" "$log.interrupted_$stamp"
done
for q in "$OUT"/*.qtried; do             # queue markers of unfinished steps: let the queue run them again
  [ -e "$q" ] && [ ! -f "${q%.qtried}.ok" ] && rm -f "$q"
done
echo "$(date +%T) ==== resumed after restart" >> "$OUT/queue.log"
nohup bash scripts/run_audit_reruns.sh A >> "$OUT/queue.log" 2>&1 &
nohup bash scripts/run_audit_reruns.sh G >> "$OUT/queue.log" 2>&1 &
nohup bash scripts/run_audit_queue.sh W1 >> "$OUT/queue.log" 2>&1 &
sleep 90
nohup bash scripts/run_audit_queue.sh W2 >> "$OUT/queue.log" 2>&1 &
nohup bash scripts/retry_failed_reruns.sh >> "$OUT/queue.log" 2>&1 &     # re-runs steps that fail (e.g. out of memory)
echo "started lanes A, G, queue workers W1, W2 and the retry sweeper; progress in $OUT/queue.log"
