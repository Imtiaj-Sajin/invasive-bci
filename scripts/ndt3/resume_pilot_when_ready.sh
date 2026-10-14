#!/usr/bin/env bash
# Restart the NDT3 pilot once the memory-heavy audit rerun steps (monkeys C and M) are finished. The NDT3 WSL VM holds
# about 10 GB of commit memory, which starved those steps on 2026-10-14. The pilot resumes from its saved rows.
# Usage (Git Bash, background): bash scripts/ndt3/resume_pilot_when_ready.sh
cd "$(dirname "$0")/../.."
OUT=results/audit_reruns
heavy_left() {
  local n=0
  for s in decay_alpha nn_ladder reweight_controls align_variants label_free_plus matched_output; do
    for k in C M; do
      log="$OUT/${s}_$k.log"
      [ -f "$log.ok" ] && continue
      grep -q "GAVE UP $log" "$OUT/queue.log" 2>/dev/null && continue
      n=$((n + 1))
    done
  done
  echo $n
}
while [ "$(heavy_left)" -gt 0 ]; do sleep 300; done
echo "$(date +%T) heavy rerun steps finished; resuming NDT3 pilot" >> results/ndt3/pilot/resume.log
run() { MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -u root -- bash "/mnt/g/codes/invasive bci/scripts/ndt3/run_wsl.sh" "$1" "/mnt/g/codes/invasive bci/scripts/ndt3/pilot.py" --subjects "${@:2}"; }
run /mnt/h/ndt3/pilot_N.log N
run /mnt/h/ndt3/pilot_T.log T5 T8
echo "$(date +%T) NDT3 pilot runs ended" >> results/ndt3/pilot/resume.log
