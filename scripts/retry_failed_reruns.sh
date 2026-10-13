#!/usr/bin/env bash
# Sweeper for the audit reruns: finds steps that failed (log present, no .ok, no .lock, not running, log unchanged for
# 3 minutes) and runs them again, one at a time, once enough commit memory is free. The failures of 2026-10-13 were
# out-of-memory errors when several monkey C/M steps loaded all their sessions at once. A step is tried at most
# MAX_TRIES times by the sweeper; it loops until every expected step has its .ok marker or has used up its tries.
#
# Usage: bash scripts/retry_failed_reruns.sh >> results/audit_reruns/queue.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=results/audit_reruns
MAX_TRIES=${MAX_TRIES:-3}
ALL="N C M T6 T5 T9 T7 T10 T8"
HUM="T6 T5 T9 T7 T10 T8"

root() { case "$1" in T8) echo H:/ibci-data ;; T9) echo G:/ibci-data ;; T7|T10) echo F:/ibci-data ;; *) echo D:/ibci-data ;; esac; }
commit_free_gb() { powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" | tr -dc 0-9; }
running() {   # running SCRIPT_BASENAME SUBJECT|-   -> true if a python process runs it
  local pat="scripts/$1"; [ "$2" != - ] && pat="$pat.* --subject $2( |\$)"
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$pat' } | Select-Object -First 1 -ExpandProperty ProcessId" | grep -q .
}

STEPS=()
for k in $ALL; do STEPS+=("decay_alpha|$k|scripts/decay_alpha.py --subject $k"); done
for k in $ALL; do STEPS+=("reweight_controls|$k|scripts/reweight_controls.py --subject $k"); done
for k in $HUM; do STEPS+=("gain_mechanism|$k|scripts/gain_mechanism.py --subject $k"); done
for k in $ALL; do STEPS+=("align_variants|$k|scripts/align_variants.py --subject $k"); done
for k in $ALL; do STEPS+=("label_free_plus|$k|scripts/label_free_plus.py --subject $k"); done
for k in $HUM; do STEPS+=("gain_efficiency|$k|scripts/gain_efficiency.py --subject $k"); done
for k in C M; do STEPS+=("matched_output|$k|scripts/matched_output_control.py --subject $k --alpha 1e4 --out results/matched_output_tuned"); done
STEPS+=("recal_policy|N|scripts/recal_policy.py")
for k in $ALL; do STEPS+=("nn_ladder|$k|scripts/nn_ladder.py --subject $k"); done

while :; do
  left=0
  for s in "${STEPS[@]}"; do
    IFS='|' read -r name k cmd <<< "$s"
    log="$OUT/${name}_$k.log"
    [ -f "$log.ok" ] && continue
    tries=$(ls "$log".try_* 2>/dev/null | wc -l)
    [ "$tries" -ge "$MAX_TRIES" ] && continue                   # given up; reported below
    left=$((left + 1))
    [ -f "$log" ] || continue                                   # not started yet: its lane will run it
    [ -d "$log.lock" ] && continue                              # a queue worker or this sweeper has it
    [ -n "$(find "$log" -mmin -3)" ] && continue                # just written: running or finishing
    script=$(basename "${cmd%% *}"); subj=$k; [ "$name" = recal_policy ] && subj=-
    running "$script" "$subj" && continue                       # lane A, G or a worker is running it
    [ "$name" = nn_ladder ] && [ ! -f "$OUT/decay_alpha_$k.log.ok" ] && continue
    need=6; case "$k" in C|M) need=12 ;; esac                   # monkeys C and M hold all sessions in memory
    mkdir "$log.lock" 2>/dev/null || continue
    while [ "$(commit_free_gb)" -lt "$need" ]; do sleep 120; done
    cp "$log" "$log.try_$((tries + 1))"                          # keep the failed log, count the attempt
    echo "$(date +%T) RETRY start $log (attempt $((tries + 1)))"
    # shellcheck disable=SC2086
    if IBCI_DATA="$(root "$k")" "$PY" $cmd > "$log" 2>&1; then touch "$log.ok"; echo "$(date +%T) RETRY done  $log"
    else echo "$(date +%T) RETRY FAILED $log"; fi
    rmdir "$log.lock"
  done
  [ "$left" -eq 0 ] && break
  sleep 300
done
for s in "${STEPS[@]}"; do
  IFS='|' read -r name k _ <<< "$s"; log="$OUT/${name}_$k.log"
  [ -f "$log.ok" ] || echo "$(date +%T) GAVE UP $log after $MAX_TRIES retries"
done
echo "$(date +%T) ==== sweeper finished"
