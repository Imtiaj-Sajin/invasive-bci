#!/usr/bin/env bash
# Reruns after the code audit of 2026-10-13 (decision log in docs/RESEARCH_LOG.md):
#   - cv_shrunk now includes the anchor ('no update') as a candidate, which changes every CV-chosen supervised rung
#     (re-weighting L3, refit-to-prior L6p, retrained L6) and the weights used by the gain-electrode link;
#   - label-free analyses, label efficiency and the network ladder now use the tuned penalty (alpha = 1e4);
#   - gain_mechanism matches decoder channels to yield electrodes by (array, x, y) (T10 was matched to the wrong array);
#   - the network ladder uses the same first 300 labelled trials as the linear corrections.
#
# Three lanes: A = tuned ladders (decay_alpha), B = the other per-pair analyses, G = network ladder on the GPU (after
# each individual's lane-A result, which it reads). Each step writes <log>.ok when it finishes, so the queue can be
# restarted safely. Memory rules in scripts/rerun_lib.sh.
#
# Usage: bash scripts/run_audit_reruns.sh A|B|G >> results/audit_reruns/queue.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=results/audit_reruns
mkdir -p "$OUT"
ALL="N C M T6 T5 T9 T7 T10 T8"
HUM="T6 T5 T9 T7 T10 T8"

root() { case "$1" in T8) echo H:/ibci-data ;; T9) echo G:/ibci-data ;; T7|T10) echo F:/ibci-data ;; *) echo D:/ibci-data ;; esac; }
. scripts/rerun_lib.sh   # memory rules: one monkey C/M step at a time, commit-memory gate

step() {   # step NAME SUBJECT COMMAND...
  local log="$OUT/$1_$2.log" k="$2"; shift 2
  if [ -f "$log.ok" ]; then echo "skip  $log"; return; fi
  heavy_acquire "$k"
  wait_mem "$k"
  echo "$(date +%T) start $log"
  if IBCI_DATA="$(root "$k")" "$PY" "$@" > "$log" 2>&1; then touch "$log.ok"; echo "$(date +%T) done  $log"
  else echo "$(date +%T) FAILED $log"; fi
  heavy_release "$k"
}

case "$1" in
  A)
    for k in $ALL; do step decay_alpha "$k" scripts/decay_alpha.py --subject "$k"; done ;;
  B)
    for k in $ALL; do step reweight_controls "$k" scripts/reweight_controls.py --subject "$k"; done
    for k in $HUM; do step gain_mechanism "$k" scripts/gain_mechanism.py --subject "$k"; done
    for k in $ALL; do step align_variants "$k" scripts/align_variants.py --subject "$k"; done
    for k in $ALL; do step label_free_plus "$k" scripts/label_free_plus.py --subject "$k"; done
    for k in $HUM; do step gain_efficiency "$k" scripts/gain_efficiency.py --subject "$k"; done
    for k in C M; do step matched_output "$k" scripts/matched_output_control.py --subject "$k" --alpha 1e4 \
                       --out results/matched_output_tuned; done
    step recal_policy N scripts/recal_policy.py ;;
  G)
    for k in $ALL; do
      while [ ! -f "$OUT/decay_alpha_$k.log.ok" ]; do sleep 120; done
      step nn_ladder "$k" scripts/nn_ladder.py --subject "$k"
    done ;;
esac
echo "$(date +%T) ==== lane $1 finished"
