#!/usr/bin/env bash
# Sequential queue for the revision analyses (one CPU-heavy job at a time; 24 GB RAM cannot hold several 192-channel
# jobs at once). Each step is skipped when its finished log already contains the final summary table, so the queue
# can be restarted safely. The GPU network ladder (scripts/nn_ladder.py) runs separately alongside.
#
# Usage: bash scripts/run_revision_queue.sh > results/revision_queue.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe

finished() { grep -q "^gap_target" "$1" 2>/dev/null; }

step() {   # step LOG ENV COMMAND...
  local log="$1" env="$2"; shift 2
  if finished "$log"; then echo "skip  $log"; return; fi
  echo "$(date +%T) start $log"
  if [ -n "$env" ]; then env "$env" "$PY" "$@" > "$log" 2>&1; else "$PY" "$@" > "$log" 2>&1; fi
  finished "$log" && echo "$(date +%T) done  $log" || echo "$(date +%T) FAILED $log"
}

T9ENV="IBCI_DATA=G:/ibci-data"

# 1. T9 correction ladder (resumes from its CSV), regularization, summaries
if ! grep -q "== .* done" results/pipeline_T9.log 2>/dev/null; then
  echo "$(date +%T) start T9 pipeline"
  bash scripts/run_braingate_participant.sh T9 - G:/ibci-data/braingate > results/pipeline_T9.log 2>&1 \
    && echo "$(date +%T) done  T9 pipeline" || echo "$(date +%T) FAILED T9 pipeline"
fi

# 2. decay with the tuned penalty
step results/decay_alpha_T5.log "" scripts/decay_alpha.py --subject T5
step results/decay_alpha_T9.log "$T9ENV" scripts/decay_alpha.py --subject T9

# 3. alignment with and without variance normalization
for s in C M T6 T5; do step "results/align_variants_$s.log" "" scripts/align_variants.py --subject "$s"; done
step results/align_variants_T9.log "$T9ENV" scripts/align_variants.py --subject T9

# 4. CORAL and the Degenhart stabilizer
for s in C M T6 T5; do step "results/label_free_plus_$s.log" "" scripts/label_free_plus.py --subject "$s"; done
step results/label_free_plus_T9.log "$T9ENV" scripts/label_free_plus.py --subject T9

# 5. how many labelled trials each supervised correction needs (humans)
for s in T6 T5; do step "results/gain_efficiency_$s.log" "" scripts/gain_efficiency.py --subject "$s"; done
step results/gain_efficiency_T9.log "$T9ENV" scripts/gain_efficiency.py --subject T9

# 6. refresh summaries, statistics, tables and figures with everything included
"$PY" scripts/rule_summary.py > /dev/null && "$PY" scripts/decay_summary.py > /dev/null \
  && "$PY" scripts/paper_stats.py > /dev/null && "$PY" scripts/revision_stats.py > /dev/null \
  && "$PY" scripts/make_supp_tables.py > /dev/null && "$PY" scripts/make_paper_figures.py > /dev/null \
  && echo "$(date +%T) summaries refreshed"
echo "$(date +%T) QUEUE COMPLETE"
