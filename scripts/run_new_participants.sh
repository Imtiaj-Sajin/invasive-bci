#!/usr/bin/env bash
# Six additional BrainGate decoding participants (T3, T2, T7, T10, T8, T11), smallest first.
#
# Two loops run at the same time:
#   convert : streams each archive from the owners' server straight into compact session files (no archive is stored;
#             the gzip CRC is checked at the end) and marks it done.
#   analyse : waits for each participant's mark, then runs the analyses one at a time (24 GB RAM cannot hold several
#             192-channel jobs at once). Steps whose log already ends with the summary table are skipped, so the
#             queue can be restarted safely.
#
# Usage: bash scripts/run_new_participants.sh convert > results/new_participants/convert.log 2>&1 &
#        bash scripts/run_new_participants.sh analyse > results/new_participants/analyse.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
URL=http://31.97.211.4:53808
ORDER="T3 T2 T7 T10 T8 T11"
OUT=results/new_participants
mkdir -p "$OUT"

data_root() {   # where each participant's compact files live (chosen by free space, see RESEARCH_LOG 2026-10-07)
  case "$1" in
    T8) echo G:/ibci-data ;;
    T11) echo D:/ibci-data ;;
    *) echo F:/ibci-data ;;
  esac
}

convert() {
  for P in $ORDER; do
    [ -f "$OUT/convert_$P.done" ] && { echo "skip  convert $P"; continue; }
    R=$(data_root "$P")
    if [ "$P" = T11 ]; then   # needs about 20 GB on D:, which requires the cleanup the owners must approve first
      free=$(df -BG --output=avail /d | tail -1 | tr -dc 0-9)
      if [ "$free" -lt 24 ]; then echo "$(date +%T) WAIT  T11 needs ~24 GB on D:, only ${free} GB free"; continue; fi
    fi
    echo "$(date +%T) start convert $P -> $R/braingate/decoding/$P"
    if "$PY" scripts/tools/compact_braingate_decoding.py "$URL/decoding_$P.tar.gz" "$R/braingate/decoding/$P" \
         > "$OUT/convert_$P.log" 2>&1 && grep -q "gzip CRC verified" "$OUT/convert_$P.log"; then
      touch "$OUT/convert_$P.done"; echo "$(date +%T) done  convert $P ($(ls "$R/braingate/decoding/$P" | wc -l) sessions)"
    else
      echo "$(date +%T) FAILED convert $P (see $OUT/convert_$P.log)"
    fi
  done
  echo "$(date +%T) CONVERT LOOP COMPLETE"
}

finished() { grep -q "^gap_target\|^participant\|done" "$1" 2>/dev/null; }

step() {   # step LOG ENV COMMAND...
  local log="$1" env="$2"; shift 2
  if [ -f "$log.ok" ]; then echo "skip  $log"; return; fi
  echo "$(date +%T) start $log"
  if env "$env" "$PY" "$@" > "$log" 2>&1; then touch "$log.ok"; echo "$(date +%T) done  $log"
  else echo "$(date +%T) FAILED $log (exit $?)"; fi
}

analyse() {
  for P in $ORDER; do
    while [ ! -f "$OUT/convert_$P.done" ]; do
      grep -q "CONVERT LOOP COMPLETE" "$OUT/convert.log" 2>/dev/null && break
      sleep 60
    done
    [ -f "$OUT/convert_$P.done" ] || { echo "$(date +%T) no data for $P, skipped"; continue; }
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
  done
  echo "$(date +%T) ANALYSE LOOP COMPLETE"
}

"$1"
