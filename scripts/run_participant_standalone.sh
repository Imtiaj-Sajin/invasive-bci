#!/usr/bin/env bash
# Run every analysis for one BrainGate decoding participant on any machine (used for T11 on a second PC).
#
# Steps: (1) stream-convert the archive (URL or local .tar.gz) unless already converted, (2) the CPU analyses, one at a
# time, (3) the recurrent-network ladder on the GPU. Logs, .ok markers and result files have the same names as on the
# main PC (results/new_participants/, results/<analysis>/), so the results can simply be committed and merged.
#
# Usage: bash scripts/run_participant_standalone.sh P DATA_ROOT [ARCHIVE] [YIELD_ROOT]
#   P          participant, e.g. T11
#   DATA_ROOT  folder that will hold braingate/decoding/<P>/ (e.g. D:/ibci-data)
#   ARCHIVE    URL or path of decoding_<P>.tar.gz; omit or "-" if already converted
#   YIELD_ROOT folder that contains yield/<P>/<P>_day_*_yield.mat (needed only for the gain-electrode link)
# Environment: PY (python executable, default .venv/Scripts/python.exe or python3).
set -u
cd "$(dirname "$0")/.."
P="$1"; R="$2"; ARCHIVE="${3:--}"; Y="${4:-$R/braingate}"
PY="${PY:-$( [ -x .venv/Scripts/python.exe ] && echo .venv/Scripts/python.exe || echo python3 )}"
OUT=results/new_participants
mkdir -p "$OUT"

if [ "$ARCHIVE" != "-" ] && [ ! -f "$OUT/convert_$P.done" ]; then
  echo "$(date +%T) convert $ARCHIVE -> $R/braingate/decoding/$P"
  "$PY" scripts/tools/compact_braingate_decoding.py "$ARCHIVE" "$R/braingate/decoding/$P" > "$OUT/convert_$P.log" 2>&1 \
    && grep -q "gzip CRC verified" "$OUT/convert_$P.log" && touch "$OUT/convert_$P.done" \
    || { echo "conversion failed, see $OUT/convert_$P.log"; exit 1; }
fi

step() {   # step LOG COMMAND...
  local log="$1"; shift
  if [ -f "$log.ok" ]; then echo "skip  $log"; return; fi
  echo "$(date +%T) start $log"
  if IBCI_DATA="$R" IBCI_YIELD="$Y" "$PY" "$@" > "$log" 2>&1; then touch "$log.ok"; echo "$(date +%T) done  $log"
  else echo "$(date +%T) FAILED $log"; fi
}

echo "$(date +%T) ==== $P ($(ls "$R/braingate/decoding/$P" | wc -l) sessions)"
step "$OUT/${P}_ladder.log" scripts/replicate_braingate_decoding.py --participant "$P" --root "$R/braingate" --skip L4 L5 --resume
step "$OUT/${P}_nn_ladder.log" scripts/nn_ladder.py --subject "$P"
step "$OUT/${P}_decay_alpha.log" scripts/decay_alpha.py --subject "$P"
step "$OUT/${P}_reg_tradeoff.log" scripts/reg_tradeoff.py --dataset braingate --subject "$P" --out "results/reg_tradeoff_$P"
step "$OUT/${P}_align_variants.log" scripts/align_variants.py --subject "$P"
step "$OUT/${P}_label_free_plus.log" scripts/label_free_plus.py --subject "$P"
step "$OUT/${P}_gain_efficiency.log" scripts/gain_efficiency.py --subject "$P"
step "$OUT/${P}_reweight_controls.log" scripts/reweight_controls.py --subject "$P"
step "$OUT/${P}_gain_mechanism.log" scripts/gain_mechanism.py --subject "$P"
step "$OUT/${P}_weight_angles.log" scripts/weight_angles.py --subject "$P"
echo "$(date +%T) ==== $P finished"
