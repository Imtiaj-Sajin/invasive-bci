#!/usr/bin/env bash
# One download lane: stream-convert the given participants one after another (same markers and logs as
# run_new_participants.sh convert). The owners' server gives about 2 MB/s per connection, so several lanes in
# parallel finish sooner; each converter uses ~100 MB of RAM and little CPU.
#
# Usage: bash scripts/convert_lane.sh DATA_ROOT P [P ...]     e.g. bash scripts/convert_lane.sh F:/ibci-data T3 T2
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
URL=http://31.97.211.4:53808
OUT=results/new_participants
R="$1"; shift
for P in "$@"; do
  [ -f "$OUT/convert_$P.done" ] && { echo "skip  convert $P" >> "$OUT/convert.log"; continue; }
  echo "$(date +%T) start convert $P -> $R/braingate/decoding/$P" >> "$OUT/convert.log"
  if "$PY" scripts/tools/compact_braingate_decoding.py "$URL/decoding_$P.tar.gz" "$R/braingate/decoding/$P" \
       > "$OUT/convert_$P.log" 2>&1 && grep -q "gzip CRC verified" "$OUT/convert_$P.log"; then
    touch "$OUT/convert_$P.done"
    echo "$(date +%T) done  convert $P ($(ls "$R/braingate/decoding/$P" | wc -l) sessions)" >> "$OUT/convert.log"
  else
    echo "$(date +%T) FAILED convert $P (see $OUT/convert_$P.log)" >> "$OUT/convert.log"
  fi
done
