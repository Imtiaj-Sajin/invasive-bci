#!/usr/bin/env bash
# Full human-decoder pipeline for one BrainGate participant, from the downloaded decoding archive to the paper outputs.
#
# Usage: bash scripts/run_braingate_participant.sh T5 G:/ibci-dl/decoding_T5.tar.gz [OUT_ROOT]   (ARCHIVE "-" = already converted)
#   OUT_ROOT defaults to $IBCI_DATA/braingate (compact session files go to OUT_ROOT/decoding/<P>).
# Steps: stream-convert the archive (checks gzip CRC), correction ladder (supervised subspace rotation and full input
# remap skipped; they are not reported for humans), regularization trade-off, then every summary, table and figure.
set -euo pipefail
P="$1"
ARCHIVE="$2"
ROOT="${3:-${IBCI_DATA:-D:/ibci-data}/braingate}"
PY="${PY:-.venv/Scripts/python.exe}"
cd "$(dirname "$0")/.."

if [ "$ARCHIVE" != "-" ]; then
echo "== $(date +%T) convert $ARCHIVE -> $ROOT/decoding/$P"
"$PY" scripts/tools/compact_braingate_decoding.py "$ARCHIVE" "$ROOT/decoding/$P"
fi

echo "== $(date +%T) correction ladder"
"$PY" scripts/replicate_braingate_decoding.py --participant "$P" --root "$ROOT" --skip L4 L5 --resume \
  >> "results/replication_bg_$P.log" 2>&1

echo "== $(date +%T) regularization trade-off"
IBCI_DATA="$(dirname "$ROOT")" "$PY" scripts/reg_tradeoff.py --dataset braingate --subject "$P" \
  --out "results/reg_tradeoff_$P" > "results/reg_tradeoff_$P.log" 2>&1

echo "== $(date +%T) summaries, tables, figures"
"$PY" scripts/rule_summary.py > /dev/null
"$PY" scripts/decay_summary.py
"$PY" scripts/paper_stats.py > /dev/null
"$PY" scripts/make_supp_tables.py > /dev/null
"$PY" scripts/make_paper_figures.py > /dev/null
echo "== $(date +%T) done"
