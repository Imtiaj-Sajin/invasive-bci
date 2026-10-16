#!/usr/bin/env bash
# Regenerate every summary, number, table and figure of the paper from results/ (no model fitting), then compile.
# Order matters: summaries -> numbers.tex -> supplementary tables -> figures -> PDF.
# Usage: bash scripts/regenerate_paper.sh [--no-pdf]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
run() { echo "== $*"; "$PY" "$@"; }

run scripts/v2_summary.py
IBCI_ALL_PAIRS=1 run scripts/v2_summary.py          # all-pairs sensitivity (results/v2_summary_allpairs.json)
run scripts/decay_summary.py
run scripts/decay_summary.py --tuned                # tuned-penalty half-lives (results/decay_summary_tuned.csv)
run scripts/heterogeneity_model.py                  # mixed model, curves differ (results/heterogeneity)
run scripts/rule_summary.py
run scripts/paper_stats.py
run scripts/bench_from_paper.py                     # benchmark baselines (results/bench)
run scripts/ndt3/summarize_pilot.py                 # NDT3 pilot summary (results/ndt3/pilot_summary.json)
run scripts/revision_stats.py
run scripts/make_numbers_tex.py
run scripts/make_supp_tables.py
run scripts/make_v2_figures.py

if [ "${1:-}" != "--no-pdf" ]; then
  cd manuscript/natcomms
  for f in main supplementary; do
    pdflatex -interaction=nonstopmode -halt-on-error "$f.tex" > /dev/null
    pdflatex -interaction=nonstopmode -halt-on-error "$f.tex" > /dev/null
    echo "== compiled $f.pdf"
  done
fi
