# Guide for AI agents and new contributors

This repo is a research project on **invasive / intracortical brain-computer interfaces (iBCI)**.
Goal: a Q1-journal-quality, *computational* study whose outputs (method, benchmark, code, data) other
researchers will reuse and cite for years. Read this file first, then `docs/RESEARCH_LOG.md`.

## Where knowledge lives

| Path | What it holds |
| --- | --- |
| `docs/RESEARCH_LOG.md` | Chronological log of what was done, found, and decided. Append, never rewrite history. |
| `docs/knowledge/` | Curated reference notes (field landscape, datasets, methods, hardware/electrode facts). Each claim should carry a source URL. |
| `docs/decisions/` | Decision records: why a topic/method/dataset was chosen and what was rejected. |
| `src/` | Library code for the project. |
| `scripts/` | Runnable entry points (download, preprocess, train, evaluate). |
| `results/` | Small result artefacts (tables, figures, JSON metrics). Large files stay out of git. |
| `data/` | Downloaded datasets. **Git-ignored.** Recreate with the download scripts. |

## Rules of the road

- **Commits (owner's instruction, 2026-10-06):** commit as the repo-local git user Md Wahiduzzaman Suva
  (GitHub wshuv-o, wshuvo360@gmail.com). End every commit message with two trailers:
  `Co-authored-by: Md. Imtiaj Alam Sajin <imtiajsajin@gmail.com>` and
  `Co-authored-by: Esm E Moula Chowdhury Abha <esmechowdhuryabha@gmail.com>`.
  **Never** add a Claude or any other AI attribution trailer. Pushing to `origin` is allowed.
- **Authors** (order as listed by the owners; all American International University-Bangladesh, Dhaka):
  Md. Imtiaj Alam Sajin (ORCID 0009-0009-2423-1835), Esm E Moula Chowdhury Abha (0009-0008-3776-2283),
  Md Wahiduzzaman Suva (0009-0007-6227-7282).

- Log every meaningful step in `docs/RESEARCH_LOG.md` with the date (YYYY-MM-DD).
- Put sources (URLs, DOIs) next to factual claims in `docs/knowledge/`. Mark unverified claims `[UNVERIFIED]`.
- Never commit datasets or model checkpoints larger than a few MB.
- Experiments must be reproducible from a script plus a fixed seed. Record the exact command in the log.

## Compute environment (owner's PC, checked 2026-10-05)

- CPU: Intel i5-12400 (6C/12T), RAM: 24 GB
- GPU: NVIDIA RTX 3060 Ti, **8 GB VRAM**, driver 610.88, CUDA 12.6
- Python 3.11.3 at `C:\Python311\python.exe`; torch 2.7.0+cu126 (CUDA works), numpy 2.2, scipy, sklearn, h5py, mne
- Disk is tight: G: ~29 GB free, D: ~56 GB free, **C: ~11 GB free** (keep pip/hf caches off C:)
- OS: Windows 11; both PowerShell and Git Bash are available
