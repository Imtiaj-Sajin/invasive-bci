# IEEE Transactions on Neural Systems and Rehabilitation Engineering version (planned)

Third target, built from the same text, numbers (`scripts/make_numbers_tex.py`), reference metadata
(`../tools/make_bib.py`) and figure code (`scripts/make_v2_figures.py`) as `../natcomms/` (master text) and `../jne/`.
When the manuscript changes, all three versions are updated together so they never disagree.

## Requirements (EMBS TNSRE submission guidelines, checked 2026-10-16)

Source: https://www.embs.org/tnsre/for-authors/submission-guidelines/

- Regular paper: **10 pages or less in the IEEE transactions template, not counting references**. Longer manuscripts
  "will not be admitted for review".
- Abstract: at most 250 words (unstructured).
- Template: IEEE transactions (Word or LaTeX, `IEEEtran`); main text at least 10 pt, tables and figures at least 8 pt.
- Figures and tables interspersed in the text, numbered, with self-contained captions.
- References: IEEE style, numbers in square brackets, in a separate section at the end.
- **Gold, fully open access. APC US$2075 for articles submitted in 2025** (2026 figure to be checked), due on acceptance;
  IEEE member 5% and EMBS member 20% discounts. Licence CC BY or CC BY-NC-ND.
- Methods must state ethics oversight (committee name and protocol number) and informed consent for human data. This
  study re-analyzes public, de-identified data; the statement must say so and cite the original studies' approvals.
  An exemption note from the authors' institution (AIUB) should be obtained before submission.
- Submission via the IEEE Author Portal, with the IEEE copyright form and the author consent form; reviewers may be
  suggested.

## What changes relative to the master text

- About half the length: keep decay, regularization, label-free corrections with the latent ceiling, labeled
  recalibration and the benchmark in the main text; move the monitoring, NDT3 pilot, electrode and simulator sections to
  the supplementary material, with one sentence each in the main text.
- Five or six figures at two-column width; the per-individual ladder and the remaining figures go to the supplement.
- Numbered IEEE sections (Introduction, Methods, Results, Discussion, Conclusion).

Nothing here is written until the analyses and the master text are final.
