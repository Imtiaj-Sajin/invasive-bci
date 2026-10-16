# Journal of Neural Engineering version

Built from the same text as `../natcomms/` (master text), with the same verified numbers
(`scripts/make_numbers_tex.py`), reference metadata (`../tools/make_bib.py`) and figure code
(`scripts/make_v2_figures.py`). A third version for IEEE TNSRE lives in `../ieee_tnsre/`. When the manuscript changes,
all three versions are updated together so they never disagree.

## Requirements (IOP Publishing JNE pages, checked 2026-10-16)

Source: https://publishingsupport.iopscience.iop.org/journals/journal-of-neural-engineering/about-journal-neural-engineering/

- Papers: "not normally more than 12 000 words (14 journal pages)".
- Structured abstract with the headings Objective, Approach, Main results, Significance. IOP's general guidance: an
  abstract should not normally exceed 300 words.
- No required template (free-format submission); numeric references in order of citation.
- Data: IOP research data policy; supplementary files allowed.
- Hybrid journal: publishing under the subscription route is free; optional gold open access APC GBP 2655 /
  USD 3665, with discounts for many lower- and middle-income countries.
- Conflict-of-interest statement in the cover letter.

## Why JNE as the primary target (decided 2026-10-16, after the hostile review)

- Q1 and the core journal of the intracortical BCI engineering community, whose readers would reuse the recalibration
  recipes, the benchmark and the simulator.
- More room than Nature Communications or TNSRE: items kept in the Supplementary Information elsewhere can move into
  the main text.

## What changes relative to the master text

- Structured abstract; numbered sections with Methods before Results (IOP convention); numeric references renumbered
  for the new citation order.
- Candidates to move from the supplement into the main text: the per-individual correction ladder, data-efficiency
  curves per participant, the latent-alignment ceiling table.
