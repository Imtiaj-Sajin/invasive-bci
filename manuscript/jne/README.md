# Journal of Neural Engineering version (planned)

The second target, prepared after the Nature Communications version (`../natcomms/`) is final. Both versions use
the same verified numbers (`scripts/make_numbers_tex.py --out ../natcomms/numbers.tex ../jne/numbers.tex`), the same
reference metadata (`../tools/make_bib.py`) and the same figure code (`scripts/make_v2_figures.py`).

## Why JNE as the second venue

- Q1 and the core journal of the intracortical BCI engineering community, where the readers who would reuse the
  recalibration recipes and the simulator publish.
- More room than Nature Communications: no tight display-item limit, so the main text can show what the NC version
  keeps in the Supplementary Information.

## What changes relative to the NC version (check the current author guidelines before writing)

- Abstract: JNE uses a structured abstract (Objective, Approach, Main results, Significance).
- Template: IOP Publishing LaTeX template; numeric references.
- Main text can include, as full figures or tables rather than supplementary items:
  - the per-individual correction ladder (current Supplementary Table 2) as a figure;
  - the network-decoder ladder for all six individuals;
  - the data-efficiency curves per participant;
  - the label-free monitoring pilot (Supplementary Note 1);
  - the electrode robustness controls (thresholds, fixed microvolt threshold, waveform, array-wide shifts);
  - the simulator model and its validation in more detail, as a resource section.
- Methods can be fuller (decoder and correction details now split between Methods and Supplementary Notes).

Nothing here is written until the analyses and the NC text are final, so the two versions stay consistent.
