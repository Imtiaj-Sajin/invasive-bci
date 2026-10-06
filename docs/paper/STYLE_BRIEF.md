# Writing brief for the Nature Communications manuscript (2026-10-06)

*Condensed from a sub-agent report using Nature Communications' guidelines and six exemplar papers.
The full report is summarized in the log.*

## Hard requirements (use the stricter value where official sources conflict)

- **Title:** 15 words or fewer, no punctuation or colon, no acronyms.
- **Abstract:** 150 words or fewer, unstructured, no references or acronyms. Open with background. Have a sentence starting
  "Here, we show" (present tense).
- **Main text:** 4,500–5,000 words (Introduction + Results + Discussion).
- **Display items:** up to 10 (plan 7–8 figures). Legends: 350 words or fewer, starting with a one-sentence title.
- **Methods:** up to about 3,000–4,500 words. **References:** up to 70, Nature numbered style. Datasets are cited with DOIs.
- **Sections:** Introduction (heading, no subheadings; the last paragraph starts "Here, we"), Results (subheadings of 60
  characters or fewer stating findings), Discussion (no subheadings), Methods (subheadings), Data availability,
  Code availability, References, Acknowledgements, Author contributions, Competing interests.
- **Banned words:** novel, new, first, unprecedented, extremely, remarkably, strikingly. No "data not shown".
- **Supplementary:** cited as "Supplementary Fig. 1" or "Supplementary Note 1".
- **Statistics:** exact n and its unit, the test, one- or two-sided, the CI, an exact P (capital italic *P*, ×10ⁿ). State the independent unit.
- **Figures:** 88 or 180 mm wide; Arial 5–7 pt; panels labelled a, b, c in lowercase bold; colours described in words; no red/green pairs.
- **LaTeX:** `\documentclass[pdflatex,sn-nature]{sn-jnl}`. Paste the .bbl into the .tex for submission.
- **Preprints:** allowed. AI use: copy editing need not be declared; substantive use is documented in Methods.

## Style rules

- **Paragraphs:** one idea per paragraph. Claim first, then evidence, then meaning.
- **Sentences:** mean of about 20 words, cap at about 35. Split sentences with stacked clauses.
- **Voice and tense:** active voice with "we". Past tense for what we did, present for what the data show.
- **Terminology:** fixed. Use overnight drop, renormalization, label-free stabilizer, neural drift, decoder-output drift,
  silencing, revival, edge electrode, regularize for the future.
- **Punctuation:** no em dashes. En dashes only for ranges.
- **Hedging:** hedge once, at the right strength (indicates / suggests / may).
- **Re-analysis of prior work** (MINDFUL): state what they reported, what we computed and how the two differ, in a neutral tone.
- **Simulator claims:** limited to what was calibrated and validated.
- **Spelling:** US English throughout.

## Reference voice: Frank Willett (Stanford NPTL; Scholar profile g1x3RKgAAAAJ, provided by the owner)

Notes from first-author papers: Nature 2021 (handwriting), Nature 2023 (speech), Cell 2020, Sci Rep 2019, JNE 2016.

- **Abstract pattern:**
  1. A patient-centred promise.
  2. "However/So far…" gap.
  3. One "Here we…" sentence.
  4. Numbers tied to benchmarks a reader can picture.
  5. Optionally "Finally,…" with a secondary insight.
  6. A measured close ("These results show a feasible path…").
- **Hedged priority:** "To our knowledge…", "we are aware of no prior work that…".
- **Paragraphs:** context → content → conclusion. Open with an aim ("We tested this by…", "Next, we asked whether…").
  Close with a takeaway ("Taken together, these results suggest…").
- **Voice:** active "we". Participants are identified by code with brief clinical context. Hedges are moderate.
- **Our adaptation:** keep his structure, but use *shorter* sentences than his (about 20 words on average versus his 24–30) and no em dashes.

## Evidence-based readability rules (Gopen & Swan 1990; Mensh & Kording 2017; Plavén-Sigray 2017; Ryba 2021; Martínez & Mammola 2021)

1. **Sentence order:** the verb comes right after the subject; old information first, new information last (stress position).
2. **One stress-worthy point per sentence.** Split a sentence when a second key fact appears.
3. **Abstract style:** no noun stacks, no acronyms, "we", about 110–150 words. This tested as most readable and most understood (Ryba 2021).
4. **Jargon:** none in the title or abstract. Defining jargon does not remove its cost.
5. **Terms:** one term per concept, never synonyms.
6. **Paragraphs:** context-content-conclusion. Cover each topic in one place, and use parallel syntax for parallel points.
7. **Signposts:** "Next", "First… Finally", "In contrast", "Taken together".
8. **Titles:** declarative, results-stating; not questions; short and with common words.
9. **Figures:** each title states the conclusion; the legend explains the method.
10. **Reviewers:** match claims to the evidence, address alternatives, give complete methods, and avoid stacked defensive caveats.
