# Research log

Newest entries at the bottom. Dates are YYYY-MM-DD.

## 2026-10-05 — Project start, topic search

**Brief from owner:** find an invasive-BCI / intracortical-electrode topic that can be published in a Q1
journal and, above all, be *useful to others in future* (high reuse and citation). Any topic is acceptable.
Owner trusts the agent to choose, then carry out the full work in this repo.

**Constraints found:**
- Repo was empty at start.
- Hardware: RTX 3060 Ti 8 GB, 24 GB RAM, about 85 GB usable disk across G: and D: (see `CLAUDE.md`).
  This rules out training large neural foundation models from scratch and any topic that needs
  terabytes of raw 30 kHz broadband data.
- No wet lab and no patients, so the work must use public data or simulation.

**Research plan:** four parallel literature/data sweeps:
1. Field landscape 2024–2026 and clinical bottlenecks.
2. Inventory of public intracortical datasets (sizes, time spans, licences).
3. ML state of the art for iBCI decoding (foundation models, recalibration, speech, low-power, robustness).
4. Electrode degradation, on-implant compression, simulators, spike sorting.

Results will be saved under `docs/knowledge/`.

**Early findings:**
- DANDI API reachable. Exact sizes:
  - NLB: MC_Maze 000128 (0.69 GB), MC_RTT 000129 (0.05 GB), MC_Maze_Large/Medium/Small 000138/139/140
    (0.15/0.08/0.03 GB), Area2_Bump 000127 (1.82 GB).
  - 000070 "Neural population dynamics during reaching": 53.3 GB (too large for comfort).
  - **000688 "Long-term recordings of motor and premotor cortical spiking activity during reaching in
    monkeys": 13.2 GB, 111 assets.** A candidate for studying signal change over months.
- Stimulation side (ICMS, visual cortical prostheses): mostly psychophysics with little public data.
  Existing tool: `dynaphos`, a differentiable phosphene simulator (eLife 2024,
  https://elifesciences.org/articles/85812). Deprioritised because public data is limited.
