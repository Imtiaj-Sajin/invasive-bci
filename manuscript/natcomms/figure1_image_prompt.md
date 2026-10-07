# Backup prompt for an AI-generated overview image

**Policy warning.** Nature Portfolio has restricted images made with generative AI in published papers since 2023, and
its current AI policy forbids AI-made photorealistic images (<https://www.nature.com/nature-portfolio/editorial-policies/ai>;
the exact rules for illustrations were not confirmed on 2026-10-07 [UNVERIFIED]). To be safe, an image made from this
prompt should **not** go into the manuscript or Supplementary Information. It can be used for slides, a poster, a
social-media post or a lab web page. The manuscript's Figure 1 is built from real data by
`scripts/render_brain.py` and `scripts/make_v2_figures.py --only fig1`.

## Prompt (paste as one block)

Create a clean, scientific overview illustration in the style of a Nature or Nature Neuroscience figure, landscape
format, aspect ratio 16:9, on a plain off-white background (#FCFCFB). No text other than the short labels listed
below, in a simple sans-serif font (Arial or Helvetica), dark grey (#222222). Flat, restrained colours. No glow,
no lens flare, no sci-fi elements, no gradients on the background, no people's faces in close-up.

Layout, left to right, as four equal panels separated by generous white space. Each panel has a small bold lowercase
letter in the top-left corner (a, b, c, d).

Panel a, "Implant". A realistic, softly shaded light-grey human brain seen from the left side, front of the brain to
the left. On the top of the brain, just in front of the central sulcus, in the hand area of the motor cortex, place
two tiny square silicon microelectrode arrays (Utah arrays), each 4 × 4 mm. They should look small relative to the
brain, about the size of a fingernail on a fist. Draw a thin dark-purple (#4A3AA7) square outline around each array.
A thin grey leader line goes from the arrays to a magnified inset in the upper right of the panel. The inset shows one
Utah array in close-up, three-quarter view: a square silicon base with a 10 × 10 grid of thin, sharp, evenly spaced
needles, 1.5 mm long, with metallic platinum tips, sitting on a pink-grey cortical surface. Label: "Utah array,
96 electrodes".

Panel b, "Years pass". Two small square grids side by side, each 10 × 10 light-grey cells. In the left grid, most cells
contain a small black downward spike waveform (a sharp negative dip followed by a small positive bump). In the right
grid, only about one third of the cells contain a waveform, and the waveforms are smaller. Under the left grid write
"Year 0" and under the right grid write "Year 7". A thin grey arrow points from the left grid to the right grid.

Panel c, "Cursor control". A simple flat computer monitor showing a dark-grey screen with eight hollow white circles
arranged evenly on a ring around a central dot. From the centre, eight bundles of thin smooth coloured paths reach the
eight circles (colours: blue #2A78D6, orange #EB6834, green #1BAF7A, yellow #EDA100, pink #E87BA4,
dark green #008300, purple #4A3AA7, red #E34948). In front of the monitor, seen from behind and to the side, a person sits in a
wheelchair, shown as a simple, respectful silhouette, with a thin cable running from a small connector on the top of
the head. Label: "Closed-loop cursor".

Panel d, "Decoder decay and correction". Three small square plots in a vertical stack, each with eight hollow grey
target circles on a ring. Top plot: coloured paths reach the targets well; label "Same day". Middle plot: the paths
are short, tangled and all drift to the left, missing the targets; label "Old decoder". Bottom plot: the paths again
fan out toward the correct targets, slightly less neat than the top; label "Old decoder, re-weighted".

Overall style: minimal, precise, calm, high resolution (at least 3000 pixels wide), thin lines (0.5–1 pt), consistent
colours across panels, lots of white space, suitable for a top scientific journal. Do not add a title, logos,
watermarks or extra labels.

## Tips

- If the brain looks cartoonish, add: "photorealistic MRI-derived cortical surface, matte grey, soft studio light".
- If the arrays look too large, add: "the arrays must be tiny, less than 3% of the brain's length".
- If extra text appears, regenerate with: "remove all text except the labels I listed".
- Generate panels one at a time if the full layout fails, then combine them in Inkscape, Affinity Designer or
  PowerPoint.
