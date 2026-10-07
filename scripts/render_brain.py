"""Render a real cortical surface (FreeSurfer fsaverage, via nilearn) with Utah array sites on motor cortex.

The surface is the standard MRI-derived template brain, shaded by sulcal depth. Array sites are marked at the hand
knob of the left precentral gyrus, where BrainGate arrays are typically placed (MNI approximately x = -36, y = -22,
z = 64, and a second site about 13 mm lateral and ventral). Each site is shown as a 4 x 4 mm patch of surface vertices
(the footprint of one Utah array) and outlined by a square marker so that it stays visible at figure size.

Usage: SCIPY_ARRAY_API=1 python scripts/render_brain.py [--out results/figures/v2/brain_render.png]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from nilearn import datasets, plotting, surface  # noqa: E402
from scipy import ndimage  # noqa: E402

SITES = (np.array([-36.0, -22.0, 64.0]), np.array([-44.0, -14.0, 54.0]))
COLOR = "#4a3aa7"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures/v2/brain_render.png")
    ap.add_argument("--data", default="G:/nilearn_data")
    ap.add_argument("--elev", type=float, default=28.0)
    ap.add_argument("--azim", type=float, default=180.0)
    args = ap.parse_args()
    fs = datasets.fetch_surf_fsaverage("fsaverage", data_dir=args.data)
    coords, _ = surface.load_surf_mesh(fs["pial_left"])
    sulc = surface.load_surf_data(fs["sulc_left"])
    roi = np.full(len(coords), np.nan)
    centers = []
    for target in SITES:
        near = np.flatnonzero(np.linalg.norm(coords - target, axis=1) < 8.0)
        c = coords[near[np.argmin(sulc[near])]]                           # crown of the gyrus near the target
        centers.append(c)
        roi[np.linalg.norm(coords - c, axis=1) < 2.3] = 1.0               # about the 4 x 4 mm footprint of one array
    fig = plt.figure(figsize=(6, 4.5), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    plotting.plot_surf(fs["pial_left"], surf_map=roi, bg_map=sulc, hemi="left", view=(args.elev, args.azim),
                       cmap=matplotlib.colors.ListedColormap([COLOR]), vmin=0.5, vmax=1.5, bg_on_data=False, axes=ax,
                       figure=fig, colorbar=False)
    fig.patch.set_alpha(0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    # locate the rendered array patches in the image and outline each with a square, so they stay visible when small
    rgb = np.array(matplotlib.colors.to_rgb(COLOR)) * 255
    hit = (np.abs(img[..., :3].astype(float) - rgb).sum(-1) < 60) & (img[..., 3] > 0)
    lab, n = ndimage.label(ndimage.binary_dilation(hit, iterations=3))
    cents = ndimage.center_of_mass(hit, lab, range(1, n + 1))
    ink = (img[..., 3] > 0) & (img[..., :3].min(-1) < 245)
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    img = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    img[img[..., :3].min(-1) >= 245, 3] = 0                               # white background becomes transparent
    h, w = img.shape[:2]
    out = plt.figure(figsize=(w / 300, h / 300), dpi=300)
    oa = out.add_axes([0, 0, 1, 1])
    oa.imshow(img, interpolation="none")
    for r, c in cents:
        oa.plot(c - cols[0], r - rows[0], "s", ms=8, mfc="none", mec=COLOR, mew=1.3)
    oa.set_xlim(-0.5, w - 0.5)
    oa.set_ylim(h - 0.5, -0.5)
    oa.axis("off")
    out.savefig(args.out, dpi=300, transparent=True)
    print(n, "array sites outlined")
    print("saved", args.out)


if __name__ == "__main__":
    main()
