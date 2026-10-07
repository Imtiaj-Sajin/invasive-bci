"""Software renderings for Figure 1 (PyVista / VTK), from real geometry only.

brain : left pial surface of the FreeSurfer fsaverage template (MRI-derived, obtained with
        mne.datasets.fetch_fsaverage), shaded by sulcal depth in a tissue colour, with the precentral gyrus of the
        Desikan-Killiany atlas (aparc) coloured and two Utah array footprints (4 x 4 mm plates with a 10 x 10 grid) on
        its crown near the hand-knob coordinates used in scripts/render_brain.py. Sites are approximate.
utah  : a Utah array built from its published dimensions (10 x 10 shanks, 400 um pitch, 1.5 mm long, tapered,
        on a 4 x 4 mm silicon base) with a gold wire bundle. It is a rendering, not a photograph.

Usage: SCIPY_ARRAY_API=1 python scripts/render_3d.py [--what brain utah] [--out results/figures/v2]
"""
import argparse
import os

import numpy as np
import pyvista as pv

SITES = (np.array([-36.0, -22.0, 64.0]), np.array([-44.0, -14.0, 54.0]))
TISSUE_LIGHT, TISSUE_DARK = np.array([236, 196, 160]) / 255, np.array([170, 112, 78]) / 255
MOTOR_COLOR = "#3d9a5b"


def _array_plate(center, normal, size=4.0, n=10, lift=1.2):
    """A 4 x 4 mm plate with a 10 x 10 grid of bumps, centred on the cortex and facing outward."""
    normal = normal / np.linalg.norm(normal)
    ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    c = center + lift * normal
    plate = pv.Box(bounds=(-size / 2, size / 2, -size / 2, size / 2, 0, 0.35))
    rot = np.c_[u, v, normal]
    plate.points = plate.points @ rot.T + c
    pitch = size / n
    g = (np.arange(n) - (n - 1) / 2) * pitch
    pts = np.array([c + 0.4 * normal + a * u + b * v for a in g for b in g])
    bumps = pv.PolyData(pts).glyph(geom=pv.Sphere(radius=0.11, theta_resolution=10, phi_resolution=10),
                                   orient=False, scale=False)
    return plate, bumps


def render_brain(out, subjects_dir="G:/mne_data/subjects", size=(2400, 1800)):
    """fsaverage pial surface with the precentral gyrus (Desikan-Killiany atlas, FreeSurfer aparc) coloured and two
    array footprints on its crown near the hand-knob coordinates. Writes the image and the sites' pixel positions."""
    import json

    import nibabel as nib
    surf = os.path.join(subjects_dir, "fsaverage", "surf")
    coords, faces = nib.freesurfer.read_geometry(os.path.join(surf, "lh.pial"))
    sulc = nib.freesurfer.read_morph_data(os.path.join(surf, "lh.sulc"))
    labels, _, names = nib.freesurfer.read_annot(os.path.join(subjects_dir, "fsaverage", "label", "lh.aparc.annot"))
    names = [n.decode() for n in names]
    precentral = labels == names.index("precentral")
    mesh = pv.PolyData(coords, np.c_[np.full(len(faces), 3), faces].ravel())
    mesh = mesh.compute_normals(auto_orient_normals=True)
    w = np.clip((sulc - np.percentile(sulc, 5)) / (np.percentile(sulc, 95) - np.percentile(sulc, 5)), 0, 1)
    rgb = (1 - w)[:, None] * TISSUE_LIGHT + w[:, None] * TISSUE_DARK
    motor = np.array([int(MOTOR_COLOR[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    shade = (1 - 0.6 * w)[:, None]                                       # keep the folds visible inside the region
    rgb[precentral] = (motor * shade)[precentral]
    plates, centers = [], []
    for target in SITES:
        crown = precentral & (sulc < np.percentile(sulc[precentral], 25))  # crown of the precentral gyrus
        cand = np.flatnonzero(crown)
        c = coords[cand[np.argmin(np.linalg.norm(coords[cand] - target, axis=1))]]
        d = np.linalg.norm(coords - c, axis=1)
        normal = 0.7 * mesh.point_data["Normals"][d < 3.0].mean(0) + 0.3 * np.array([-0.85, 0.0, 0.53])
        plates.append(_array_plate(c, normal, lift=0.3))
        centers.append(c)
    mesh.point_data["rgb"] = (rgb * 255).astype(np.uint8)
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True, specular=0.25, specular_power=18, ambient=0.18,
                diffuse=0.85)
    for plate, bumps in plates:
        pl.add_mesh(plate, color="#ffffff", smooth_shading=False, specular=0.3, ambient=0.45)
        pl.add_mesh(bumps, color="#3a3f47", smooth_shading=True, specular=0.5)
    centre = coords.mean(0)
    el = np.radians(12)
    pl.camera.position = centre + 520 * np.array([-np.cos(el), -0.12, np.sin(el)])
    pl.camera.focal_point = centre + np.array([0, -4, 4])
    pl.camera.up = (0, 0, 1)
    pl.camera.view_angle = 17
    pl.enable_anti_aliasing("ssaa")
    path = os.path.join(out, "brain_3d.png")
    pl.screenshot(path, transparent_background=True)
    import vtk
    pix = []
    for c in centers:
        co = vtk.vtkCoordinate()
        co.SetCoordinateSystemToWorld()
        co.SetValue(*c)
        x, y = co.GetComputedDoubleDisplayValue(pl.renderer)
        pix.append([x, size[1] - y])                                       # image rows count from the top
    json.dump({"size": list(size), "sites_px": pix}, open(os.path.join(out, "brain_3d_sites.json"), "w"))
    pl.close()
    print("saved", path, "sites", np.round(pix, 1).tolist())


def render_utah(out, size=(2000, 1600)):
    n, pitch, length = 10, 0.4, 1.5
    g = (np.arange(n) - (n - 1) / 2) * pitch
    base = pv.Box(bounds=(-2.0, 2.0, -2.0, 2.0, -0.3, 0.0))
    shank = pv.Cone(center=(0, 0, length / 2), direction=(0, 0, 1), height=length, radius=0.045, resolution=40,
                    capping=True)
    tip = pv.Cone(center=(0, 0, length - 0.09), direction=(0, 0, 1), height=0.2, radius=0.016, resolution=30)
    pts = pv.PolyData(np.array([(x, y, 0.0) for x in g for y in g]))
    shanks = pts.glyph(geom=shank, orient=False, scale=False)
    tips = pts.glyph(geom=tip, orient=False, scale=False)
    # wire bundle: a flat gold ribbon leaving one edge of the base and curving away
    t = np.linspace(0, 1, 200)
    path = np.c_[0.4 * np.sin(2.0 * t), -2.0 - 3.6 * t, -0.15 - 1.2 * t ** 2]
    ribbon = pv.Spline(path, 400).ribbon(width=0.55, normal=(0, 0, 1))
    wires = [pv.Spline(path + np.array([dx, 0, 0.03]), 300).tube(radius=0.03) for dx in np.linspace(-0.4, 0.4, 8)]
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.add_mesh(base, color="#585d65", specular=0.6, specular_power=30, ambient=0.2)
    pl.add_mesh(shanks, color="#3b3f46", specular=0.9, specular_power=45, ambient=0.15, smooth_shading=True)
    pl.add_mesh(tips, color="#f2f2f2", specular=1.0, specular_power=60, ambient=0.6, smooth_shading=True)
    pl.add_mesh(ribbon, color="#8a6a2e", specular=0.7, specular_power=30, ambient=0.1)
    for wtube in wires:
        pl.add_mesh(wtube, color="#c99a3c", specular=1.0, specular_power=50, ambient=0.12, smooth_shading=True)
    pl.add_light(pv.Light(position=(-6, -8, 10), focal_point=(0, 0, 0), intensity=1.0))
    pl.add_light(pv.Light(position=(8, -2, 6), focal_point=(0, 0, 0), intensity=0.5))
    pl.camera.position = (-7.5, -9.5, 7.0)
    pl.camera.focal_point = (0.3, -1.2, 0.3)
    pl.camera.up = (0, 0, 1)
    pl.camera.view_angle = 30
    pl.enable_anti_aliasing("ssaa")
    path_out = os.path.join(out, "utah_3d.png")
    pl.screenshot(path_out, transparent_background=True)
    pl.close()
    print("saved", path_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="*", default=["brain", "utah"])
    ap.add_argument("--out", default="results/figures/v2")
    args = ap.parse_args()
    if "brain" in args.what:
        render_brain(args.out)
    if "utah" in args.what:
        render_utah(args.out)


if __name__ == "__main__":
    main()
