"""Loader for FALCON H2 (DANDI 000950): human participant T5, attempted handwriting, 192 channels (2 x 96 Utah
arrays), threshold crossings in 20 ms bins, sessions 2022-05-18 .. 2023-10-09 (~17 months).

Only the calibration splits carry full sessions (held-in-calib: 21 sessions; held-out-calib: 5 short sessions);
we use them for channel-level health statistics (activity over time, death/revival, abrupt changes). There is no
electrode grid map in the files, so spatial analyses are not possible on H2.
"""
import glob
import os
import re

import numpy as np

from .. import DATA_ROOT

H2_DIR = os.path.join(DATA_ROOT, "falcon_h2")
BIN_S = 0.02


def list_files(split: str = "calib", h2_dir: str = H2_DIR):
    """NWB files of the calibration splits (held-in and held-out), sorted by date."""
    files = glob.glob(os.path.join(h2_dir, f"*-{split}_ses-*.nwb"))
    return sorted(files, key=lambda p: re.search(r"ses-(\d{8})", p).group(1))


def load_tc(path: str):
    """Return (date 'YYYY-MM-DD', tc (T, 192) float32 counts per 20 ms bin, kept-bin mask)."""
    from pynwb import NWBHDF5IO

    d = re.search(r"ses-(\d{8})", path).group(1)
    with NWBHDF5IO(path, "r", load_namespaces=True) as io:
        n = io.read()
        tc = np.asarray(n.acquisition["binned_spikes"].data[:], dtype=np.float32)
        keep = np.asarray(n.acquisition["eval_mask"].data[:], dtype=bool) if "eval_mask" in n.acquisition else None
    return f"{d[:4]}-{d[4:6]}-{d[6:]}", tc, keep
