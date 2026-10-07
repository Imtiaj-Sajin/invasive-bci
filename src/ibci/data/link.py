"""Loader for the LINK dataset (DANDI 001201).

Each NWB file holds one session of one target style (CO = center-out, RD = random). Some days have both styles,
stored as two files sharing a date. A session is converted once into a compact ``.npz`` cache:

    sbp   (T, 96) float32  spiking-band power, 20 ms bins
    tc    (T, 96) float32  threshold-crossing counts per 20 ms bin
    kin   (T, 4)  float32  [index_pos, mrs_pos, index_vel, mrs_vel]
    t     (T,)    float64  bin timestamps (s)
    trial_start, trial_stop (n_trials,) bin indices; target_pos (n_trials, 2)
    imp   (96,)   float32  electrode impedance (ohm) stored with this session; all-NaN when not stored
    date  str  YYYY-MM-DD; style str CO|RD
"""
import glob
import os
import re
from dataclasses import dataclass

import numpy as np

from .. import DATA_ROOT

LINK_DIR = os.path.join(DATA_ROOT, "link")
CACHE_DIR = os.path.join(DATA_ROOT, "link_cache")
BIN_S = 0.02


@dataclass
class Session:
    date: str
    style: str
    sbp: np.ndarray
    tc: np.ndarray
    kin: np.ndarray
    t: np.ndarray
    trial_start: np.ndarray
    trial_stop: np.ndarray
    target_pos: np.ndarray
    imp: np.ndarray

    @property
    def day(self) -> int:
        """Days since the first LINK session (2020-01-27)."""
        return int((np.datetime64(self.date) - np.datetime64("2020-01-27")).astype(int))

    @property
    def key(self) -> str:
        return f"{self.date}_{self.style}"


def _convert(nwb_path: str) -> dict:
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
        n = io.read()
        by_name = {getattr(o, "name", None): o for o in n.objects.values() if hasattr(o, "data")}
        sbp_ts = by_name["SpikingBandPower"]
        t = np.asarray(sbp_ts.timestamps[:], dtype=np.float64)
        sbp = np.asarray(sbp_ts.data[:], dtype=np.float32)
        tc = np.asarray(by_name["ThresholdCrossings"].data[:], dtype=np.float32)
        kin = np.concatenate(
            [np.asarray(by_name[k].data[:], dtype=np.float32).reshape(-1, 1)
             for k in ("index_position", "mrs_position", "index_velocity", "mrs_velocity")], axis=1)
        trials = n.trials.to_dataframe()
        start = np.searchsorted(t, trials["start_time"].values - 1e-6)
        stop = np.searchsorted(t, trials["stop_time"].values - 1e-6)
        target = np.stack([trials["index_target_position"].values.astype(np.float32),
                           trials["mrs_target_position"].values.astype(np.float32)], axis=1)
        style = str(trials["target_style"].values[0])
        el = n.electrodes.to_dataframe()
        # impedance is not stored for every session; NaN marks "not measured"
        imp = el["imp"].values.astype(np.float32) if "imp" in el.columns else np.full(len(el), np.nan, np.float32)
        date = n.session_start_time.date().isoformat()
    assert sbp.shape == tc.shape and sbp.shape[0] == kin.shape[0] == t.shape[0]
    return dict(sbp=sbp, tc=tc, kin=kin, t=t, trial_start=start, trial_stop=stop,
                target_pos=target, imp=imp, date=date, style=style)


def build_cache(link_dir: str = LINK_DIR, cache_dir: str = CACHE_DIR, verbose: bool = True) -> list:
    """Convert every NWB file to ``<cache_dir>/<date>_<style>.npz``; files already converted are skipped.

    A small ``converted.txt`` index maps NWB file names to cache keys so re-runs do not reopen NWB files.
    Returns the list of cache paths.
    """
    os.makedirs(cache_dir, exist_ok=True)
    index_path = os.path.join(cache_dir, "converted.txt")
    done = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            done = dict(line.split() for line in f if line.strip())
    out = []
    for i, p in enumerate(sorted(glob.glob(os.path.join(link_dir, "*.nwb")))):
        name = os.path.basename(p)
        if name in done and os.path.exists(os.path.join(cache_dir, done[name] + ".npz")):
            out.append(os.path.join(cache_dir, done[name] + ".npz"))
            continue
        d = _convert(p)
        key = f"{d['date']}_{d['style']}"
        dst = os.path.join(cache_dir, key + ".npz")
        np.savez_compressed(dst, **d)
        with open(index_path, "a") as f:
            f.write(f"{name} {key}\n")
        out.append(dst)
        if verbose:
            print(f"[{i}] {name} -> {key} T={d['sbp'].shape[0]}", flush=True)
    return out


def list_sessions(cache_dir: str = CACHE_DIR) -> list:
    """Sorted list of session keys ``YYYY-MM-DD_STYLE`` available in the cache."""
    keys = [os.path.basename(p)[:-4] for p in glob.glob(os.path.join(cache_dir, "*.npz"))]
    return sorted(k for k in keys if re.match(r"\d{4}-\d{2}-\d{2}_(CO|RD)$", k))


def load_session(key: str, cache_dir: str = CACHE_DIR) -> Session:
    z = np.load(os.path.join(cache_dir, key + ".npz"))
    return Session(date=str(z["date"]), style=str(z["style"]), sbp=z["sbp"], tc=z["tc"], kin=z["kin"],
                   t=z["t"], trial_start=z["trial_start"], trial_stop=z["trial_stop"],
                   target_pos=z["target_pos"], imp=z["imp"])


def electrode_layout(link_dir: str = LINK_DIR, cache_dir: str = CACHE_DIR):
    """DataFrame (96 rows, channel order of the features) with array_name, bank, pin, row, col.

    The layout is identical across sessions (checked 2026-10-05): a full 8x8 'Medial' array (64 ch) and the
    4-column half of a 'Lateral' array (32 ch). Cached as ``layout.csv``.
    """
    import pandas as pd

    path = os.path.join(cache_dir, "layout.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    from pynwb import NWBHDF5IO

    first = sorted(glob.glob(os.path.join(link_dir, "*.nwb")))[0]
    with NWBHDF5IO(first, "r", load_namespaces=True) as io:
        e = io.read().electrodes.to_dataframe()[["array_name", "bank", "pin", "row", "col"]].reset_index(drop=True)
    os.makedirs(cache_dir, exist_ok=True)
    e.to_csv(path, index=False)
    return e


def chebyshev_distance(layout) -> np.ndarray:
    """(96, 96) grid distance in electrode steps (400 um pitch); np.inf across different arrays."""
    r, c = layout["row"].to_numpy(int), layout["col"].to_numpy(int)
    a = np.asarray(layout["array_name"].astype(str).tolist())
    d = np.maximum(np.abs(r[:, None] - r[None]), np.abs(c[:, None] - c[None])).astype(float)
    d[a[:, None] != a[None]] = np.inf
    return d
