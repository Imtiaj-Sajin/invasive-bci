"""Loader for the Perich/Miller long-term reaching dataset (DANDI 000688).

Subjects: C (Chewie, 68 sessions 2013-10..2016-10), M (Mihili, 28 sessions 2014-01..2015-06), T (MrT, 12 sessions,
3 weeks), J (Jaco, 3 sessions). Tasks: CO (center-out) and RT (random target). Units are spike-sorted; to obtain
channel-level features comparable to LINK threshold crossings we sum the spikes of all units on an electrode,
in 20 ms bins, over the span from the first trial start to the last trial stop. Kinematics are cursor position and
velocity (x, y), linearly interpolated from 100 Hz to bin centres. Channels are identified by electrode label
(e.g. ``elecM1bankApin1``), which is stable across sessions of a subject.

Cache: <IBCI_DATA>/perich_cache/<subject>_<task>_<date>.npz with
    counts (T, n_ch) float32, labels (n_ch,), kin (T, 4) [x, y, vx, vy], t (T,), trial_start/trial_stop (bins),
    result (n_trials,) str, date, task, subject
"""
import glob
import os
import re
from dataclasses import dataclass

import numpy as np

from .. import DATA_ROOT

PERICH_DIR = os.path.join(DATA_ROOT, "perich")
CACHE_DIR = os.path.join(DATA_ROOT, "perich_cache")
BIN_S = 0.02
TRAIN_FRAC = 0.8


@dataclass
class PSession:
    subject: str
    task: str
    date: str
    counts: np.ndarray
    labels: np.ndarray
    kin: np.ndarray
    t: np.ndarray
    trial_start: np.ndarray
    trial_stop: np.ndarray
    result: np.ndarray

    @property
    def key(self):
        return f"{self.subject}_{self.task}_{self.date}"


def _convert(path: str) -> dict:
    from pynwb import NWBHDF5IO

    m = re.search(r"sub-(\w)_ses-(\w+)-(\d{8})", os.path.basename(path))
    subject, task, date = m.group(1), m.group(2), m.group(3)
    with NWBHDF5IO(path, "r", load_namespaces=True) as io:
        n = io.read()
        trials = n.trials.to_dataframe()
        t0, t1 = float(trials.start_time.min()), float(trials.stop_time.max())
        edges = np.arange(t0, t1 + BIN_S, BIN_S)
        centres = edges[:-1] + BIN_S / 2
        el = n.electrodes.to_dataframe()
        labels = np.array(el["label"].astype(str).tolist())
        counts = np.zeros((len(centres), len(labels)), dtype=np.float32)
        units = n.units.to_dataframe()
        row_of = {idx: r for r, idx in enumerate(el.index)}
        for st, ele in zip(units["spike_times"], units["electrodes"]):
            c = row_of[int(ele.index[0])]
            counts[:, c] += np.histogram(np.asarray(st), bins=edges)[0]
        pos = n.processing["behavior"]["Position"].spatial_series["cursor_pos"]
        vel = n.processing["behavior"]["Velocity"].time_series["cursor_vel"]
        kin = []
        for ts in (pos, vel):
            tt = np.asarray(ts.timestamps[:])
            d = np.asarray(ts.data[:], dtype=np.float64)
            kin.append(np.stack([np.interp(centres, tt, d[:, k]) for k in range(2)], axis=1))
        kin = np.concatenate(kin, axis=1).astype(np.float32)
        ts_ = np.searchsorted(centres, trials.start_time.values)
        te_ = np.searchsorted(centres, trials.stop_time.values)
        result = np.array(trials["result"].astype(str).tolist())
    return dict(counts=counts, labels=labels, kin=kin, t=centres, trial_start=ts_, trial_stop=te_, result=result,
                date=f"{date[:4]}-{date[4:6]}-{date[6:]}", task=task, subject=subject)


def build_cache(src_dir: str = PERICH_DIR, cache_dir: str = CACHE_DIR, verbose=True):
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for p in sorted(glob.glob(os.path.join(src_dir, "*.nwb"))):
        m = re.search(r"sub-(\w)_ses-(\w+)-(\d{8})", os.path.basename(p))
        key = f"{m.group(1)}_{m.group(2)}_{m.group(3)[:4]}-{m.group(3)[4:6]}-{m.group(3)[6:]}"
        dst = os.path.join(cache_dir, key + ".npz")
        if not os.path.exists(dst):
            np.savez_compressed(dst, **_convert(p))
            if verbose:
                print("cached", key, flush=True)
        out.append(dst)
    return out


def list_sessions(subject: str = None, cache_dir: str = CACHE_DIR):
    keys = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(cache_dir, "*.npz")))
    return [k for k in keys if subject is None or k.startswith(subject + "_")]


def load_session(key: str, cache_dir: str = CACHE_DIR) -> PSession:
    z = np.load(os.path.join(cache_dir, key + ".npz"), allow_pickle=False)
    return PSession(subject=str(z["subject"]), task=str(z["task"]), date=str(z["date"]), counts=z["counts"],
                    labels=z["labels"], kin=z["kin"], t=z["t"], trial_start=z["trial_start"],
                    trial_stop=z["trial_stop"], result=z["result"])


def aligned_counts(sessions):
    """Put every session's channels in a common label order (union of labels; absent channels are zeros)."""
    labels = sorted(set().union(*[set(s.labels.tolist()) for s in sessions]))
    idx = {l: i for i, l in enumerate(labels)}
    out = []
    for s in sessions:
        X = np.zeros((len(s.counts), len(labels)), dtype=np.float32)
        X[:, [idx[l] for l in s.labels]] = s.counts
        out.append(X)
    return out, labels


def split_bins(s: PSession, train_frac: float = TRAIN_FRAC):
    """Train = bins before the start of trial round(train_frac * n_trials); test = the rest."""
    cut = int(s.trial_start[int(round(train_frac * len(s.trial_start)))])
    return slice(0, cut), slice(cut, len(s.counts))
