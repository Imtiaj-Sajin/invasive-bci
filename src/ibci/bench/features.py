"""Alternative neural features for the benchmark: threshold-crossing counts instead of spike-band power.

Pretrained decoders such as NDT3 take threshold-crossing counts in 20 ms bins, so a fair comparison needs the linear
reference methods on the same features. Sessions built here have the same interface and the same train/evaluation
split as the spike-band-power sessions of the paper (scripts/replicate_braingate_decoding.py, ibci.anatomy.Sess):

  * BrainGate: -4.5 x RMS crossings ('tx_4_5'), summed over pairs of 10 ms bins (the paper averages spike-band power
    over the same pairs), same trial segmentation and channel order;
  * monkey N (LINK): the threshold crossings stored with each session (already 20 ms bins);
  * monkeys C and M: the paper already uses summed spike counts, so their sessions are unchanged.
"""
import glob
import os
import re
import sys

import numpy as np
from scipy.io import loadmat

from ..anatomy import K_LAT, N_LAGS
from ..data import link
from ..linear import LagDecoder
from ..preprocess import ZScore, split_bins

_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts"))


def braingate_tx(path, field="tx_4_5"):
    """(day, X (T20, C) crossings per 20 ms bin, Y, trial index); segmentation as in load_session of the paper."""
    sys.path.insert(0, _SCRIPTS)
    from replicate_braingate_decoding import channel_order
    d = loadmat(path, simplify_cells=True)
    day = int(d["post_implant_day"])
    tx = np.asarray(d["neural"][field], dtype=np.float32)[:, channel_order(d["electrodes"])]
    cur = np.asarray(d["cursor_position"], dtype=np.float32)
    tgt = np.asarray(d["target_position"], dtype=np.float32)
    tr = d["trials"]
    starts, ends = np.atleast_1d(tr["trial_start_index"]).astype(int), np.atleast_1d(tr["trial_end_index"]).astype(int)
    xs, ys, ts = [], [], []
    for k, (a, b) in enumerate(zip(starts, ends)):
        b = a + 2 * ((b - a) // 2)
        if b - a < 20:
            continue
        xs.append(tx[a:b].reshape(-1, 2, tx.shape[1]).sum(1))
        v = (tgt[a:b] - cur[a:b]).reshape(-1, 2, 2).mean(1)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        ys.append(np.where(n > 1e-6, v / np.maximum(n, 1e-6), 0.0))
        ts.append(np.full(len(xs[-1]), k))
    X, Y, T = np.concatenate(xs), np.concatenate(ys).astype(np.float32), np.concatenate(ts)
    return day, X, Y, np.unique(T, return_inverse=True)[1]


class LinkTxSess:
    """ibci.anatomy.Sess with LINK threshold crossings in place of spike-band power."""

    def __init__(self, key, alpha):
        s = link.load_session(key)
        self.key, self.day, self.style = key, s.day, s.style
        tr, te = split_bins(s.trial_start, s.tc.shape[0])
        x_tr, x_te = s.tc[tr].astype(np.float32), s.tc[te].astype(np.float32)
        self.y_tr, self.y_te = s.kin[tr], s.kin[te]
        starts = np.r_[s.trial_start[:300], tr.stop]
        self.trial_id = np.repeat(np.arange(300), np.diff(starts))
        self.z = ZScore().fit(x_tr)
        self.ztr, self.zte = self.z.transform(x_tr).astype(np.float32), self.z.transform(x_te).astype(np.float32)
        self.dec = LagDecoder.fit(self.ztr, self.y_tr, N_LAGS, alpha)
        self.V = np.linalg.svd(self.ztr, full_matrices=False)[2][:K_LAT].T

    @property
    def x_te(self):
        return self.zte * self.z.std + self.z.mean


def sessions_tx(subject, alpha, keys):
    """Threshold-crossing sessions keyed like the spike-band-power ones (lazy for BrainGate)."""
    sys.path.insert(0, _SCRIPTS)
    from decay_alpha import LazySessions, sessions_for
    if subject in ("C", "M"):
        return sessions_for(subject, alpha, keys)
    if subject == "N":
        return LazySessions({k: k for k in keys}, lambda key, _: LinkTxSess(key, alpha))
    from replicate_braingate_decoding import BGSess
    root = os.path.join(os.environ.get("IBCI_DATA", "D:/ibci-data"), "braingate", "decoding", subject)
    paths = {}
    for f in glob.glob(os.path.join(root, "*_decoding.mat")):
        day = int(re.search(r"_day_(\d+)_", f).group(1))
        key = f"{subject}_day{day}"
        if key in keys:
            paths[key] = f
    return LazySessions(paths, lambda key, f: BGSess(key, *braingate_tx(f), alpha))

