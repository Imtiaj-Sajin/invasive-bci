"""Year-scale decoder-transfer benchmark for intracortical BCIs (the session pairs of the paper).

Task. A decoder is trained on day i. On a later day j (1 to about 480 days later) a method must predict the user's
movement intention on held-out day-j data. It may use

  * day i: neural features and kinematics of the training segment (``src_x``, ``src_y``, ``src_trial``);
  * day j: neural features of the training segment, unlabelled (``cal_x``, ``cal_trial``), plus the kinematics of the
    first ``n_labels`` trials only (``cal_y_lab``, on the bins ``cal_lab``). ``n_labels = 0`` is label-free;
  * nothing from the day-j evaluation segment except its neural features (``eval_x``, causal use only).

Score: R2 on the day-j evaluation segment (mean over output dimensions), and retention = R2 / reference R2, where the
reference is a same-day decoder trained on all labelled day-j training trials (ridge, 8 causal lags, penalty 1e4).

Splits are fixed in benchmark/manifests/<subject>.csv: one pair per training session and target gap (1, 7, 30, 120,
480 days; the closest later session within 25% of the gap). A pair counts in the summary only if its reference R2 is
above 0.1 (``valid``; docs/decisions/0003). Features are spike-band power in 20 ms bins; outputs are finger position and
velocity (monkey N), hand velocity (monkeys C and M) or the unit cursor-to-target direction (BrainGate participants).

Usage::

    from ibci import bench
    rows = []
    for t in bench.tasks("T6", n_labels=50):
        pred = my_method(t)                      # (len(t.eval_x), n_outputs)
        rows.append(t.score(pred))
    bench.summarize(rows)
"""
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..metrics import r2_per_dof

GAPS = (1, 7, 30, 120, 480)
BUDGETS = (0, 10, 20, 50, 100, 300)
SUBJECTS = ("N", "C", "M", "T6", "T5", "T9", "T7", "T10", "T8")
REF_ALPHA = 1e4
MIN_REF_R2 = 0.1
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MANIFEST_DIR = os.path.join(_ROOT, "benchmark", "manifests")


def manifest(subject):
    """Fixed session pairs of one individual: train, test, gap_target, days, ref_r2, valid."""
    return pd.read_csv(os.path.join(MANIFEST_DIR, f"{subject}.csv"))


def _raw(sess, part):
    z = sess.ztr if part == "tr" else sess.zte
    return (z * sess.z.std + sess.z.mean).astype(np.float32)


@dataclass
class Task:
    subject: str
    train: str
    test: str
    gap: int
    days: int
    n_labels: int
    valid: bool
    _src: object = field(repr=False)
    _tgt: object = field(repr=False)

    # day i (source)
    @property
    def src_x(self):
        return _raw(self._src, "tr")

    @property
    def src_y(self):
        return self._src.y_tr

    @property
    def src_trial(self):
        return self._src.trial_id

    # day j (target) training segment
    @property
    def cal_x(self):
        return _raw(self._tgt, "tr")

    @property
    def cal_trial(self):
        return self._tgt.trial_id

    @property
    def cal_lab(self):
        """Bins of the first n_labels trials (the only labelled day-j bins)."""
        return self._tgt.trial_id < self.n_labels

    @property
    def cal_y_lab(self):
        return self._tgt.y_tr[self.cal_lab]

    # day j evaluation segment (features only)
    @property
    def eval_x(self):
        return _raw(self._tgt, "te")

    @property
    def n_outputs(self):
        return self._tgt.y_te.shape[1]

    def reference_r2(self):
        return float(r2_per_dof(self._tgt.dec.predict(self._tgt.zte), self._tgt.y_te).mean())

    def score(self, pred, method=None):
        pred = np.asarray(pred)
        assert pred.shape == self._tgt.y_te.shape, f"prediction shape {pred.shape} != {self._tgt.y_te.shape}"
        r2 = float(r2_per_dof(pred, self._tgt.y_te).mean())
        ref = self.reference_r2()
        return {"method": method, "subject": self.subject, "train": self.train, "test": self.test,
                "gap_target": self.gap, "days": self.days, "n_labels": self.n_labels, "valid": self.valid,
                "r2": r2, "ref_r2": ref, "retention": r2 / ref}


def _sessions(subject, keys):
    """Session objects exactly as in the paper's tuned-penalty analysis (scripts/decay_alpha.py)."""
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    from decay_alpha import sessions_for
    return sessions_for(subject, REF_ALPHA, keys)


def tasks(subject, n_labels=0, gaps=GAPS, valid_only=True, limit=None):
    """Yield one Task per manifest pair (in manifest order). Sessions are loaded lazily."""
    assert n_labels in BUDGETS, f"n_labels must be one of {BUDGETS}"
    m = manifest(subject)
    m = m[m.gap_target.isin(gaps)]
    if valid_only:
        m = m[m.valid]
    if limit:
        m = m.head(limit)
    sess = _sessions(subject, set(m.train) | set(m.test))
    for r in m.itertuples():
        yield Task(subject, r.train, r.test, int(r.gap_target), int(r.days), n_labels, bool(r.valid),
                   sess[r.train], sess[r.test])


def summarize(rows, n_boot=2000, seed=0):
    """Per individual and gap: number of pairs, median R2 and median retention with a 95% bootstrap interval that
    resamples training sessions (pairs sharing a training session are not independent)."""
    from ..stats import cluster_bootstrap
    d = pd.DataFrame(rows)
    d = d[d.valid]
    out = []
    for (m, s, n, g), q in d.groupby(["method", "subject", "n_labels", "gap_target"], dropna=False):
        est, lo, hi = cluster_bootstrap(q, "retention", n_boot=n_boot, seed=seed)
        out.append({"method": m, "subject": s, "n_labels": n, "gap_target": g, "pairs": len(q),
                    "r2": float(q.r2.median()), "retention": est, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out)
