"""Reference methods: the correction ladder of the paper, written as benchmark methods (``method(task) -> prediction``).

They reuse the day-i decoder fitted exactly as in the paper (ridge, 8 causal lags, penalty 1e4) and the same
cross-validated shrinkage, so they reproduce the paper's numbers (checked by scripts/bench_check.py). External methods
should use only the public Task arrays (src_x, src_y, cal_x, cal_y_lab, eval_x, ...).
"""
import numpy as np

from ..anatomy import (ALPHAS, LAMS_CF, cv_shrunk, gain_features, lagged, latent_features, pred_latent,
                       procrustes_latent_map, N_LAGS)


def none(t):
    """Day-i decoder with day-i normalization (no update)."""
    s, j = t._src, t._tgt
    return s.dec.predict((j.x_te - s.z.mean) / s.z.std)


def mean_update(t):
    """Day-i decoder; feature means re-estimated on unlabelled day-j data."""
    s, j = t._src, t._tgt
    return s.dec.predict((j.x_te - j.z.mean) / s.z.std)


def renorm(t):
    """Day-i decoder; feature means and scales re-estimated on unlabelled day-j data."""
    return t._src.dec.predict(t._tgt.zte)


def procrustes(t):
    """Renormalization plus an unsupervised rotation of the top-16 day-j subspace onto the day-i subspace."""
    s, j = t._src, t._tgt
    Q0 = procrustes_latent_map(j.V, s.V)
    G, B = latent_features(j.zte, j.V, s.V, s.dec)
    return pred_latent(G, B, (Q0.reshape(-1), s.dec.b))


def reweight(t, k_folds=5):
    """Renormalization plus one gain per channel and an offset per output, fitted on the labelled trials and shrunk
    toward 'no change' by cross-validation (the 'no change' anchor is a candidate)."""
    s, j = t._src, t._tgt
    C = j.ztr.shape[1]
    _, f = cv_shrunk(gain_features(j.ztr, s.dec), j.y_tr, j.trial_id, t.n_labels, LAMS_CF, np.ones(C), k_folds,
                     intercept_anchor=s.dec.b)
    return f(gain_features(j.zte, s.dec))


def prior_refit(t, k_folds=5):
    """Ridge decoder refitted on the labelled trials, shrunk toward the day-i decoder (penalty by cross-validation)."""
    s, j = t._src, t._tgt
    _, f = cv_shrunk(lagged(j.ztr, N_LAGS), j.y_tr, j.trial_id, t.n_labels, ALPHAS, s.dec.W.astype(np.float64),
                     k_folds, scale_by_rows=False, intercept_anchor=s.dec.b)
    return f(lagged(j.zte, N_LAGS))


def retrain(t, k_folds=5):
    """New ridge decoder on the labelled trials only (penalty by cross-validation)."""
    j = t._tgt
    H = lagged(j.ztr, N_LAGS)
    _, f = cv_shrunk(H, j.y_tr, j.trial_id, t.n_labels, ALPHAS, np.zeros((H.shape[1], j.y_tr.shape[1])), k_folds,
                     scale_by_rows=False)
    return f(lagged(j.zte, N_LAGS))


LABEL_FREE = {"none": none, "mean_update": mean_update, "renorm": renorm, "procrustes": procrustes}
SUPERVISED = {"reweight": reweight, "prior_refit": prior_refit, "retrain": retrain}
# column of the paper's ladder (results/decay_alpha/<subject>_alpha10000.csv) that each method reproduces
LADDER_COLUMN = {"none": "L0", "mean_update": "L1", "renorm": "L2", "procrustes": "L4u",
                 "reweight": "L3_n{n}", "prior_refit": "L6p_n{n}", "retrain": "L6_n{n}"}
