"""Oracle-ladder building blocks for drift anatomy (see scripts/drift_anatomy.py for the protocol)."""
import numpy as np
import torch

from .data import link
from .linear import LagDecoder, lagged, ridge, ridge_to_prior
from .metrics import r2_per_dof
from .preprocess import ZScore, split_bins

N_LAGS, ALPHA, K_LAT = 8, 0.1, 16
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMS = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])  # used by remap_locality.py


class Sess:
    def __init__(self, key):
        s = link.load_session(key)
        self.key, self.day, self.style = key, s.day, s.style
        tr, te = split_bins(s.trial_start, s.sbp.shape[0])
        x_tr, x_te = s.sbp[tr], s.sbp[te]
        self.y_tr, self.y_te = s.kin[tr], s.kin[te]
        self.alive = s.tc.mean(0) / link.BIN_S > 2.0   # channels with threshold-crossing activity (> 2 Hz)
        starts = np.r_[s.trial_start[:300], tr.stop]
        self.trial_id = np.repeat(np.arange(300), np.diff(starts))  # trial index of each training bin
        self.z = ZScore().fit(x_tr)
        # only the z-scored copies are stored (memory); raw features are recomputed on demand
        self.ztr, self.zte = self.z.transform(x_tr).astype(np.float32), self.z.transform(x_te).astype(np.float32)
        self.dec = LagDecoder.fit(self.ztr, self.y_tr, N_LAGS, ALPHA)
        self.V = np.linalg.svd(self.ztr, full_matrices=False)[2][:K_LAT].T  # (C, k)

    @property
    def x_tr(self):
        return self.ztr * self.z.std + self.z.mean

    @property
    def x_te(self):
        return self.zte * self.z.std + self.z.mean


class SessMeta:
    """Session key, day and style without loading data (for pair selection)."""

    def __init__(self, key):
        self.key, self.style = key, key[-2:]
        self.day = int((np.datetime64(key[:10]) - np.datetime64("2020-01-27")).astype(int))


class SessCache:
    """Load Sess objects on demand, keeping at most ``maxsize`` in memory (least recently used evicted)."""

    def __init__(self, maxsize=120):
        from collections import OrderedDict
        self.maxsize, self.d = maxsize, OrderedDict()

    def __getitem__(self, key):
        if key in self.d:
            self.d.move_to_end(key)
        else:
            self.d[key] = Sess(key)
            if len(self.d) > self.maxsize:
                self.d.popitem(last=False)
        return self.d[key]


def r2(pred, y):
    return float(r2_per_dof(pred, y).mean())


def lstsq_shrunk(F, y, n_free, anchor, lam):
    """argmin ||y - F theta||^2 + lam * ||theta[:p] - anchor||^2 where the last n_free params are unpenalised."""
    p = F.shape[1] - n_free
    A = F.T @ F
    A[:p, :p] += lam * np.eye(p)
    rhs = F.T @ y
    rhs[:p] += lam * anchor
    return np.linalg.solve(A, rhs)


def pick_lam(fit_fn, eval_fn, idx_fit, idx_val):
    errs = [eval_fn(fit_fn(idx_fit, lam), idx_val) for lam in LAMS]
    return LAMS[int(np.argmin(errs))]


# ---- L3: per-channel gain ------------------------------------------------------------------------------------
def gain_features(Z, dec):
    """F[t, c, k] = sum_l z_c(t-l) W[l, c, k]; prediction = sum_c g_c F[t, c, k] + d_k."""
    H = lagged(Z, N_LAGS).reshape(len(Z), N_LAGS, -1)
    return np.einsum("tlc,lck->tck", H, dec.W3)


def fit_gain(Fs, Ys, lam):
    T, C, K = Fs.shape
    F = np.concatenate([Fs.transpose(0, 2, 1).reshape(T * K, C), np.tile(np.eye(K), (T, 1))], axis=1)
    th = lstsq_shrunk(F.astype(np.float64), Ys.reshape(-1).astype(np.float64), K, np.ones(C), lam * T)
    return th[:C], th[C:]


def pred_gain(Fs, params):
    g, d = params
    return np.einsum("tck,c->tk", Fs, g) + d


# ---- L4: latent rotation --------------------------------------------------------------------------------------
def latent_features(Z, Vj, Vi, dec):
    """Rotate only the dominant subspace: x~ = (I - Vj Vj^T) z + Vi Q Vj^T z; everything else passes unchanged.

    Returns (G, base): prediction = base + sum_ab Q_ab G[:, ab, :] + d, where base is the (bias-free) decoder output
    on the complement part and G[t, a, b, k] = sum_l u_b(t-l) (Vi^T W_l)[a, k] with u = Vj^T z.
    """
    U = Z @ Vj
    Hu = lagged(U, N_LAGS).reshape(len(Z), N_LAGS, -1)        # (T, L, k)
    What = np.einsum("ca,lck->lak", Vi, dec.W3)               # (L, k, K)
    G = np.einsum("tlb,lak->tabk", Hu, What).reshape(len(Z), -1, What.shape[-1])
    base = lagged(Z - U @ Vj.T, N_LAGS) @ dec.W
    return G, base


def fit_latent(G, base, Ys, anchor, lam):
    T, P, K = G.shape
    F = np.concatenate([G.transpose(0, 2, 1).reshape(T * K, P), np.tile(np.eye(K), (T, 1))], axis=1)
    th = lstsq_shrunk(F.astype(np.float64), (Ys - base).reshape(-1).astype(np.float64), K, anchor.reshape(-1), lam * T)
    return th[:P], th[P:]


def pred_latent(G, base, params):
    q, d = params
    return base + np.einsum("tpk,p->tk", G, q) + d


# ---- L5: full input remap (torch, L-BFGS) ----------------------------------------------------------------------
def fit_remap(Z, Y, dec, lam, iters=60, mask=None):
    """Learn x~ = (I + D * mask) z + h in front of the frozen decoder; penalty lam * (||D||^2 + ||h||^2).

    ``mask`` (C, C) restricts which channel pairs may mix (None = full map)."""
    Zt = torch.tensor(Z, device=DEV)
    Yt = torch.tensor(Y, device=DEV)
    C = Z.shape[1]
    W = torch.tensor(dec.W3, device=DEV)                      # (L, C, K)
    b = torch.tensor(dec.b, device=DEV)
    mk = torch.ones(C, C, device=DEV) if mask is None else torch.tensor(mask, device=DEV, dtype=torch.float32)
    D = torch.zeros(C, C, device=DEV, requires_grad=True)
    h = torch.zeros(C, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([D, h], max_iter=iters, line_search_fn="strong_wolfe")
    eye = torch.eye(C, device=DEV)

    def predict(Zin):
        X = Zin @ (eye + D * mk).T + h
        P = b.expand(len(X), -1).clone()
        for l in range(N_LAGS):
            P[l:] += X[: len(X) - l] @ W[l]
        return P

    def closure():
        opt.zero_grad()
        # same scaling as the closed-form rungs: mean over bins of summed squared error + lam * ||D||^2
        loss = ((predict(Zt) - Yt) ** 2).sum(1).mean() + lam * (((D * mk) ** 2).sum() + (h ** 2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return (D.detach(), h.detach(), predict)


def pred_remap(Z, params):
    M, h, predict = params
    with torch.no_grad():
        return predict(torch.tensor(Z, device=DEV)).cpu().numpy()


def holdout_split(trial_id, n):
    """Labelled bins = first n trials; fit on the first 80% of them, validate on the last 20%."""
    lab = trial_id < n
    cut = max(1, int(round(n * 0.8)))
    return lab, lab & (trial_id < cut), lab & (trial_id >= cut)


def cv_pick(fit_fn, err_fn, trial_id, n, grid, k=5):
    """Pick a regularization value by k-fold CV over contiguous blocks of the first n trials.

    fit_fn(mask, lam) -> params; err_fn(params, mask) -> summed squared error on mask."""
    k = max(2, min(k, n // 2))
    blocks = np.array_split(np.arange(n), k)
    err = np.zeros(len(grid))
    for blk in blocks:
        val = np.isin(trial_id, blk)
        fit = (trial_id < n) & ~val
        for g_i, lam in enumerate(grid):
            err[g_i] += err_fn(fit_fn(fit, lam), val)
    return grid[int(np.argmin(err))]


# ---- fast CV for shrunk least squares (one eigendecomposition per fold, all regularization values) ---------------
def _fold_blocks(trial_id, n, k):
    k = max(2, min(k, n // 2))
    return [np.isin(trial_id, blk) for blk in np.array_split(np.arange(n), k)]


def _with_intercept(F):
    """Append intercept columns: (T, P, K) -> (T, P+K, K) with one-hot ones; (T, P) -> (T, P+1)."""
    T = F.shape[0]
    if F.ndim == 3:
        K = F.shape[2]
        ones = np.broadcast_to(np.eye(K, dtype=F.dtype), (T, K, K))
        return np.concatenate([F, ones], axis=1)
    return np.concatenate([F, np.ones((T, 1), F.dtype)], axis=1)


def _eig_solver(F, r, fit, center=True):
    """Normal equations on rows ``fit`` (centered unless center=False). F (T, P, K) stacked-output features or
    (T, P) shared features with r (T, K). Returns (evals, evecs, Q^T F^T r, means) for (A + lam I) theta = F^T r + lam*anchor."""
    if not center:
        Fz = np.zeros(F.shape[1:], F.dtype)
        rz = np.zeros(r.shape[1], r.dtype)
        if F.ndim == 3:
            Ff = F[fit].transpose(2, 0, 1).reshape(-1, F.shape[1]).astype(np.float64)
            A, b = Ff.T @ Ff, Ff.T @ r[fit].T.reshape(-1).astype(np.float64)
        else:
            Ff = F[fit].astype(np.float64)
            A, b = Ff.T @ Ff, Ff.T @ r[fit]
        ev, Q = np.linalg.eigh(A)
        return ev, Q, Q.T @ b, Fz, rz
    if F.ndim == 3:   # stacked outputs share theta: sum_k Fc_k^T Fc_k
        Fm = F[fit].mean(0)                                   # (P, K)
        rm = r[fit].mean(0)                                   # (K,)
        Fc = (F[fit] - Fm).transpose(2, 0, 1).reshape(-1, F.shape[1]).astype(np.float64)
        rc = (r[fit] - rm).T.reshape(-1).astype(np.float64)
        A, b = Fc.T @ Fc, Fc.T @ rc
    else:             # shared features, separate weights per output (ridge)
        Fm, rm = F[fit].mean(0), r[fit].mean(0)
        Fc = (F[fit] - Fm).astype(np.float64)
        A, b = Fc.T @ Fc, Fc.T @ (r[fit] - rm)
    ev, Q = np.linalg.eigh(A)
    return ev, Q, Q.T @ b, Fm, rm


def _solve(eig, lam, anchor):
    ev, Q, Qtb, Fm, rm = eig
    rhs = Qtb + lam * (Q.T @ anchor)
    theta = Q @ (rhs / (ev + lam)[:, None] if rhs.ndim == 2 else rhs / (ev + lam))
    return theta, Fm, rm


def _predict(F, theta, Fm, rm):
    if F.ndim == 3:
        return np.einsum("tpk,p->tk", F - Fm, theta) + rm
    return (F - Fm) @ theta + rm


def cv_shrunk(F, r, trial_id, n, grid, anchor, k=5, scale_by_rows=True, intercept_anchor=None):
    """k-fold CV (contiguous trial blocks within the first n trials) for argmin ||r - F theta - d||^2 + lam' ||theta - anchor||^2
    with lam' = lam * n_fit_rows (if scale_by_rows). Intercepts d are unpenalised, unless ``intercept_anchor`` (K,) is
    given: then d is shrunk toward it with the same lam' (everything shrinks toward the previous decoder, so tiny
    calibration sets cannot shift the output offset). Returns (best_lam, predict_fn(F_new) fitted on all n trials)."""
    lab = trial_id < n
    center = intercept_anchor is None
    if not center:
        ia = np.asarray(intercept_anchor, dtype=np.float64)
        anchor = (np.concatenate([anchor, ia]) if F.ndim == 3 else np.vstack([anchor, ia[None]]))
        F = _with_intercept(F)
    err = np.zeros(len(grid))
    for val in _fold_blocks(trial_id, n, k):
        fit = lab & ~val
        eig = _eig_solver(F, r, fit, center)
        mult = fit.sum() if scale_by_rows else 1.0
        for g_i, lam in enumerate(grid):
            th, Fm, rm = _solve(eig, lam * mult, anchor)
            err[g_i] += float(((_predict(F[val], th, Fm, rm) - r[val]) ** 2).sum())
    best = grid[int(np.argmin(err))]
    eig = _eig_solver(F, r, lab, center)
    th, Fm, rm = _solve(eig, best * (lab.sum() if scale_by_rows else 1.0), anchor)
    if center:
        return best, (lambda Fn: _predict(Fn, th, Fm, rm))
    return best, (lambda Fn: _predict(_with_intercept(Fn), th, Fm, rm))


LAMS_CF = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])  # closed-form input corrections
LAMS_REMAP = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0])            # L-BFGS remap (coarser grid)
ALPHAS = np.array([0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6])          # ridge / ridge-to-prior


def procrustes_latent_map(Vj, Vi):
    """Q mapping day-j latent coordinates to day-i coordinates (Vj R ~ Vi  =>  Q = R^T)."""
    U_, _, Wt_ = np.linalg.svd(Vj.T @ Vi)
    return (U_ @ Wt_).T


def stable_procrustes_map(Vj, Vi, keep=0.6, iters=5):
    """Degenhart-style alignment: Procrustes on the loadings of 'stable' channels only.

    Iteratively fits R on the current stable set and keeps the ``keep`` fraction of channels with the smallest
    loading residual ||Vj[c] R - Vi[c]||. Returns Q = R^T (day-j latent coords -> day-i coords) and the stable mask."""
    C = Vj.shape[0]
    stable = np.ones(C, bool)
    for _ in range(iters):
        U_, _, Wt_ = np.linalg.svd(Vj[stable].T @ Vi[stable])
        R = U_ @ Wt_
        res = np.linalg.norm(Vj @ R - Vi, axis=1)
        stable = res <= np.quantile(res, keep)
    return R.T, stable


def ladder(si: Sess, sj: Sess, n_list, k_folds=5, k_folds_remap=3):
    out = {}
    zf = (sj.x_te - si.z.mean) / si.z.std
    out["L0"] = r2(si.dec.predict(zf), sj.y_te)
    out["L1"] = r2(si.dec.predict((sj.x_te - sj.z.mean) / si.z.std), sj.y_te)
    out["L2"] = r2(si.dec.predict(sj.zte), sj.y_te)
    # unsupervised Procrustes latent alignment (L4u)
    Q0 = procrustes_latent_map(sj.V, si.V)
    G_te, B_te = latent_features(sj.zte, sj.V, si.V, si.dec)
    out["L4u"] = r2(pred_latent(G_te, B_te, (Q0.reshape(-1), si.dec.b)), sj.y_te)
    Qs, _ = stable_procrustes_map(sj.V, si.V)
    out["L4s"] = r2(pred_latent(G_te, B_te, (Qs.reshape(-1), si.dec.b)), sj.y_te)

    F_tr, F_te = gain_features(sj.ztr, si.dec), gain_features(sj.zte, si.dec)
    G_tr, B_tr = latent_features(sj.ztr, sj.V, si.V, si.dec)
    H_tr = lagged(sj.ztr, N_LAGS)
    H_te = lagged(sj.zte, N_LAGS)
    Y = sj.y_tr
    tid = sj.trial_id

    def sse(p, m):
        return float(((p - Y[m]) ** 2).sum())

    C = sj.ztr.shape[1]
    for n in n_list:
        lab = tid < n
        lam, f = cv_shrunk(F_tr, Y, tid, n, LAMS_CF, np.ones(C), k_folds, intercept_anchor=si.dec.b)
        out[f"L3_n{n}"] = r2(f(F_te), sj.y_te)
        out[f"lam_L3_n{n}"] = float(lam)

        lam, f = cv_shrunk(G_tr, Y - B_tr, tid, n, LAMS_CF, Q0.reshape(-1), k_folds, intercept_anchor=si.dec.b)
        out[f"L4_n{n}"] = r2(B_te + f(G_te), sj.y_te)
        out[f"lam_L4_n{n}"] = float(lam)

        lam = cv_pick(lambda m, l: fit_remap(sj.ztr[m], Y[m], si.dec, l),
                      lambda prm, m: sse(pred_remap(sj.ztr[m], prm), m), tid, n, LAMS_REMAP, k_folds_remap)
        out[f"L5_n{n}"] = r2(pred_remap(sj.zte, fit_remap(sj.ztr[lab], Y[lab], si.dec, lam)), sj.y_te)
        out[f"lam_L5_n{n}"] = float(lam)

        a, f = cv_shrunk(H_tr, Y, tid, n, ALPHAS, np.zeros((H_tr.shape[1], Y.shape[1])), k_folds, scale_by_rows=False)
        out[f"L6_n{n}"] = r2(f(H_te), sj.y_te)
        out[f"lam_L6_n{n}"] = float(a)

        a, f = cv_shrunk(H_tr, Y, tid, n, ALPHAS, si.dec.W.astype(np.float64), k_folds, scale_by_rows=False,
                         intercept_anchor=si.dec.b)
        out[f"L6p_n{n}"] = r2(f(H_te), sj.y_te)
        out[f"lam_L6p_n{n}"] = float(a)
    out["own"] = r2(sj.dec.predict(sj.zte), sj.y_te)
    return out


