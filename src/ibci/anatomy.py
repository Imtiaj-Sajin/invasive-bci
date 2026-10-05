"""Oracle-ladder building blocks for drift anatomy (see scripts/drift_anatomy.py for the protocol)."""
import numpy as np
import torch

from .data import link
from .linear import LagDecoder, lagged, ridge, ridge_to_prior
from .metrics import r2_per_dof
from .preprocess import ZScore, split_bins

N_LAGS, ALPHA, K_LAT = 8, 0.1, 16
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMS = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])


class Sess:
    def __init__(self, key):
        s = link.load_session(key)
        self.key, self.day, self.style = key, s.day, s.style
        tr, te = split_bins(s.trial_start, s.sbp.shape[0])
        self.x_tr, self.x_te = s.sbp[tr], s.sbp[te]
        self.y_tr, self.y_te = s.kin[tr], s.kin[te]
        starts = np.r_[s.trial_start[:300], tr.stop]
        self.trial_id = np.repeat(np.arange(300), np.diff(starts))  # trial index of each training bin
        self.z = ZScore().fit(self.x_tr)
        self.ztr, self.zte = self.z.transform(self.x_tr), self.z.transform(self.x_te)
        self.dec = LagDecoder.fit(self.ztr, self.y_tr, N_LAGS, ALPHA)
        self.V = np.linalg.svd(self.ztr, full_matrices=False)[2][:K_LAT].T  # (C, k)


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
    """Learn x~ = (I + D * mask) z + h in front of the frozen decoder; penalty lam * ||D||^2.

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
        loss = ((predict(Zt) - Yt) ** 2).sum(1).mean() + lam * ((D * mk) ** 2).sum()
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


def ladder(si: Sess, sj: Sess, n_list):
    out = {}
    zf = (sj.x_te - si.z.mean) / si.z.std
    out["L0"] = r2(si.dec.predict(zf), sj.y_te)
    out["L1"] = r2(si.dec.predict((sj.x_te - sj.z.mean) / si.z.std), sj.y_te)
    out["L2"] = r2(si.dec.predict(sj.zte), sj.y_te)
    # unsupervised Procrustes latent alignment (L4u)
    U_, _, Wt_ = np.linalg.svd(sj.V.T @ si.V)
    R = U_ @ Wt_
    G_te, B_te = latent_features(sj.zte, sj.V, si.V, si.dec)
    out["L4u"] = r2(pred_latent(G_te, B_te, (R.reshape(-1), si.dec.b)), sj.y_te)

    F_tr, F_te = gain_features(sj.ztr, si.dec), gain_features(sj.zte, si.dec)
    G_tr, B_tr = latent_features(sj.ztr, sj.V, si.V, si.dec)
    H_tr = lagged(sj.ztr, N_LAGS)
    H_te = lagged(sj.zte, N_LAGS)
    for n in n_list:
        lab, fit_m, val_m = holdout_split(sj.trial_id, n)
        Y = sj.y_tr
        sse = lambda p, m: float(((p - Y[m]) ** 2).sum())  # noqa: E731

        lam = pick_lam(lambda m, l: fit_gain(F_tr[m], Y[m], l), lambda prm, m: sse(pred_gain(F_tr[m], prm), m), fit_m, val_m)
        out[f"L3_n{n}"] = r2(pred_gain(F_te, fit_gain(F_tr[lab], Y[lab], lam)), sj.y_te)

        lam = pick_lam(lambda m, l: fit_latent(G_tr[m], B_tr[m], Y[m], R, l),
                       lambda prm, m: sse(pred_latent(G_tr[m], B_tr[m], prm), m), fit_m, val_m)
        out[f"L4_n{n}"] = r2(pred_latent(G_te, B_te, fit_latent(G_tr[lab], B_tr[lab], Y[lab], R, lam)), sj.y_te)

        lam = pick_lam(lambda m, l: fit_remap(sj.ztr[m], Y[m], si.dec, l),
                       lambda prm, m: sse(pred_remap(sj.ztr[m], prm), m), fit_m, val_m)
        out[f"L5_n{n}"] = r2(pred_remap(sj.zte, fit_remap(sj.ztr[lab], Y[lab], si.dec, lam)), sj.y_te)

        alphas = np.array([0.1, 1, 10, 100, 1e3, 1e4])
        a = alphas[np.argmin([sse(H_tr[val_m] @ W + b, val_m) for W, b in [ridge(H_tr[fit_m], Y[fit_m], a) for a in alphas]])]
        W, b = ridge(H_tr[lab], Y[lab], a)
        out[f"L6_n{n}"] = r2(H_te @ W + b, sj.y_te)
        a = alphas[np.argmin([sse(H_tr[val_m] @ W + b, val_m)
                              for W, b in [ridge_to_prior(H_tr[fit_m], Y[fit_m], si.dec.W, si.dec.b, a) for a in alphas]])]
        W, b = ridge_to_prior(H_tr[lab], Y[lab], si.dec.W, si.dec.b, a)
        out[f"L6p_n{n}"] = r2(H_te @ W + b, sj.y_te)
    out["own"] = r2(sj.dec.predict(sj.zte), sj.y_te)
    return out


