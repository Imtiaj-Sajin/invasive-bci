"""Feature-level simulator of chronic intracortical drift and electrode failure.

A simulated future day is produced by transforming a *real* base session (z-scored spike-band power Z, kinematics
Y), so the behaviour and latent dynamics stay realistic while the recording changes the way chronic arrays change:

  1. mixing      Z <- (I + E) Z,  E = s_mix(dt) * [ (1-beta) G_global/sqrt(C) + beta (G (.) K_local) / sqrt(n_nb) ]
                 (signals re-distribute across channels; K_local = grid neighbours; recoverable by an input remap)
  2. turnover    a fraction rho_c ~ Beta(mean=rho(dt)) of each channel's signal is replaced by a new unit whose tuning
                 is a random rotation (in the session's top-k latent space) of a randomly chosen channel's loadings,
                 plus private noise -> genuinely new tuning on that electrode
  3. failure     channels switch between alive and silent (two-state Markov process with rates h_off, h_on; most
                 channel losses in LINK are transient); a silent channel keeps a fraction silent_signal_frac of its
                 signal variance (sub-threshold multi-unit activity still shows in SBP) and the rest is noise. Plus abrupt
                 per-channel gain jumps (heavy-tailed) at rate h_jump
  4. raw-unit    per-channel offset and log-scale random walks (only matter for decoders that do not renormalize)

Each magnitude has an instant session-to-session component (present for any dt >= 1) plus a slow component that
grows with elapsed days through a saturating curve f(dt) = 1 - exp(-dt / tau). All parameters live in
``SimParams`` and are calibrated against real data (scripts/calibrate_sim.py), not chosen by hand.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SimParams:
    s_mix0: float = 0.2       # instant (session-to-session) mixing strength, present for any dt >= 1
    s_mix: float = 0.3        # slow, accumulating mixing strength (asymptote)
    tau_mix: float = 30.0     # days
    beta_local: float = 0.5   # share of mixing that is spatially local
    rho0: float = 0.05        # instant turnover fraction
    rho_inf: float = 0.3      # asymptotic turnover fraction per channel
    tau_rho: float = 200.0    # days
    rho_conc: float = 5.0     # Beta concentration of per-channel turnover
    h_off: float = 2e-3       # alive -> silent switching rate (per channel per day); fitted from activity data
    h_on: float = 2e-3        # silent -> alive switching rate (per channel per day)
    silent_signal_frac: float = 0.37  # share of signal variance a TC-silent channel keeps in SBP (LINK: median
                                      # within-channel ratio of SBP tuning R2, silent vs active = 0.366)
    h_jump: float = 5e-3      # abrupt gain-jump events per channel per day
    jump_sd: float = 0.5      # log-gain sd of a jump
    rw_offset: float = 0.02   # raw offset random walk sd per sqrt(day), in units of channel sd
    rw_logscale: float = 0.02 # raw log-scale random walk sd per sqrt(day)
    k_latent: int = 20

    def to_dict(self):
        return asdict(self)


def _sat(dt, tau):
    return 1.0 - np.exp(-dt / max(tau, 1e-6))


def p_silent(dt, h_off, h_on):
    """P(silent at dt | alive at 0) for a two-state continuous-time Markov chain."""
    tot = h_off + h_on
    return 0.0 if tot <= 0 else h_off / tot * (1.0 - np.exp(-tot * dt))


def local_kernel(dist: np.ndarray) -> np.ndarray:
    """Grid neighbours (Chebyshev distance 1 on the same array), zero diagonal."""
    return ((dist == 1)).astype(float)


@dataclass
class Drift:
    """One sampled realization of everything that changes between the base day and the simulated day."""
    A: np.ndarray            # (C, C) mixing matrix I + E
    rho: np.ndarray          # (C,) turnover fraction
    src: np.ndarray          # (C,) channel whose loadings seed each new unit
    Q: np.ndarray            # (k, k) rotation of new-unit tuning in latent space
    V: np.ndarray            # (C, k) base-session latent basis (from its training segment)
    mean: np.ndarray         # (C,) base-session mean used for latent projection
    resid_sd: np.ndarray     # (C,) private-noise sd of each channel in the base session
    gain: np.ndarray         # (C,) multiplicative gain (abrupt jumps)
    dead: np.ndarray         # (C,) bool
    offset: np.ndarray       # (C,) raw-unit offset (channel-sd units)
    logscale: np.ndarray     # (C,) raw-unit log-scale change
    silent_frac: float = 0.37  # signal share kept by silent channels


def sample_drift(Z_base: np.ndarray, dt: float, p: SimParams, dist: np.ndarray, rng: np.random.Generator) -> Drift:
    """Sample a drift realization for a simulated day ``dt`` days after the base session Z_base (T, C)."""
    T, C = Z_base.shape
    K = local_kernel(dist)
    n_nb = np.maximum(K.sum(1, keepdims=True), 1)
    Gg = rng.standard_normal((C, C)) / np.sqrt(C)
    np.fill_diagonal(Gg, 0)
    Gl = rng.standard_normal((C, C)) * K / np.sqrt(n_nb)
    # variances of the instant and slow components add
    s_tot = np.sqrt(p.s_mix0 ** 2 * (dt > 0) + (p.s_mix * _sat(dt, p.tau_mix)) ** 2)
    E = s_tot * ((1 - p.beta_local) * Gg + p.beta_local * Gl)

    k = min(p.k_latent, C - 1)
    mean = Z_base.mean(0)
    _, S, Vt = np.linalg.svd(Z_base - mean, full_matrices=False)
    V = Vt[:k].T
    var = (Z_base - mean).var(0)
    resid_sd = np.sqrt(np.maximum(var - (V ** 2 * (S[:k] ** 2 / T)).sum(1), 1e-3 * var))
    rho_mean = float(np.clip(p.rho0 * (dt > 0) + (p.rho_inf - p.rho0) * _sat(dt, p.tau_rho), 1e-4, 0.999))
    rho = rng.beta(rho_mean * p.rho_conc, (1 - rho_mean) * p.rho_conc, size=C)
    Q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    src = rng.integers(0, C, size=C)

    dead = rng.random(C) < p_silent(dt, p.h_off, p.h_on)
    n_jumps = rng.poisson(p.h_jump * dt, size=C)
    gain = np.exp(np.array([rng.normal(0, p.jump_sd, n).sum() for n in n_jumps]))
    offset = rng.normal(0, p.rw_offset * np.sqrt(dt), C)
    logscale = rng.normal(0, p.rw_logscale * np.sqrt(dt), C)
    return Drift(np.eye(C) + E, rho, src, Q, V, mean, resid_sd, gain, dead, offset, logscale, p.silent_signal_frac)


def apply_drift(Z: np.ndarray, d: Drift, rng: np.random.Generator) -> np.ndarray:
    """Apply a drift realization to any segment Z (T, C) of the base session (z-scored units)."""
    T, C = Z.shape
    X = Z @ d.A.T
    F = (Z - d.mean) @ d.V                                   # latent trajectories in the base basis
    new_unit = F @ (d.V[d.src] @ d.Q).T + rng.standard_normal((T, C)) * d.resid_sd[d.src]
    new_unit = (new_unit - new_unit.mean(0)) / (new_unit.std(0) + 1e-9)
    sd = X.std(0) + 1e-9
    X = np.sqrt(1 - d.rho) * X + np.sqrt(d.rho) * new_unit * sd
    X = X * d.gain
    if d.dead.any():
        k = d.silent_frac
        nd = int(d.dead.sum())
        X[:, d.dead] = np.sqrt(k) * X[:, d.dead] + np.sqrt(1 - k) * rng.standard_normal((T, nd)) * X[:, d.dead].std(0)
    return X.astype(np.float32)


def to_raw(X: np.ndarray, d: Drift, raw_mean: np.ndarray, raw_std: np.ndarray) -> np.ndarray:
    """Map simulated z-scored features back to raw units with the offset / scale random walks applied."""
    return ((X + d.offset) * (raw_std * np.exp(d.logscale)) + raw_mean).astype(np.float32)
