"""Unit tests for core numerical pieces. Run: python -m pytest tests -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ibci import sim  # noqa: E402
from ibci.linear import LagDecoder, lagged, ridge, ridge_to_prior  # noqa: E402
from ibci.metrics import r2_per_dof  # noqa: E402
from ibci.preprocess import add_history  # noqa: E402


def test_lagged_matches_add_history():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((50, 3)).astype(np.float32)
    a = lagged(Z, 4).reshape(50, 4, 3)          # current lag first
    b = add_history(Z, 4)                       # oldest lag first
    assert np.allclose(a, b[:, ::-1])
    assert np.allclose(a[0, 1:], 0)             # no data before t=0


def test_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 10))
    Y = X @ rng.standard_normal((10, 2)) + 0.1 * rng.standard_normal((200, 2)) + 3.0
    W, b = ridge(X, Y, 2.0)
    sk = Ridge(alpha=2.0).fit(X, Y)
    assert np.allclose(W, sk.coef_.T, atol=1e-6) and np.allclose(b, sk.intercept_, atol=1e-6)


def test_ridge_to_prior_limits():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 5))
    Y = rng.standard_normal((100, 1))
    W0 = rng.standard_normal((5, 1))
    W_big, _ = ridge_to_prior(X, Y, W0, np.zeros(1), 1e9)
    assert np.allclose(W_big, W0, atol=1e-5)             # infinite shrinkage -> prior
    W_small, _ = ridge_to_prior(X, Y, W0, np.zeros(1), 1e-9)
    assert np.allclose(W_small, ridge(X, Y, 1e-9)[0], atol=1e-4)


def test_decoder_recovers_linear_system():
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((3000, 8)).astype(np.float32)
    Wtrue = rng.standard_normal((2 * 8, 2)).astype(np.float32)
    Y = lagged(Z, 2) @ Wtrue
    dec = LagDecoder.fit(Z, Y, n_lags=2, alpha=1e-3)
    assert r2_per_dof(dec.predict(Z), Y).min() > 0.999


def test_sim_identity_at_zero_drift():
    rng = np.random.default_rng(4)
    Z = rng.standard_normal((500, 96)).astype(np.float32)
    dist = np.ones((96, 96))
    p = sim.SimParams(s_mix0=0, s_mix=0, rho0=0, rho_inf=0, h_off=0, h_jump=0)
    d = sim.sample_drift(Z, 30.0, p, dist, rng)
    X = sim.apply_drift(Z, d, rng)
    # rho is clipped to >= 1e-4, so the output is the input up to a negligible mixture
    assert np.corrcoef(X.ravel(), Z.ravel())[0, 1] > 0.999


def test_sim_more_drift_with_time():
    rng = np.random.default_rng(5)
    Z = rng.standard_normal((2000, 96)).astype(np.float32)
    dist = np.ones((96, 96))
    p = sim.SimParams(h_off=0, h_jump=0)
    sims = []
    for dt in (1, 30, 1000):
        r = np.random.default_rng(7)
        d = sim.sample_drift(Z, dt, p, dist, r)
        sims.append(np.mean([np.corrcoef(sim.apply_drift(Z, d, r)[:, c], Z[:, c])[0, 1] for c in range(96)]))
    assert sims[0] > sims[1] > sims[2]
