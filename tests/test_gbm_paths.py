"""Tests for the GBM path generator."""

import numpy as np

from simulation.gbm import GBMSimulator


def test_paths_start_at_S0(market):
    g = GBMSimulator(seed=1)
    paths = g.generate_paths(market["S0"], 1.0, 0.05, 0.0, 0.2, n_paths=1000, n_steps=50)
    assert np.allclose(paths[:, 0], market["S0"])
    assert paths.shape == (1000, 51)


def test_expected_terminal_value(market, gbm_paths):
    S0, r, q, T = market["S0"], market["r"], market["q"], market["T"]
    S_T = gbm_paths[:, -1]
    theory = S0 * np.exp((r - q) * T)
    stderr = S_T.std(ddof=1) / np.sqrt(S_T.size)
    assert abs(S_T.mean() - theory) < 3 * stderr


def test_log_variance(market, gbm_paths):
    sigma, T = market["sigma"], market["T"]
    log_ret = np.log(gbm_paths[:, -1] / market["S0"])
    assert abs(log_ret.var(ddof=1) - sigma**2 * T) < 0.01


def test_antithetic_pairing(market):
    g = GBMSimulator(seed=7)
    paths = g.generate_paths_antithetic(market["S0"], 1.0, 0.05, 0.0, 0.2, n_paths=500, n_steps=20)
    assert paths.shape == (1000, 21)
    n = 500
    # Recover the implied normals; the two halves should be exact negatives.
    log_inc = np.diff(np.log(paths), axis=1)
    drift_vol = log_inc[:n] + log_inc[n:]
    # Z and -Z imply increments that sum to 2*drift (the random part cancels).
    assert np.allclose(drift_vol, drift_vol[0], atol=1e-9)


def test_reproducibility(market):
    a = GBMSimulator(seed=99).generate_paths(100, 1, 0.05, 0, 0.2, 100, 10)
    b = GBMSimulator(seed=99).generate_paths(100, 1, 0.05, 0, 0.2, 100, 10)
    assert np.array_equal(a, b)
