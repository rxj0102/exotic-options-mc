"""Tests for the Heston and Merton jump-diffusion simulators."""

import warnings

import numpy as np

from simulation.heston import HestonSimulator
from simulation.jump_diffusion import MertonJumpDiffusion


def test_heston_variance_non_negative():
    # Full-truncation scheme must keep variance non-negative on every path.
    h = HestonSimulator(seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S, V = h.generate_paths(
            100, 1.0, 0.05, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7, n_paths=20_000, n_steps=252
        )
    assert (V >= 0).all()
    assert S.shape == V.shape == (20_000, 253)


def test_heston_empirical_correlation():
    rho = -0.7
    h = HestonSimulator(seed=2)
    S, V = h.generate_paths(
        100, 1.0, 0.05, 0.0, 0.04, 2.0, 0.04, 0.3, rho, n_paths=40_000, n_steps=252
    )
    dlogS = np.diff(np.log(S), axis=1).ravel()
    dV = np.diff(V, axis=1).ravel()
    corr = np.corrcoef(dlogS, dV)[0, 1]
    assert abs(corr - rho) < 0.05


def test_heston_feller_warning():
    h = HestonSimulator(seed=3)
    # 2*kappa*theta = 2*1*0.04 = 0.08 < xi^2 = 0.25 -> violated -> warns.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h.generate_paths(
            100, 0.5, 0.05, 0.0, 0.04, 1.0, 0.04, 0.5, -0.5, n_paths=1000, n_steps=50
        )
    assert any(issubclass(x.category, RuntimeWarning) for x in w)


def test_heston_martingale():
    h = HestonSimulator(seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S, _ = h.generate_paths(
            100, 1.0, 0.05, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7, n_paths=50_000, n_steps=252
        )
    ST = S[:, -1]
    se = ST.std(ddof=1) / np.sqrt(ST.size)
    assert abs(ST.mean() - 100 * np.exp(0.05)) < 3 * se


def test_merton_reproducible_and_shaped():
    a = MertonJumpDiffusion(seed=9).generate_paths(
        100, 1, 0.05, 0, 0.2, 1.0, -0.1, 0.15, 500, 50
    )
    b = MertonJumpDiffusion(seed=9).generate_paths(
        100, 1, 0.05, 0, 0.2, 1.0, -0.1, 0.15, 500, 50
    )
    assert a.shape == (500, 51)
    assert np.array_equal(a, b)
    assert np.allclose(a[:, 0], 100.0)
