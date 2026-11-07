"""Tests for variance-reduction techniques."""

import numpy as np
from scipy.stats import qmc

from benchmarks.analytical import bsm_price
from simulation.gbm import GBMSimulator
from variance_reduction.antithetic import AntitheticVariates
from variance_reduction.control_variates import ControlVariates
from variance_reduction.importance_sampling import ImportanceSampling
from variance_reduction.quasi_mc import QuasiMonteCarlo


def test_antithetic_reduces_variance(market):
    g = GBMSimulator(seed=3)
    paths = g.generate_paths_antithetic(
        market["S0"],
        market["T"],
        market["r"],
        market["q"],
        market["sigma"],
        n_paths=50_000,
        n_steps=100,
    )
    payoff = lambda p: np.maximum(p[:, 1:].mean(axis=1) - 100.0, 0.0)
    res = AntitheticVariates().price(paths, payoff, market["r"], market["T"])
    assert res.extra["vr_ratio"] > 1.0


def test_control_variate_ratio(market, gbm_paths):
    S0, K, r, q, sigma, T = (
        market["S0"],
        100.0,
        market["r"],
        market["q"],
        market["sigma"],
        market["T"],
    )
    cv = ControlVariates().geometric_asian_control(
        gbm_paths, S0, K, T, r, q, sigma, "call"
    )
    assert cv.extra["vr_ratio"] >= 2.0


def test_sobol_lower_discrepancy(market):
    rng = np.random.default_rng(0)
    n, d = 1024, 4
    sob = qmc.Sobol(d=d, scramble=True, seed=rng).random(n)
    prng = rng.random((n, d))
    assert qmc.discrepancy(sob) < qmc.discrepancy(prng)


def test_importance_sampling_otm(market):
    S0, r, q, sigma, T = (
        market["S0"],
        market["r"],
        market["q"],
        market["sigma"],
        market["T"],
    )
    K = 1.3 * S0
    res = ImportanceSampling(seed=5).price_otm_call(S0, K, T, r, q, sigma, 100_000)
    analytic = bsm_price(S0, K, T, r, q, sigma, "call")
    # Accuracy and a substantial variance-reduction ratio.
    assert abs(res.price - analytic) < 3 * res.stderr
    assert res.extra["vr_ratio"] >= 5.0


def test_qmc_rqmc_beats_crude(market):
    S0, r, q, sigma, T = (
        market["S0"],
        market["r"],
        market["q"],
        market["sigma"],
        market["T"],
    )
    payoff = lambda p: np.maximum(p[:, 1:].mean(axis=1) - 100.0, 0.0)
    qmc_engine = QuasiMonteCarlo(seed=11)
    rqmc = qmc_engine.price_rqmc(payoff, S0, T, r, q, sigma, 4096, 64, n_replications=8)
    g = GBMSimulator(seed=11)
    crude_paths = g.generate_paths(S0, T, r, q, sigma, 4096 * 8, 64)
    crude_stderr = np.std(np.exp(-r * T) * payoff(crude_paths), ddof=1) / np.sqrt(
        4096 * 8
    )
    assert rqmc.stderr < crude_stderr
