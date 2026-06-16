"""Shared pytest fixtures for common contracts and simulators."""

import warnings

import pytest

from simulation.gbm import GBMSimulator


@pytest.fixture(scope="session")
def market():
    """Standard market parameters used across tests."""
    return {"S0": 100.0, "r": 0.05, "q": 0.0, "sigma": 0.2, "T": 1.0}


@pytest.fixture(scope="session")
def gbm_paths(market):
    """A reusable, reproducible set of GBM paths (100k x 252)."""
    g = GBMSimulator(seed=12345)
    return g.generate_paths(
        market["S0"], market["T"], market["r"], market["q"], market["sigma"],
        n_paths=100_000, n_steps=252,
    )


@pytest.fixture(autouse=True)
def _silence_feller_warnings():
    """Silence the Heston Feller-condition warning during tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield
