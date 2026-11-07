"""Monte Carlo pricers are checked against closed forms and no-arbitrage identities."""

import numpy as np
import pytest

from benchmarks.analytical import bsm_price
from benchmarks.barrier_analytical import rubinstein_reiner_barrier as rr
from benchmarks.digital_analytical import cash_or_nothing
from pricing.asian import AsianOptionPricer
from pricing.barrier import BarrierOptionPricer
from pricing.digital import DigitalOptionPricer
from simulation.gbm import GBMSimulator
from variance_reduction.antithetic import AntitheticVariates

S0, K, T, r, q, sig = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25


def within(res, exact, z=4.0):
    return abs(res.price - exact) <= z * res.stderr + 1e-9


@pytest.fixture(scope="module")
def paths():
    return GBMSimulator(seed=7).generate_paths(S0, T, r, q, sig, n_paths=60_000, n_steps=252)


# ----------------------------------------------------------- simulator
def test_gbm_is_a_martingale_after_discounting(paths):
    fwd = S0 * np.exp((r - q) * T)
    st = paths[:, -1]
    assert abs(st.mean() - fwd) < 4 * st.std() / np.sqrt(len(st))
    assert np.allclose(paths[:, 0], S0)


def test_antithetic_pairs_are_mirrored_and_stderr_uses_pairs():
    sim = GBMSimulator(seed=1)
    p = sim.generate_paths_antithetic(S0, T, r, q, sig, n_paths=5000, n_steps=4)
    n = p.shape[0] // 2
    assert np.allclose(np.log(p[:n, 1:] / S0) + np.log(p[n:, 1:] / S0),
                       2 * (r - q - 0.5 * sig**2) * T / 4 * np.arange(1, 5))
    call = lambda x: np.maximum(x[:, -1] - K, 0)
    res = AntitheticVariates().price(p, call, r, T)
    assert within(res, bsm_price(S0, K, T, r, q, sig, "call"))
    assert res.extra["vr_ratio"] > 1.5  # monotone payoff -> real variance reduction


# ----------------------------------------------------------- barrier
@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize("side,H", [("down", 90.0), ("up", 115.0)])
def test_barrier_in_plus_out_equals_vanilla_analytical(opt, side, H):
    for strike in (95.0, 100.0, 105.0):
        i = rr(S0, strike, H, T, r, q, sig, f"{side}-and-in", opt)
        o = rr(S0, strike, H, T, r, q, sig, f"{side}-and-out", opt)
        assert i + o == pytest.approx(bsm_price(S0, strike, T, r, q, sig, opt), abs=1e-8)


@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize("btype,H", [
    ("down-and-out", 90.0), ("down-and-in", 90.0),
    ("up-and-out", 115.0), ("up-and-in", 115.0),
])
def test_barrier_mc_bridge_matches_closed_form(paths, opt, btype, H):
    pr = BarrierOptionPricer()
    res = pr.price_mc(paths, K, H, r, T, btype, opt, sigma=sig, brownian_bridge=True)
    exact = rr(S0, K, H, T, r, q, sig, btype, opt)
    assert within(res, exact, z=4.5), (btype, opt, res.price, exact)


def test_discrete_monitoring_overprices_knock_out(paths):
    pr = BarrierOptionPricer()
    disc = pr.price_mc(paths, K, 90.0, r, T, "down-and-out", "call").price
    cont = rr(S0, K, 90.0, T, r, q, sig, "down-and-out", "call")
    assert disc > cont  # discrete monitoring misses crossings between dates


# ------------------------------------------------------------- Asian
def test_geometric_asian_mc_matches_kemna_vorst(paths):
    pr = AsianOptionPricer("continuous")
    res = pr.price_geometric(paths, K, r, T, "call")
    assert within(res, pr.price_geometric_analytical(S0, K, T, r, q, sig, "call", n_obs=252))


@pytest.mark.parametrize("freq,n_obs", [("weekly", 52), ("monthly", 12)])
def test_averaging_dates_are_evenly_spaced(freq, n_obs):
    cols = AsianOptionPricer(freq)._averaging_columns(252)
    assert len(cols) == n_obs and cols[-1] == 252
    assert np.allclose(np.diff(cols), 252 / n_obs, atol=1.0)


def test_control_variate_uses_actual_number_of_dates():
    """Coarse grid (50 steps) with 'daily' frequency averages 50 dates, not 252."""
    sim = GBMSimulator(seed=3)
    p = sim.generate_paths(S0, T, r, q, sig, n_paths=40_000, n_steps=50)
    pr = AsianOptionPricer("daily")
    cv = pr.price_with_control_variate(p, K, r, T, q, sig, S0, "call")
    crude = pr.price_arithmetic(p, K, r, T, "call")
    assert abs(cv.price - crude.price) < 4 * crude.stderr  # unbiased vs crude
    assert cv.stderr < crude.stderr


def test_arithmetic_asian_above_geometric_and_below_vanilla(paths):
    pr = AsianOptionPricer("continuous")
    a = pr.price_arithmetic(paths, K, r, T, "call").price
    g = pr.price_geometric(paths, K, r, T, "call").price
    assert g <= a <= bsm_price(S0, K, T, r, q, sig, "call")


# ----------------------------------------------------------- digital
def test_cash_or_nothing_matches_closed_form(paths):
    res = DigitalOptionPricer().price_mc(paths, K, r, T, "cash_or_nothing", "call")
    assert within(res, cash_or_nothing(S0, K, T, r, q, sig, "call"))
