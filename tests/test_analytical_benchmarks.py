"""Tests for analytical benchmark formulas."""

import numpy as np

from benchmarks.analytical import bsm_price
from benchmarks.barrier_analytical import rubinstein_reiner_barrier
from benchmarks.digital_analytical import cash_or_nothing, no_touch, one_touch


def test_bsm_known_value():
    # Textbook value for S=K=100, r=5%, sigma=20%, T=1: call ~= 10.4506.
    price = bsm_price(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
    assert abs(price - 10.450583572185565) < 1e-8


def test_put_call_parity():
    c = bsm_price(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
    p = bsm_price(100, 100, 1.0, 0.05, 0.0, 0.2, "put")
    assert abs((c - p) - (100 - 100 * np.exp(-0.05))) < 1e-10


def test_rubinstein_reiner_parity():
    args = dict(S0=100, K=100, H=120, T=1.0, r=0.05, q=0.0, sigma=0.2)
    ui = rubinstein_reiner_barrier(**args, barrier_type="up-and-in", option_type="call")
    uo = rubinstein_reiner_barrier(**args, barrier_type="up-and-out", option_type="call")
    vanilla = bsm_price(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
    assert abs((ui + uo) - vanilla) < 1e-8


def test_digital_cash_or_nothing():
    from scipy.stats import norm

    S0, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.0, 0.2
    d2 = (np.log(S0 / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    expected = np.exp(-r * T) * norm.cdf(d2)
    got = cash_or_nothing(S0, K, T, r, q, sigma, "call")
    assert abs(got - expected) < 1e-8


def test_touch_no_touch_sum():
    # A one-touch plus a no-touch on the same barrier pays the discount factor.
    ot = one_touch(100, 120, 1.0, 0.05, 0.0, 0.2)
    nt = no_touch(100, 120, 1.0, 0.05, 0.0, 0.2)
    assert abs((ot + nt) - np.exp(-0.05)) < 1e-10
