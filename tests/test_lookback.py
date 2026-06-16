"""Tests for lookback option pricing."""

import numpy as np

from benchmarks.analytical import bsm_price
from pricing.lookback import LookbackOptionPricer


def test_floating_exceeds_vanilla(market, gbm_paths):
    r, T, sigma, S0, q = market["r"], market["T"], market["sigma"], market["S0"], market["q"]
    pricer = LookbackOptionPricer()
    floating_call = pricer.price_floating_strike(gbm_paths, r, T, "call")
    vanilla = bsm_price(S0, S0, T, r, q, sigma, "call")
    # A floating lookback call dominates the corresponding vanilla.
    assert floating_call.price >= vanilla


def test_floating_call_matches_gsg(market, gbm_paths):
    S0, r, q, sigma, T = market["S0"], market["r"], market["q"], market["sigma"], market["T"]
    pricer = LookbackOptionPricer()
    mc = pricer.price_floating_strike(gbm_paths, r, T, "call", sigma=sigma, continuity_correction=True)
    analytic = pricer.floating_call_analytical(S0, T, r, q, sigma)
    assert abs(mc.price - analytic) < 3 * mc.stderr


def test_floating_put_matches_gsg(market, gbm_paths):
    S0, r, q, sigma, T = market["S0"], market["r"], market["q"], market["sigma"], market["T"]
    pricer = LookbackOptionPricer()
    mc = pricer.price_floating_strike(gbm_paths, r, T, "put", sigma=sigma, continuity_correction=True)
    analytic = pricer.floating_put_analytical(S0, T, r, q, sigma)
    assert abs(mc.price - analytic) < 3 * mc.stderr


def test_fixed_strike_intrinsic_bound(market, gbm_paths):
    r, T, S0 = market["r"], market["T"], market["S0"]
    K = 90.0
    pricer = LookbackOptionPricer()
    call = pricer.price_fixed_strike(gbm_paths, K, r, T, "call")
    # Fixed-strike lookback call dominates the (discounted) intrinsic value.
    assert call.price >= max(S0 - K, 0.0) * np.exp(-r * T) - 3 * call.stderr
