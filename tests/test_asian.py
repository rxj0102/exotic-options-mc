"""Tests for Asian option pricing."""

import numpy as np

from pricing.asian import AsianOptionPricer


def test_geometric_matches_kemna_vorst(market, gbm_paths):
    S0, K, r, q, sigma, T = (
        market["S0"], 100.0, market["r"], market["q"], market["sigma"], market["T"]
    )
    pricer = AsianOptionPricer("continuous")
    mc = pricer.price_geometric(gbm_paths, K, r, T, "call")
    analytic = pricer.price_geometric_analytical(S0, K, T, r, q, sigma, "call", n_obs=252)
    assert abs(mc.price - analytic) < 3 * mc.stderr


def test_arithmetic_exceeds_geometric(market, gbm_paths):
    pricer = AsianOptionPricer("continuous")
    arith = pricer.price_arithmetic(gbm_paths, 100.0, market["r"], market["T"], "call")
    geo = pricer.price_geometric(gbm_paths, 100.0, market["r"], market["T"], "call")
    # Jensen's inequality: arithmetic average >= geometric average.
    assert arith.price > geo.price


def test_control_variate_reduces_variance(market, gbm_paths):
    S0, K, r, q, sigma, T = (
        market["S0"], 100.0, market["r"], market["q"], market["sigma"], market["T"]
    )
    pricer = AsianOptionPricer("continuous")
    crude = pricer.price_arithmetic(gbm_paths, K, r, T, "call")
    cv = pricer.price_with_control_variate(gbm_paths, K, r, T, q, sigma, S0, "call")
    assert cv.extra["vr_ratio"] >= 2.0
    # The CV estimate must materially tighten the standard error.
    assert cv.stderr < crude.stderr


def test_floating_vs_fixed(market, gbm_paths):
    pricer = AsianOptionPricer("continuous")
    floating = pricer.price_arithmetic(gbm_paths, 100.0, market["r"], market["T"], "call", floating=True)
    fixed = pricer.price_arithmetic(gbm_paths, 100.0, market["r"], market["T"], "call", floating=False)
    # Both should be non-negative, finite prices.
    assert floating.price >= 0 and fixed.price >= 0
