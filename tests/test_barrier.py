"""Tests for barrier option pricing."""

import numpy as np

from benchmarks.analytical import bsm_price
from benchmarks.barrier_analytical import rubinstein_reiner_barrier
from pricing.barrier import BarrierOptionPricer


def test_in_out_parity(market, gbm_paths):
    K, r, T, H = 100.0, market["r"], market["T"], 120.0
    pricer = BarrierOptionPricer()
    ui = pricer.price_mc(gbm_paths, K, H, r, T, "up-and-in", "call")
    uo = pricer.price_mc(gbm_paths, K, H, r, T, "up-and-out", "call")
    vanilla = np.exp(-r * T) * np.maximum(gbm_paths[:, -1] - K, 0.0)
    # Knock-in + knock-out reconstruct the vanilla exactly path-by-path.
    assert abs((ui.price + uo.price) - vanilla.mean()) < 1e-9


def test_analytical_in_out_parity(market):
    S0, K, q, sigma, r, T, H = (
        market["S0"],
        100.0,
        market["q"],
        market["sigma"],
        market["r"],
        market["T"],
        120.0,
    )
    ui = rubinstein_reiner_barrier(S0, K, H, T, r, q, sigma, "up-and-in", "call")
    uo = rubinstein_reiner_barrier(S0, K, H, T, r, q, sigma, "up-and-out", "call")
    vanilla = bsm_price(S0, K, T, r, q, sigma, "call")
    assert abs((ui + uo) - vanilla) < 1e-8


def test_bridge_matches_rubinstein_reiner(market, gbm_paths):
    S0, K, q, sigma, r, T, H = (
        market["S0"],
        100.0,
        market["q"],
        market["sigma"],
        market["r"],
        market["T"],
        120.0,
    )
    pricer = BarrierOptionPricer()
    mc = pricer.price_mc(
        gbm_paths, K, H, r, T, "up-and-out", "call", sigma=sigma, brownian_bridge=True
    )
    analytic = pricer.price_analytical(S0, K, H, T, r, q, sigma, "up-and-out", "call")
    assert abs(mc.price - analytic) < 3 * mc.stderr + 0.05


def test_up_and_out_monotonic_in_barrier(market, gbm_paths):
    K, r, T = 100.0, market["r"], market["T"]
    pricer = BarrierOptionPricer()
    prices = [
        pricer.price_mc(gbm_paths, K, H, r, T, "up-and-out", "call").price
        for H in (110.0, 120.0, 140.0)
    ]
    # As the barrier moves further from spot, the up-and-out call is worth more.
    assert prices[0] < prices[1] < prices[2]


def test_bridge_improves_accuracy(market, gbm_paths):
    S0, K, q, sigma, r, T, H = (
        market["S0"],
        100.0,
        market["q"],
        market["sigma"],
        market["r"],
        market["T"],
        120.0,
    )
    pricer = BarrierOptionPricer()
    analytic = pricer.price_analytical(S0, K, H, T, r, q, sigma, "up-and-out", "call")
    discrete = pricer.price_mc(gbm_paths, K, H, r, T, "up-and-out", "call").price
    bridge = pricer.price_mc(
        gbm_paths, K, H, r, T, "up-and-out", "call", sigma=sigma, brownian_bridge=True
    ).price
    assert abs(bridge - analytic) < abs(discrete - analytic)
