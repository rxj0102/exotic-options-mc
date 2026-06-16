"""Closed-form analytical prices used for validation and control variates."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats

__all__ = [
    "bsm_price",
    "bsm_d1_d2",
    "bsm_greeks",
]

_NORM = stats.norm


def bsm_d1_d2(
    S0: float, K: float, T: float, r: float, q: float, sigma: float
) -> Tuple[float, float]:
    """Return the Black-Scholes-Merton ``d1`` and ``d2`` terms.

    Parameters
    ----------
    S0, K : float
        Spot and strike.
    T : float
        Time to maturity in years.
    r, q : float
        Risk-free rate and dividend yield.
    sigma : float
        Volatility.

    Returns
    -------
    (float, float)
        The ``(d1, d2)`` pair. As ``T -> 0`` or ``sigma -> 0`` the terms
        tend to ``+/- inf`` according to moneyness.
    """
    if T <= 0 or sigma <= 0:
        # Degenerate limit: forward intrinsic decides the sign.
        fwd = S0 * np.exp((r - q) * T)
        sign = np.inf if fwd > K else (-np.inf if fwd < K else 0.0)
        return sign, sign
    vol = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol
    d2 = d1 - vol
    return d1, d2


def bsm_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes-Merton price of a European vanilla option.

    Parameters
    ----------
    option_type : {"call", "put"}
        Option flavour.

    Returns
    -------
    float
        Present value of the option.
    """
    option_type = option_type.lower()
    if T <= 0 or sigma <= 0:
        fwd = S0 * np.exp(-q * T)
        disc_k = K * np.exp(-r * T)
        if option_type == "call":
            return max(fwd - disc_k, 0.0)
        return max(disc_k - fwd, 0.0)

    d1, d2 = bsm_d1_d2(S0, K, T, r, q, sigma)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    if option_type == "call":
        return float(S0 * disc_q * _NORM.cdf(d1) - K * disc_r * _NORM.cdf(d2))
    if option_type == "put":
        return float(K * disc_r * _NORM.cdf(-d2) - S0 * disc_q * _NORM.cdf(-d1))
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bsm_greeks(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    """Return the analytical Greeks of a European vanilla option.

    Returns
    -------
    dict
        Keys ``delta``, ``gamma``, ``vega``, ``theta`` and ``rho``. Vega and
        rho are quoted per unit (i.e. per 100% move), not per percentage
        point.
    """
    option_type = option_type.lower()
    d1, d2 = bsm_d1_d2(S0, K, T, r, q, sigma)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    pdf = _NORM.pdf(d1)
    sqrtT = np.sqrt(T)

    gamma = disc_q * pdf / (S0 * sigma * sqrtT) if (S0 > 0 and sigma > 0 and T > 0) else 0.0
    vega = S0 * disc_q * pdf * sqrtT
    if option_type == "call":
        delta = disc_q * _NORM.cdf(d1)
        theta = (
            -S0 * disc_q * pdf * sigma / (2 * sqrtT)
            - r * K * disc_r * _NORM.cdf(d2)
            + q * S0 * disc_q * _NORM.cdf(d1)
        )
        rho = K * T * disc_r * _NORM.cdf(d2)
    else:
        delta = -disc_q * _NORM.cdf(-d1)
        theta = (
            -S0 * disc_q * pdf * sigma / (2 * sqrtT)
            + r * K * disc_r * _NORM.cdf(-d2)
            - q * S0 * disc_q * _NORM.cdf(-d1)
        )
        rho = -K * T * disc_r * _NORM.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }
