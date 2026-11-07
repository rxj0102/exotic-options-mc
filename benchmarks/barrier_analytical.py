"""Closed-form barrier option prices (Reiner-Rubinstein, 1991).

The implementation follows the standard ``A``-``F`` building-block
decomposition (see Haug, *The Complete Guide to Option Pricing Formulas*).
All eight single-barrier types are supported, with an optional rebate.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["rubinstein_reiner_barrier"]

_NORM = stats.norm

_VALID = {
    "up-and-in",
    "up-and-out",
    "down-and-in",
    "down-and-out",
}


def rubinstein_reiner_barrier(
    S0: float,
    K: float,
    H: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    barrier_type: str,
    option_type: str = "call",
    rebate: float = 0.0,
) -> float:
    """Closed-form price of a single-barrier option.

    Parameters
    ----------
    S0, K : float
        Spot and strike.
    H : float
        Barrier level.
    T : float
        Maturity in years.
    r, q : float
        Risk-free rate and dividend yield (cost of carry ``b = r - q``).
    sigma : float
        Volatility.
    barrier_type : {"up-and-in", "up-and-out", "down-and-in", "down-and-out"}
        Barrier style.
    option_type : {"call", "put"}
        Option flavour.
    rebate : float, default 0.0
        Cash rebate. For knock-out options it is paid at expiry if the option
        is knocked out; for knock-in options it is paid at expiry if the
        option never knocks in.

    Returns
    -------
    float
        Present value of the barrier option.
    """
    barrier_type = barrier_type.lower()
    option_type = option_type.lower()
    if barrier_type not in _VALID:
        raise ValueError(f"barrier_type must be one of {_VALID}, got {barrier_type!r}")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    # Degenerate / already-breached handling.
    if T <= 0 or sigma <= 0:
        from .analytical import bsm_price

        return bsm_price(S0, K, T, r, q, sigma, option_type)

    b = r - q
    vol = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    lam = np.sqrt(mu**2 + 2 * r / sigma**2)

    phi = 1.0 if option_type == "call" else -1.0
    eta = 1.0 if "down" in barrier_type else -1.0

    HS = H / S0
    x1 = np.log(S0 / K) / vol + (1 + mu) * vol
    x2 = np.log(S0 / H) / vol + (1 + mu) * vol
    y1 = np.log(H**2 / (S0 * K)) / vol + (1 + mu) * vol
    y2 = np.log(H / S0) / vol + (1 + mu) * vol
    z = np.log(HS) / vol + lam * vol

    eq = np.exp((b - r) * T)
    er = np.exp(-r * T)

    A = phi * S0 * eq * _NORM.cdf(phi * x1) - phi * K * er * _NORM.cdf(
        phi * x1 - phi * vol
    )
    B = phi * S0 * eq * _NORM.cdf(phi * x2) - phi * K * er * _NORM.cdf(
        phi * x2 - phi * vol
    )
    C = phi * S0 * eq * HS ** (2 * (mu + 1)) * _NORM.cdf(
        eta * y1
    ) - phi * K * er * HS ** (2 * mu) * _NORM.cdf(eta * y1 - eta * vol)
    D = phi * S0 * eq * HS ** (2 * (mu + 1)) * _NORM.cdf(
        eta * y2
    ) - phi * K * er * HS ** (2 * mu) * _NORM.cdf(eta * y2 - eta * vol)
    # Knock-out rebate (paid at expiry).
    E = (
        rebate
        * er
        * (
            _NORM.cdf(eta * x2 - eta * vol)
            - HS ** (2 * mu) * _NORM.cdf(eta * y2 - eta * vol)
        )
    )
    # Knock-in rebate (paid at expiry if never knocked in).
    F = rebate * (
        HS ** (mu + lam) * _NORM.cdf(eta * z)
        + HS ** (mu - lam) * _NORM.cdf(eta * z - 2 * eta * lam * vol)
    )

    in_the_strike_above = K > H
    price = _assemble(barrier_type, option_type, in_the_strike_above, A, B, C, D, E, F)
    return float(max(price, 0.0))


def _assemble(
    barrier_type: str,
    option_type: str,
    strike_above_barrier: bool,
    A: float,
    B: float,
    C: float,
    D: float,
    E: float,
    F: float,
) -> float:
    """Combine the building blocks per the Reiner-Rubinstein case table."""
    ka = strike_above_barrier  # K > H

    if barrier_type == "down-and-in" and option_type == "call":
        return C + E if ka else A - B + D + E
    if barrier_type == "up-and-in" and option_type == "call":
        return A + E if ka else B - C + D + E
    if barrier_type == "down-and-in" and option_type == "put":
        return B - C + D + E if ka else A + E
    if barrier_type == "up-and-in" and option_type == "put":
        return A - B + D + E if ka else C + E

    if barrier_type == "down-and-out" and option_type == "call":
        return A - C + F if ka else B - D + F
    if barrier_type == "up-and-out" and option_type == "call":
        return F if ka else A - B + C - D + F
    if barrier_type == "down-and-out" and option_type == "put":
        return A - B + C - D + F if ka else F
    if barrier_type == "up-and-out" and option_type == "put":
        return B - D + F if ka else A - C + F

    raise ValueError(f"Unhandled combination: {barrier_type} {option_type}")
