"""Closed-form prices for digital (binary) options."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .analytical import bsm_d1_d2

__all__ = ["cash_or_nothing", "asset_or_nothing", "one_touch", "no_touch"]

_NORM = stats.norm


def cash_or_nothing(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
    cash: float = 1.0,
) -> float:
    """Price of a cash-or-nothing digital option.

    Pays ``cash`` if the option finishes in the money.

    Returns
    -------
    float
        ``cash * exp(-r T) * N(d2)`` for a call, ``cash * exp(-r T) * N(-d2)``
        for a put.
    """
    option_type = option_type.lower()
    _, d2 = bsm_d1_d2(S0, K, T, r, q, sigma)
    disc = np.exp(-r * T)
    if option_type == "call":
        return float(cash * disc * _NORM.cdf(d2))
    if option_type == "put":
        return float(cash * disc * _NORM.cdf(-d2))
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def asset_or_nothing(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Price of an asset-or-nothing digital option.

    Pays the asset value ``S_T`` if the option finishes in the money.

    Returns
    -------
    float
        ``S0 exp(-q T) N(d1)`` for a call, ``S0 exp(-q T) N(-d1)`` for a put.
    """
    option_type = option_type.lower()
    d1, _ = bsm_d1_d2(S0, K, T, r, q, sigma)
    disc_q = np.exp(-q * T)
    if option_type == "call":
        return float(S0 * disc_q * _NORM.cdf(d1))
    if option_type == "put":
        return float(S0 * disc_q * _NORM.cdf(-d1))
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def one_touch(
    S0: float,
    H: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    cash: float = 1.0,
) -> float:
    """Price of a one-touch option paying ``cash`` at maturity if ``H`` is hit.

    Uses the reflection-principle formula for the probability that a GBM
    touches the barrier ``H`` before ``T`` (payment at maturity).

    Parameters
    ----------
    H : float
        Barrier level (may be above or below the spot).

    Returns
    -------
    float
        Present value of the one-touch payout.
    """
    if T <= 0 or sigma <= 0:
        hit = (H >= S0) if H >= S0 else (H <= S0)
        return float(cash * np.exp(-r * T)) if hit else 0.0

    mu = r - q - 0.5 * sigma**2
    vol = sigma * np.sqrt(T)
    a = mu / sigma**2
    logHS = np.log(H / S0)
    # P(max/min touches H) under the physical-style measure with drift mu.
    if H >= S0:
        d1 = (logHS - mu * T) / vol
        d2 = (-logHS - mu * T) / vol
        prob = _NORM.cdf(-d1) + (H / S0) ** (2 * a) * _NORM.cdf(d2)
    else:
        d1 = (logHS - mu * T) / vol
        d2 = (-logHS - mu * T) / vol
        prob = _NORM.cdf(d1) + (H / S0) ** (2 * a) * _NORM.cdf(-d2)
    return float(cash * np.exp(-r * T) * np.clip(prob, 0.0, 1.0))


def no_touch(
    S0: float,
    H: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    cash: float = 1.0,
) -> float:
    """Price of a no-touch option paying ``cash`` if ``H`` is never hit.

    Returns
    -------
    float
        ``cash exp(-r T) - one_touch`` by static replication.
    """
    return float(cash * np.exp(-r * T) - one_touch(S0, H, T, r, q, sigma, cash))
