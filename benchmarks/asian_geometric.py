"""Closed-form geometric Asian option price (Kemna-Vorst, 1990)."""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["kemna_vorst_geometric_asian"]

_NORM = stats.norm


def kemna_vorst_geometric_asian(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_obs: int = 252,
    option_type: str = "call",
) -> float:
    """Closed-form price of a fixed-strike geometric-average Asian option.

    The geometric average of a set of jointly log-normal prices is itself
    log-normal, so the option admits a Black-style formula. For discrete
    monitoring over ``n`` equally spaced dates :math:`t_i = iT/n` the log
    geometric average :math:`\\log G` is normal with

    .. math::

        \\mathbb{E}[\\log(G/S_0)] &= (b - \\tfrac12 \\sigma^2)
            \\frac{T(n+1)}{2n}, \\\\
        \\operatorname{Var}[\\log(G/S_0)] &=
            \\sigma^2 T \\frac{(n+1)(2n+1)}{6 n^2},

    where :math:`b = r - q`. As :math:`n \\to \\infty` this reduces to the
    continuous Kemna-Vorst result with :math:`\\sigma_g = \\sigma/\\sqrt 3`.

    Parameters
    ----------
    S0, K : float
        Spot and strike.
    T : float
        Maturity in years.
    r, q : float
        Risk-free rate and dividend yield.
    sigma : float
        Volatility of the underlying.
    n_obs : int, default 252
        Number of equally spaced averaging dates.
    option_type : {"call", "put"}
        Option flavour.

    Returns
    -------
    float
        Present value of the geometric Asian option.
    """
    option_type = option_type.lower()
    if T <= 0 or sigma <= 0:
        fwd = S0 * np.exp((r - q) * T)
        disc = np.exp(-r * T)
        payoff = max(fwd - K, 0.0) if option_type == "call" else max(K - fwd, 0.0)
        return float(disc * payoff)

    n = int(n_obs)
    b = r - q
    # Log-mean and log-variance of the geometric average.
    M = (b - 0.5 * sigma**2) * T * (n + 1) / (2.0 * n)
    var_g = sigma**2 * T * (n + 1) * (2 * n + 1) / (6.0 * n**2)
    Sigma = np.sqrt(var_g)

    # Forward (risk-neutral expectation) of the geometric average.
    F = S0 * np.exp(M + 0.5 * var_g)
    disc_r = np.exp(-r * T)

    d1 = (np.log(F / K) + 0.5 * var_g) / Sigma
    d2 = d1 - Sigma

    if option_type == "call":
        return float(disc_r * (F * _NORM.cdf(d1) - K * _NORM.cdf(d2)))
    if option_type == "put":
        return float(disc_r * (K * _NORM.cdf(-d2) - F * _NORM.cdf(-d1)))
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
