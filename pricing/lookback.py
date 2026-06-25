"""Lookback option pricing (fixed and floating strike)."""

from __future__ import annotations

import time

import numpy as np
from scipy import stats

from simulation.base import MCResult, MonteCarloEngine

__all__ = ["LookbackOptionPricer"]

_NORM = stats.norm

# Broadie-Glasserman-Kou continuity-correction constant: -zeta(1/2)/sqrt(2*pi).
_BGK_BETA = 0.5826


class LookbackOptionPricer:
    """Monte Carlo pricer for lookback options with analytical benchmarks.

    Payoff conventions
    ------------------
    * Fixed-strike call: ``max(max(S) - K, 0)``
    * Fixed-strike put: ``max(K - min(S), 0)``
    * Floating-strike call: ``S_T - min(S)`` (always non-negative)
    * Floating-strike put: ``max(S) - S_T`` (always non-negative)

    Analytical floating-strike prices use the Goldman-Sosin-Gatto (1979)
    closed form for a newly issued option (running extremum equal to spot).
    """

    # ------------------------------------------------------------------
    # Monte Carlo pricing
    # ------------------------------------------------------------------
    @staticmethod
    def _continuity_factor(sigma: float, T: float, n_steps: int) -> float:
        """BGK multiplicative shift from a discrete extremum to a continuous one."""
        dt = T / n_steps
        return float(np.exp(_BGK_BETA * sigma * np.sqrt(dt)))

    def price_fixed_strike(
        self,
        paths: np.ndarray,
        K: float,
        r: float,
        T: float,
        option_type: str = "call",
        sigma: float = None,
        continuity_correction: bool = False,
    ) -> MCResult:
        """Price a fixed-strike lookback option by Monte Carlo.

        Parameters
        ----------
        sigma : float, optional
            Volatility; required when ``continuity_correction`` is ``True``.
        continuity_correction : bool, default False
            Apply the Broadie-Glasserman-Kou shift so the discretely sampled
            extremum estimates its continuous counterpart.
        """
        start = time.perf_counter()
        option_type = option_type.lower()
        running_max = paths.max(axis=1)
        running_min = paths.min(axis=1)
        if continuity_correction:
            if sigma is None:
                raise ValueError("continuity_correction=True requires sigma")
            factor = self._continuity_factor(sigma, T, paths.shape[1] - 1)
            running_max = running_max * factor
            running_min = running_min / factor
        if option_type == "call":
            payoff = np.maximum(running_max - K, 0.0)
        else:
            payoff = np.maximum(K - running_min, 0.0)
        disc = np.exp(-r * T) * payoff
        res = MonteCarloEngine.summarize(
            disc, n_paths=paths.shape[0], method="fixed_lookback"
        )
        res.elapsed_seconds = time.perf_counter() - start
        return res

    def price_floating_strike(
        self,
        paths: np.ndarray,
        r: float,
        T: float,
        option_type: str = "call",
        sigma: float = None,
        continuity_correction: bool = False,
    ) -> MCResult:
        """Price a floating-strike lookback option by Monte Carlo.

        Parameters
        ----------
        sigma : float, optional
            Volatility; required when ``continuity_correction`` is ``True``.
        continuity_correction : bool, default False
            Apply the Broadie-Glasserman-Kou shift to the running extremum.
        """
        start = time.perf_counter()
        option_type = option_type.lower()
        S_T = paths[:, -1]
        running_max = paths.max(axis=1)
        running_min = paths.min(axis=1)
        if continuity_correction:
            if sigma is None:
                raise ValueError("continuity_correction=True requires sigma")
            factor = self._continuity_factor(sigma, T, paths.shape[1] - 1)
            running_max = running_max * factor
            running_min = running_min / factor
        if option_type == "call":
            payoff = S_T - running_min
        else:
            payoff = running_max - S_T
        disc = np.exp(-r * T) * payoff
        res = MonteCarloEngine.summarize(
            disc, n_paths=paths.shape[0], method="floating_lookback"
        )
        res.elapsed_seconds = time.perf_counter() - start
        return res

    def price_partial(
        self,
        paths: np.ndarray,
        K: float,
        r: float,
        T: float,
        lookback_start: float,
        lookback_end: float,
        option_type: str = "call",
    ) -> MCResult:
        """Price a partial (windowed) fixed-strike lookback option.

        The running extremum is computed only over the time window
        ``[lookback_start, lookback_end]`` (expressed as fractions of ``T``).

        Parameters
        ----------
        lookback_start, lookback_end : float
            Window bounds in ``[0, 1]`` as fractions of maturity.
        """
        start = time.perf_counter()
        option_type = option_type.lower()
        if lookback_start > lookback_end:
            raise ValueError(
                f"lookback_start ({lookback_start}) must not exceed "
                f"lookback_end ({lookback_end})."
            )
        n_steps = paths.shape[1] - 1
        i0 = max(int(np.floor(lookback_start * n_steps)), 0)
        i1 = min(int(np.ceil(lookback_end * n_steps)), n_steps)
        window = paths[:, i0 : i1 + 1]
        if option_type == "call":
            payoff = np.maximum(window.max(axis=1) - K, 0.0)
        else:
            payoff = np.maximum(K - window.min(axis=1), 0.0)
        disc = np.exp(-r * T) * payoff
        res = MonteCarloEngine.summarize(
            disc, n_paths=paths.shape[0], method="partial_lookback"
        )
        res.elapsed_seconds = time.perf_counter() - start
        return res

    # ------------------------------------------------------------------
    # Analytical benchmarks (Goldman-Sosin-Gatto, newly issued)
    # ------------------------------------------------------------------
    @staticmethod
    def floating_call_analytical(
        S0: float, T: float, r: float, q: float, sigma: float
    ) -> float:
        """Goldman-Sosin-Gatto floating-strike lookback call (new option).

        Returns
        -------
        float
            Present value assuming the running minimum equals ``S0``.
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        vol = sigma * np.sqrt(T)
        b = r - q
        if abs(b) < 1e-8:
            # Avoid the r == q singularity with a small perturbation.
            b = 1e-8 if b >= 0 else -1e-8
        a1 = (b + 0.5 * sigma**2) * T / vol
        a2 = a1 - vol
        a3 = (-b + 0.5 * sigma**2) * T / vol  # = -a2 for a newly issued option
        eq, er = np.exp(-q * T), np.exp(-r * T)
        ratio = sigma**2 / (2 * b)
        return float(
            S0 * eq * _NORM.cdf(a1)
            - S0 * eq * ratio * _NORM.cdf(-a1)
            - S0 * er * (_NORM.cdf(a2) - ratio * _NORM.cdf(-a3))
        )

    @staticmethod
    def floating_put_analytical(
        S0: float, T: float, r: float, q: float, sigma: float
    ) -> float:
        """Goldman-Sosin-Gatto floating-strike lookback put (new option).

        Returns
        -------
        float
            Present value assuming the running maximum equals ``S0``.
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        vol = sigma * np.sqrt(T)
        b = r - q
        if abs(b) < 1e-8:
            b = 1e-8 if b >= 0 else -1e-8
        b1 = (-b + 0.5 * sigma**2) * T / vol
        b2 = b1 - vol
        b3 = (b - 0.5 * sigma**2) * T / vol  # -b3 = b1 for a newly issued option
        eq, er = np.exp(-q * T), np.exp(-r * T)
        ratio = sigma**2 / (2 * b)
        return float(
            S0 * er * (_NORM.cdf(b1) - ratio * _NORM.cdf(-b3))
            + S0 * eq * ratio * _NORM.cdf(-b2)
            - S0 * eq * _NORM.cdf(b2)
        )
