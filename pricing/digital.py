"""Digital (binary) option pricing."""

from __future__ import annotations

import time

import numpy as np

from benchmarks.digital_analytical import (asset_or_nothing, cash_or_nothing,
                                           no_touch, one_touch)
from simulation.base import MCResult, MonteCarloEngine

__all__ = ["DigitalOptionPricer"]


class DigitalOptionPricer:
    """Monte Carlo and analytical pricer for digital options.

    Supported subtypes: cash-or-nothing, asset-or-nothing, one-touch and
    no-touch. A smoothed indicator payoff is available to reduce the
    discretisation bias of the cash-or-nothing Monte Carlo estimate near the
    strike.
    """

    def price_mc(
        self,
        paths: np.ndarray,
        K: float,
        r: float,
        T: float,
        subtype: str = "cash_or_nothing",
        option_type: str = "call",
        cash: float = 1.0,
        H: float = None,
        epsilon: float = None,
    ) -> MCResult:
        """Price a digital option by Monte Carlo.

        Parameters
        ----------
        subtype : {"cash_or_nothing", "asset_or_nothing", "one_touch", "no_touch"}
        option_type : {"call", "put"}
            Direction for the European digitals (ignored for touch options).
        cash : float, default 1.0
            Payout of cash-settled digitals.
        H : float, optional
            Barrier for one-touch / no-touch options.
        epsilon : float, optional
            Half-width of the smoothing band for the cash-or-nothing payoff.
            When ``None`` a hard indicator is used.

        Returns
        -------
        MCResult
        """
        start = time.perf_counter()
        subtype = subtype.lower()
        option_type = option_type.lower()
        S_T = paths[:, -1]
        disc = np.exp(-r * T)

        if subtype == "cash_or_nothing":
            if epsilon is not None and epsilon > 0:
                payoff = cash * self._smoothed_indicator(S_T, K, epsilon, option_type)
            else:
                ind = (S_T > K) if option_type == "call" else (S_T < K)
                payoff = cash * ind.astype(float)
        elif subtype == "asset_or_nothing":
            ind = (S_T > K) if option_type == "call" else (S_T < K)
            payoff = S_T * ind.astype(float)
        elif subtype in ("one_touch", "no_touch"):
            if H is None:
                raise ValueError("Touch options require a barrier H")
            touched = self._touched(paths, H)
            ind = touched if subtype == "one_touch" else ~touched
            payoff = cash * ind.astype(float)
        else:
            raise ValueError(f"Unknown digital subtype {subtype!r}")

        res = MonteCarloEngine.summarize(
            disc * payoff, n_paths=paths.shape[0], method=f"digital_{subtype}"
        )
        res.elapsed_seconds = time.perf_counter() - start
        return res

    @staticmethod
    def _touched(paths: np.ndarray, H: float) -> np.ndarray:
        """Return a boolean mask of paths that touch barrier ``H``."""
        if H >= paths[0, 0]:
            return paths.max(axis=1) >= H
        return paths.min(axis=1) <= H

    @staticmethod
    def _smoothed_indicator(
        S_T: np.ndarray, K: float, epsilon: float, option_type: str
    ) -> np.ndarray:
        """Smoothed call-spread approximation of a digital indicator.

        Approximates ``1{S_T > K}`` by the slope of a tight call spread:
        ``(max(S_T-(K-eps),0) - max(S_T-(K+eps),0)) / (2 eps)``.
        """
        call_ind = (
            np.maximum(S_T - (K - epsilon), 0.0) - np.maximum(S_T - (K + epsilon), 0.0)
        ) / (2 * epsilon)
        if option_type == "call":
            return call_ind
        return 1.0 - call_ind

    # ------------------------------------------------------------------
    # Analytical
    # ------------------------------------------------------------------
    def price_analytical(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        subtype: str = "cash_or_nothing",
        option_type: str = "call",
        cash: float = 1.0,
        H: float = None,
    ) -> float:
        """Closed-form digital option price.

        Returns
        -------
        float
        """
        subtype = subtype.lower()
        if subtype == "cash_or_nothing":
            return cash_or_nothing(S0, K, T, r, q, sigma, option_type, cash)
        if subtype == "asset_or_nothing":
            return asset_or_nothing(S0, K, T, r, q, sigma, option_type)
        if subtype == "one_touch":
            if H is None:
                raise ValueError("one_touch requires barrier H")
            return one_touch(S0, H, T, r, q, sigma, cash)
        if subtype == "no_touch":
            if H is None:
                raise ValueError("no_touch requires barrier H")
            return no_touch(S0, H, T, r, q, sigma, cash)
        raise ValueError(f"Unknown digital subtype {subtype!r}")
