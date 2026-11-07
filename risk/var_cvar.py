"""Value-at-Risk and Conditional VaR for exotic positions."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from simulation.gbm import GBMSimulator

__all__ = ["ExoticVaR"]


class ExoticVaR:
    """Monte Carlo VaR / CVaR for a single exotic option position.

    The position value is simulated forward over a short risk horizon by
    evolving the spot under GBM and revaluing the contract; the loss
    distribution then yields VaR and Expected Shortfall.

    Parameters
    ----------
    value_fn : callable
        ``value_fn(S0) -> float`` returns the present value of the position
        for a given spot. Used to revalue after the horizon shock.
    seed : int, optional
        Seed for the spot-shock generator.
    """

    def __init__(
        self, value_fn: Callable[[float], float], seed: Optional[int] = None
    ) -> None:
        self.value_fn = value_fn
        self._rng = np.random.default_rng(seed)

    def _pnl(
        self, S0: float, sigma: float, r: float, horizon_days: int, n_paths: int
    ) -> np.ndarray:
        """Simulate the P&L distribution over the risk horizon."""
        dt = horizon_days / 252.0
        base_value = self.value_fn(S0)
        Z = self._rng.standard_normal(n_paths)
        S_h = S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        # Revalue position at each shocked spot.
        shocked = np.array([self.value_fn(s) for s in S_h])
        return shocked - base_value

    def var(
        self,
        S0: float,
        sigma: float,
        r: float = 0.0,
        confidence: float = 0.99,
        horizon_days: int = 1,
        n_paths: int = 20_000,
    ) -> float:
        """Value-at-Risk (a positive number denotes a loss).

        Returns
        -------
        float
            The loss not exceeded with probability ``confidence``.
        """
        pnl = self._pnl(S0, sigma, r, horizon_days, n_paths)
        return float(-np.quantile(pnl, 1.0 - confidence))

    def cvar(
        self,
        S0: float,
        sigma: float,
        r: float = 0.0,
        confidence: float = 0.99,
        horizon_days: int = 1,
        n_paths: int = 20_000,
    ) -> float:
        """Conditional VaR (Expected Shortfall) at ``confidence``.

        Returns
        -------
        float
            The mean loss in the worst ``1 - confidence`` tail.
        """
        pnl = self._pnl(S0, sigma, r, horizon_days, n_paths)
        threshold = np.quantile(pnl, 1.0 - confidence)
        tail = pnl[pnl <= threshold]
        return float(-tail.mean()) if tail.size else float(-threshold)

    def pnl_distribution_data(
        self,
        S0: float,
        sigma: float,
        r: float = 0.0,
        horizon_days: int = 1,
        n_paths: int = 20_000,
        n_bins: int = 60,
        confidence: float = 0.99,
    ) -> dict:
        """Histogram of the P&L distribution with VaR / CVaR markers.

        Returns
        -------
        dict
            Keys ``bin_centers``, ``counts``, ``var`` and ``cvar``.
        """
        pnl = self._pnl(S0, sigma, r, horizon_days, n_paths)
        counts, edges = np.histogram(pnl, bins=n_bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        threshold = np.quantile(pnl, 1.0 - confidence)
        tail = pnl[pnl <= threshold]
        return {
            "bin_centers": centers,
            "counts": counts,
            "var": float(-threshold),
            "cvar": float(-tail.mean()) if tail.size else float(-threshold),
        }

    @staticmethod
    def component_var(component_values: dict) -> dict:
        """Naive component VaR additive decomposition for a structured note.

        Parameters
        ----------
        component_values : dict
            Mapping from component name to its standalone VaR contribution.

        Returns
        -------
        dict
            The input plus a ``total`` key (simple sum, ignoring
            diversification).
        """
        out = dict(component_values)
        out["total"] = float(sum(component_values.values()))
        return out
