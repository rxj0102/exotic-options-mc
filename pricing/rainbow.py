"""Multi-asset rainbow option pricing."""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from simulation.base import MCResult, MonteCarloEngine

__all__ = ["RainbowOptionPricer"]


class RainbowOptionPricer:
    """Monte Carlo pricer for multi-asset rainbow options.

    Asset paths are generated jointly under correlated GBM. Supported payoffs
    include best-of, worst-of and spread options on the terminal asset values.

    Parameters
    ----------
    seed : int, optional
        Seed for the internal generator.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def simulate_terminal(
        self,
        S0: Sequence[float],
        T: float,
        r: float,
        q: Sequence[float],
        sigma: Sequence[float],
        corr: np.ndarray,
        n_paths: int = 100_000,
    ) -> np.ndarray:
        """Simulate correlated terminal asset prices.

        Parameters
        ----------
        S0, q, sigma : sequence of float
            Per-asset spot, dividend yield and volatility.
        corr : numpy.ndarray
            Correlation matrix of shape ``(n_assets, n_assets)``.
        n_paths : int
            Number of joint simulations.

        Returns
        -------
        numpy.ndarray
            Terminal prices of shape ``(n_paths, n_assets)``.
        """
        S0 = np.asarray(S0, dtype=float)
        q = np.asarray(q, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        corr = np.asarray(corr, dtype=float)
        n_assets = S0.size

        chol = np.linalg.cholesky(corr)
        Z = self._rng.standard_normal(size=(n_paths, n_assets)) @ chol.T
        drift = (r - q - 0.5 * sigma**2) * T
        diffusion = sigma * np.sqrt(T) * Z
        return S0 * np.exp(drift + diffusion)

    def price(
        self,
        S0: Sequence[float],
        K: float,
        T: float,
        r: float,
        q: Sequence[float],
        sigma: Sequence[float],
        corr: np.ndarray,
        payoff: str = "best_of_call",
        n_paths: int = 100_000,
    ) -> MCResult:
        """Price a rainbow option by Monte Carlo.

        Parameters
        ----------
        payoff : str
            One of ``"best_of_call"``, ``"worst_of_call"``, ``"best_of_put"``,
            ``"worst_of_put"`` or ``"spread"`` (``S1 - S2`` call struck at ``K``).

        Returns
        -------
        MCResult
        """
        start = time.perf_counter()
        ST = self.simulate_terminal(S0, T, r, q, sigma, corr, n_paths)
        payoff = payoff.lower()

        if payoff == "best_of_call":
            pay = np.maximum(ST.max(axis=1) - K, 0.0)
        elif payoff == "worst_of_call":
            pay = np.maximum(ST.min(axis=1) - K, 0.0)
        elif payoff == "best_of_put":
            pay = np.maximum(K - ST.max(axis=1), 0.0)
        elif payoff == "worst_of_put":
            pay = np.maximum(K - ST.min(axis=1), 0.0)
        elif payoff == "spread":
            pay = np.maximum((ST[:, 0] - ST[:, 1]) - K, 0.0)
        else:
            raise ValueError(f"Unknown rainbow payoff {payoff!r}")

        disc = np.exp(-r * T) * pay
        res = MonteCarloEngine.summarize(
            disc, n_paths=n_paths, method=f"rainbow_{payoff}"
        )
        res.elapsed_seconds = time.perf_counter() - start
        return res
