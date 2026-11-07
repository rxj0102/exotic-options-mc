"""Antithetic variates."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from simulation.base import MCResult, MonteCarloEngine

__all__ = ["AntitheticVariates"]


class AntitheticVariates:
    """Antithetic-variates estimator.

    For each draw ``Z`` the antithetic draw ``-Z`` is also used. When the
    payoff is monotone in the underlying the two are negatively correlated,
    reducing the variance of their average.
    """

    @staticmethod
    def efficiency_gain(crude_var: float, antithetic_var: float) -> float:
        """Return the variance-reduction ratio ``crude_var / antithetic_var``.

        Values greater than one indicate an improvement.
        """
        if antithetic_var <= 0:
            return float("inf")
        return float(crude_var / antithetic_var)

    def price(
        self,
        paths: np.ndarray,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        r: float,
        T: float,
    ) -> MCResult:
        """Price using paired antithetic paths.

        Parameters
        ----------
        paths : numpy.ndarray
            Array of shape ``(2 * n_pairs, n_steps + 1)`` whose first half
            and second half are antithetic partners (as produced by
            :meth:`GBMSimulator.generate_paths_antithetic`).
        payoff_fn : callable
            Maps paths to undiscounted payoffs.
        r, T : float
            Discount rate and maturity.

        Returns
        -------
        MCResult
            The ``extra`` payload contains the crude-MC variance and the
            variance-reduction ratio.
        """
        start = time.perf_counter()
        n_total = paths.shape[0]
        if n_total % 2 != 0:
            raise ValueError("Antithetic paths must come in pairs (even count).")
        n = n_total // 2
        disc = np.exp(-r * T)
        payoffs = np.asarray(payoff_fn(paths), dtype=float) * disc

        first, second = payoffs[:n], payoffs[n:]
        pair_means = 0.5 * (first + second)

        res = MonteCarloEngine.summarize(
            pair_means, n_paths=n_total, method="antithetic"
        )
        # Compare against the crude estimator using all 2n samples independently.
        crude_var = np.var(payoffs, ddof=1) / n_total
        anti_var = np.var(pair_means, ddof=1) / n
        res.extra = {
            "crude_variance": float(crude_var),
            "antithetic_variance": float(anti_var),
            "vr_ratio": self.efficiency_gain(crude_var, anti_var),
        }
        res.elapsed_seconds = time.perf_counter() - start
        return res
