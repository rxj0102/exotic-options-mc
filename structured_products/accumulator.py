"""FX accumulator pricer."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from simulation.base import MCResult, MonteCarloEngine

__all__ = ["FXAccumulatorPricer"]


class FXAccumulatorPricer:
    """Monte Carlo pricer for an FX accumulator ("I-kill-you-later").

    Each day the spot trades below the upper knock-out barrier ``H`` the
    client accumulates one unit of FX at the fixed strike ``K``; if the spot
    is below the lower barrier ``L`` the daily accumulation is leveraged.
    Once the spot reaches ``H`` the structure knocks out and accumulation
    ceases.

    Parameters
    ----------
    seed : int, optional
        Seed for the internal generator.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def price(
        self,
        S0: float,
        K: float,
        H: float,
        L: float,
        T: float,
        r_d: float,
        r_f: float,
        sigma: float,
        n_paths: int = 100_000,
        leverage: float = 2.0,
        n_steps: int = 252,
    ) -> MCResult:
        """Price an FX accumulator from the client's perspective.

        Parameters
        ----------
        H, L : float
            Upper knock-out and lower leverage barriers.
        r_d, r_f : float
            Domestic and foreign interest rates (the FX drift is ``r_d - r_f``).
        leverage : float, default 2.0
            Multiplier on the daily accumulation while below ``L``.

        Returns
        -------
        MCResult
            ``extra`` reports the knock-out probability and a payoff sample.
        """
        start = time.perf_counter()
        dt = T / n_steps
        drift = (r_d - r_f - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        Z = self._rng.standard_normal((n_paths, n_steps))
        log_paths = np.cumsum(drift + vol * Z, axis=1)
        S = S0 * np.exp(log_paths)  # daily fixings (excludes S0)

        # Knock-out: once spot >= H the accumulation halts for that path.
        knocked = np.cumsum(S >= H, axis=1) > 0
        active = ~knocked

        # Daily settlement: client buys at K, marks at S; leveraged below L.
        units = np.where(S < L, leverage, 1.0)
        daily_pnl = units * (S - K) * active
        # Discount each daily settlement back to today.
        disc = np.exp(-r_d * dt * np.arange(1, n_steps + 1))
        path_pv = (daily_pnl * disc).sum(axis=1)

        res = MonteCarloEngine.summarize(path_pv, n_paths=n_paths, method="fx_accumulator")
        res.extra = {
            "knockout_probability": float((knocked[:, -1]).mean()),
            "payoff_samples": path_pv,
        }
        res.elapsed_seconds = time.perf_counter() - start
        return res

    def payoff_distribution(self, result: MCResult, n_bins: int = 60) -> dict:
        """Histogram of the accumulator P&L distribution.

        Parameters
        ----------
        result : MCResult
            A result returned by :meth:`price` (must carry ``payoff_samples``).

        Returns
        -------
        dict
            Keys ``bin_centers`` and ``counts``.
        """
        samples = result.extra["payoff_samples"]
        counts, edges = np.histogram(samples, bins=n_bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return {"bin_centers": centers, "counts": counts}
