"""Shark-fin structured note pricer."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from pricing.barrier import BarrierOptionPricer
from simulation.base import MCResult, MonteCarloEngine
from simulation.gbm import GBMSimulator

__all__ = ["SharkFinPricer"]


class SharkFinPricer:
    """Monte Carlo pricer for a shark-fin note.

    The investor participates in the upside of a call as long as an upper
    knock-out barrier is never breached; if it is, a fixed rebate is paid
    instead. The economics are those of an up-and-out call plus a rebate
    contingent on the knock-out.

    Parameters
    ----------
    seed : int, optional
        Seed for the internal GBM simulator.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._gbm = GBMSimulator(seed=seed)
        self._barrier = BarrierOptionPricer()

    def price(
        self,
        S0: float,
        K: float,
        barrier: float,
        rebate: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        n_paths: int = 100_000,
        n_steps: int = 252,
        participation: float = 1.0,
    ) -> MCResult:
        """Price a shark-fin note.

        Parameters
        ----------
        barrier : float
            Upper knock-out barrier (absolute level).
        rebate : float
            Cash paid if the barrier is breached.
        participation : float
            Participation rate in the call payoff.

        Returns
        -------
        MCResult
            ``extra`` reports the component breakdown.
        """
        start = time.perf_counter()
        paths = self._gbm.generate_paths(S0, T, r, q, sigma, n_paths, n_steps)
        breached = self._barrier.breach_paths(paths, barrier, "up-and-out")
        S_T = paths[:, -1]

        call_leg = participation * np.maximum(S_T - K, 0.0)
        payoff = np.where(breached, rebate, call_leg)
        disc = np.exp(-r * T) * payoff

        res = MonteCarloEngine.summarize(disc, n_paths=n_paths, method="shark_fin")
        knockout_value = float(np.exp(-r * T) * (breached * rebate).mean())
        upside_value = float(np.exp(-r * T) * (~breached * call_leg).mean())
        res.extra = {
            "components": {
                "knockout_rebate": knockout_value,
                "capped_upside": upside_value,
            },
            "knockout_probability": float(breached.mean()),
            "payoff_samples": disc,
        }
        res.elapsed_seconds = time.perf_counter() - start
        return res
