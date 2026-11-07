"""Capital-protected note pricer."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from simulation.base import MCResult, MonteCarloEngine
from simulation.gbm import GBMSimulator

__all__ = ["CapitalProtectedPricer"]


class CapitalProtectedPricer:
    """Monte Carlo pricer for a capital-protected note.

    The note returns at least the protected fraction of principal at maturity
    plus a participation in the upside of the underlying. It is economically a
    zero-coupon bond plus a participation-weighted call.

    Parameters
    ----------
    seed : int, optional
        Seed for the internal GBM simulator.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._gbm = GBMSimulator(seed=seed)

    def price(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        participation_rate: float = 1.0,
        protection: float = 1.0,
        notional: float = 100.0,
        n_paths: int = 100_000,
        n_steps: int = 252,
    ) -> MCResult:
        """Price a capital-protected note.

        Parameters
        ----------
        protection : float
            Fraction of notional guaranteed at maturity (e.g. ``1.0`` for full
            protection).
        participation_rate : float
            Participation in the underlying's upside.

        Returns
        -------
        MCResult
            ``extra`` reports the bond floor and option component values.
        """
        start = time.perf_counter()
        paths = self._gbm.generate_paths(S0, T, r, q, sigma, n_paths, n_steps)
        S_T = paths[:, -1]
        units = participation_rate * notional / S0

        floor = protection * notional
        payoff = floor + units * np.maximum(S_T - K, 0.0)
        disc = np.exp(-r * T) * payoff

        res = MonteCarloEngine.summarize(
            disc, n_paths=n_paths, method="capital_protected"
        )
        bond_value = float(floor * np.exp(-r * T))
        option_value = float(np.exp(-r * T) * (units * np.maximum(S_T - K, 0.0)).mean())
        res.extra = {
            "components": {
                "zero_coupon_bond": bond_value,
                "call_participation": option_value,
            },
            "payoff_samples": disc,
        }
        res.elapsed_seconds = time.perf_counter() - start
        return res
