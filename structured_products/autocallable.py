"""Autocallable note pricer."""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from simulation.base import MCResult, MonteCarloEngine
from simulation.gbm import GBMSimulator

__all__ = ["AutocallablePricer"]


class AutocallablePricer:
    """Monte Carlo pricer for autocallable notes.

    On each observation date the note redeems early at par plus an accrued
    coupon if the spot is at or above the autocall barrier. If it survives to
    maturity, the investor receives par unless the spot has fallen below the
    capital barrier, in which case principal is reduced in line with the
    underlying.

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
        autocall_barrier: float,
        coupon: float,
        capital_barrier: float,
        tenor: float,
        obs_dates: Sequence[float],
        r: float,
        q: float,
        sigma: float,
        n_paths: int = 100_000,
        notional: float = 100.0,
        n_steps_per_year: int = 252,
    ) -> MCResult:
        """Price an autocallable note.

        Parameters
        ----------
        autocall_barrier, capital_barrier : float
            Levels as multiples of ``S0`` (e.g. ``1.0`` and ``0.6``).
        coupon : float
            Annualised coupon rate; the redemption coupon at observation ``i``
            is ``coupon * t_i`` of notional.
        obs_dates : sequence of float
            Observation times in years (each ``<= tenor``).

        Returns
        -------
        MCResult
            ``extra`` carries ``autocall_probs`` (per date), ``expected_life``
            and ``payoff_samples``.
        """
        start = time.perf_counter()
        obs_dates = np.asarray(sorted(obs_dates), dtype=float)
        n_steps = max(int(round(tenor * n_steps_per_year)), len(obs_dates))
        paths = self._gbm.generate_paths(S0, tenor, r, q, sigma, n_paths, n_steps)

        # Map observation times to path columns.
        times = np.linspace(0.0, tenor, n_steps + 1)
        obs_cols = np.searchsorted(times, obs_dates)
        obs_cols = np.clip(obs_cols, 1, n_steps)

        ac_level = autocall_barrier * S0
        cap_level = capital_barrier * S0

        n = paths.shape[0]
        alive = np.ones(n, dtype=bool)
        pv = np.zeros(n)
        autocall_counts = np.zeros(len(obs_dates))
        life = np.full(n, tenor)

        for i, col in enumerate(obs_cols):
            t_i = obs_dates[i]
            S_obs = paths[:, col]
            triggered = alive & (S_obs >= ac_level)
            redemption = notional * (1.0 + coupon * t_i)
            pv[triggered] = redemption * np.exp(-r * t_i)
            life[triggered] = t_i
            autocall_counts[i] = triggered.sum()
            alive &= ~triggered

        # Survivors to maturity.
        S_T = paths[:, -1]
        below = alive & (S_T < cap_level)
        at_par = alive & ~below
        pv[at_par] = notional * np.exp(-r * tenor)
        pv[below] = notional * (S_T[below] / S0) * np.exp(-r * tenor)

        res = MonteCarloEngine.summarize(pv, n_paths=n, method="autocallable")
        res.extra = {
            "autocall_probs": (autocall_counts / n).tolist(),
            "obs_dates": obs_dates.tolist(),
            "expected_life": float(life.mean()),
            "prob_capital_loss": float(below.mean()),
            "payoff_samples": pv,
        }
        res.elapsed_seconds = time.perf_counter() - start
        return res
