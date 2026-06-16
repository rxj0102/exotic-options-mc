"""Barrier option pricing (knock-in / knock-out, eight types)."""

from __future__ import annotations

import time

import numpy as np

from benchmarks.barrier_analytical import rubinstein_reiner_barrier
from simulation.base import MCResult, MonteCarloEngine

__all__ = ["BarrierOptionPricer"]

_VALID = {"up-and-in", "up-and-out", "down-and-in", "down-and-out"}


class BarrierOptionPricer:
    """Monte Carlo and analytical pricer for single- and double-barrier options.

    Discrete monitoring is the default; a Brownian-bridge correction is
    available to approximate continuous monitoring by accounting for the
    probability that a path breaches the barrier *between* observation dates.
    """

    def breach_paths(
        self, paths: np.ndarray, H: float, barrier_type: str
    ) -> np.ndarray:
        """Return a boolean mask of paths that breach the barrier.

        Parameters
        ----------
        paths : numpy.ndarray
            Price paths.
        H : float
            Barrier level.
        barrier_type : str
            One of the eight barrier-type strings; only the up/down prefix is
            used here.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape ``(n_paths,)`` that is ``True`` where the
            barrier is touched at least once.
        """
        barrier_type = barrier_type.lower()
        if "up" in barrier_type:
            return paths.max(axis=1) >= H
        return paths.min(axis=1) <= H

    def _breach_prob_bridge(
        self, paths: np.ndarray, H: float, sigma: float, T: float, is_up: bool
    ) -> np.ndarray:
        """Per-path survival probability under a Brownian-bridge correction.

        For each consecutive pair of observations that both lie on the safe
        side of the barrier, the probability of an intermediate crossing is
        ``exp(-2 h1 h2 / (sigma^2 dt))`` where ``h1`` and ``h2`` are the
        log-distances to the barrier. The product over steps gives the
        probability the path never crossed.

        Returns
        -------
        numpy.ndarray
            Probability that each path *survives* (never breaches).
        """
        n_steps = paths.shape[1] - 1
        dt = T / n_steps
        logH = np.log(H)
        logS = np.log(paths)
        # Distance to barrier (positive on the safe side).
        if is_up:
            h = logH - logS  # safe when > 0
        else:
            h = logS - logH
        h1, h2 = h[:, :-1], h[:, 1:]
        # Only meaningful when both endpoints are on the safe side.
        safe = (h1 > 0) & (h2 > 0)
        cross = np.exp(-2.0 * h1 * h2 / (sigma**2 * dt))
        step_survival = np.where(safe, 1.0 - cross, 0.0)
        return np.prod(step_survival, axis=1)

    def price_mc(
        self,
        paths: np.ndarray,
        K: float,
        H: float,
        r: float,
        T: float,
        barrier_type: str,
        option_type: str = "call",
        rebate: float = 0.0,
        sigma: float = None,
        brownian_bridge: bool = False,
    ) -> MCResult:
        """Price a single-barrier option by Monte Carlo.

        Parameters
        ----------
        K, H : float
            Strike and barrier.
        barrier_type : {"up-and-in", "up-and-out", "down-and-in", "down-and-out"}
        option_type : {"call", "put"}
        rebate : float, default 0.0
            Paid (at maturity) when a knock-out is triggered.
        sigma : float, optional
            Volatility; required when ``brownian_bridge`` is ``True``.
        brownian_bridge : bool, default False
            Apply the Brownian-bridge continuity correction.

        Returns
        -------
        MCResult
        """
        start = time.perf_counter()
        barrier_type = barrier_type.lower()
        option_type = option_type.lower()
        if barrier_type not in _VALID:
            raise ValueError(f"barrier_type must be one of {_VALID}")

        S_T = paths[:, -1]
        vanilla = (
            np.maximum(S_T - K, 0.0) if option_type == "call" else np.maximum(K - S_T, 0.0)
        )
        is_up = "up" in barrier_type
        is_in = "in" in barrier_type

        if brownian_bridge:
            if sigma is None:
                raise ValueError("brownian_bridge=True requires sigma")
            survive_prob = self._breach_prob_bridge(paths, H, sigma, T, is_up)
            # Discrete breach already forces survival prob to 0.
            discrete_breach = self.breach_paths(paths, H, barrier_type)
            survive_prob = np.where(discrete_breach, 0.0, survive_prob)
            breach_prob = 1.0 - survive_prob
            if is_in:
                active = breach_prob
            else:
                active = survive_prob
            payoff = active * vanilla
            if rebate and not is_in:
                payoff = payoff + breach_prob * rebate
        else:
            breached = self.breach_paths(paths, H, barrier_type)
            active = breached if is_in else ~breached
            payoff = active.astype(float) * vanilla
            if rebate and not is_in:
                payoff = payoff + (~active).astype(float) * rebate

        disc = np.exp(-r * T) * payoff
        res = MonteCarloEngine.summarize(
            disc, n_paths=paths.shape[0], method=f"barrier_{barrier_type}_{option_type}"
        )
        res.elapsed_seconds = time.perf_counter() - start
        return res

    def price_analytical(
        self,
        S0: float,
        K: float,
        H: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        barrier_type: str,
        option_type: str = "call",
        rebate: float = 0.0,
    ) -> float:
        """Reiner-Rubinstein closed-form barrier price.

        Returns
        -------
        float
        """
        return rubinstein_reiner_barrier(
            S0, K, H, T, r, q, sigma, barrier_type, option_type, rebate
        )

    def price_double_barrier(
        self,
        paths: np.ndarray,
        H_lower: float,
        H_upper: float,
        K: float,
        r: float,
        T: float,
        option_type: str = "call",
        knock: str = "out",
    ) -> MCResult:
        """Price a double-barrier knock-out/knock-in option by Monte Carlo.

        Parameters
        ----------
        H_lower, H_upper : float
            Lower and upper barriers.
        knock : {"out", "in"}
            ``"out"`` pays the vanilla payoff only if the corridor is never
            left; ``"in"`` pays only if either barrier is touched.

        Returns
        -------
        MCResult
        """
        start = time.perf_counter()
        option_type = option_type.lower()
        breached = (paths.min(axis=1) <= H_lower) | (paths.max(axis=1) >= H_upper)
        S_T = paths[:, -1]
        vanilla = (
            np.maximum(S_T - K, 0.0) if option_type == "call" else np.maximum(K - S_T, 0.0)
        )
        active = breached if knock == "in" else ~breached
        disc = np.exp(-r * T) * active.astype(float) * vanilla
        res = MonteCarloEngine.summarize(
            disc, n_paths=paths.shape[0], method=f"double_barrier_{knock}"
        )
        res.elapsed_seconds = time.perf_counter() - start
        return res
