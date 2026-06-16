"""Importance sampling via Girsanov drift shifts."""

from __future__ import annotations

import time

import numpy as np

from simulation.base import MCResult, MonteCarloEngine

__all__ = ["ImportanceSampling"]


class ImportanceSampling:
    """Importance sampling for rare-event option payoffs.

    A drift shift of the driving Brownian motion (Girsanov change of measure)
    pushes simulated paths toward the region that dominates the payoff — for a
    deep out-of-the-money call, toward the strike. The estimator is corrected
    by the Radon-Nikodym likelihood ratio

    .. math::

        L = \\exp\\!\\big(-\\theta W_T - \\tfrac12 \\theta^2 T\\big),

    so that :math:`\\mathbb{E}^P[f(S)] = \\mathbb{E}^{\\tilde P}[f(S) L]`.

    Parameters
    ----------
    seed : int, optional
        Seed for the internal generator.
    """

    def __init__(self, seed: int = None) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def optimal_shift(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Heuristic optimal drift shift ``theta`` (in standard deviations).

        Chooses the shift that recentres the terminal log-price on the strike,
        which is near-optimal for OTM digital and vanilla payoffs.

        Returns
        -------
        float
            The shift ``theta`` applied to the unit-variance driver.
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        # Mean of log(S_T/S0) under P is (r - 0.5 sigma^2) T; to land on log(K/S0)
        # the total Brownian displacement sigma*sqrt(T)*theta must cover the gap.
        gap = np.log(K / S0) - (r - 0.5 * sigma**2) * T
        return float(gap / (sigma * T))  # theta scales the per-unit-time drift

    def price_otm_call(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        n_paths: int,
        theta: float = None,
        option_type: str = "call",
    ) -> MCResult:
        """Price a (possibly deep OTM) European option via importance sampling.

        Parameters
        ----------
        theta : float, optional
            Drift shift; defaults to :meth:`optimal_shift`.

        Returns
        -------
        MCResult
            ``extra`` reports the crude variance and variance-reduction ratio.
        """
        start = time.perf_counter()
        if theta is None:
            theta = self.optimal_shift(S0, K, T, r, sigma, option_type)

        Z = self._rng.standard_normal(n_paths)
        sqrtT = np.sqrt(T)
        # Shifted Brownian terminal: W_T = sqrt(T) * Z + theta * T.
        W_T = sqrtT * Z + theta * T
        S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * W_T)
        # Likelihood ratio dP/dP~ for the shift theta on the drift of W.
        L = np.exp(-theta * W_T + 0.5 * theta**2 * T)

        disc = np.exp(-r * T)
        if option_type == "call":
            payoff = np.maximum(S_T - K, 0.0)
        else:
            payoff = np.maximum(K - S_T, 0.0)
        weighted = disc * payoff * L

        res = MonteCarloEngine.summarize(weighted, n_paths=n_paths, method="importance_sampling")

        # Crude comparison (unshifted) for the VR ratio.
        S_T_crude = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * sqrtT * Z)
        crude_payoff = disc * (
            np.maximum(S_T_crude - K, 0.0)
            if option_type == "call"
            else np.maximum(K - S_T_crude, 0.0)
        )
        crude_var = np.var(crude_payoff, ddof=1)
        is_var = np.var(weighted, ddof=1)
        res.extra = {
            "theta": float(theta),
            "crude_variance": float(crude_var),
            "is_variance": float(is_var),
            "vr_ratio": float(crude_var / is_var) if is_var > 0 else float("inf"),
        }
        res.elapsed_seconds = time.perf_counter() - start
        return res

    def price_barrier_near_breach(
        self,
        S0: float,
        K: float,
        H: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        n_paths: int,
        theta: float = None,
        n_steps: int = 252,
        barrier_type: str = "down-and-in",
        option_type: str = "call",
    ) -> MCResult:
        """Price a knock-in barrier near its breach probability via IS.

        The drift is shifted toward the barrier so that breaching paths — which
        carry all the payoff for a knock-in — are sampled far more often.

        Returns
        -------
        MCResult
        """
        start = time.perf_counter()
        if theta is None:
            # Shift toward the barrier.
            gap = np.log(H / S0) - (r - 0.5 * sigma**2) * T
            theta = float(gap / (sigma * T))

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        Z = self._rng.standard_normal((n_paths, n_steps))
        # Shifted increments add theta*dt to each Brownian step.
        dW = sqrt_dt * Z + theta * dt
        W = np.cumsum(dW, axis=1)
        t = np.arange(1, n_steps + 1) * dt
        log_paths = (r - q - 0.5 * sigma**2) * t + sigma * W
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = S0
        paths[:, 1:] = S0 * np.exp(log_paths)

        # Likelihood ratio over the whole path.
        W_T = W[:, -1]
        L = np.exp(-theta * W_T + 0.5 * theta**2 * T)

        S_T = paths[:, -1]
        vanilla = (
            np.maximum(S_T - K, 0.0) if option_type == "call" else np.maximum(K - S_T, 0.0)
        )
        if "up" in barrier_type:
            breached = paths.max(axis=1) >= H
        else:
            breached = paths.min(axis=1) <= H
        active = breached if "in" in barrier_type else ~breached
        weighted = np.exp(-r * T) * active.astype(float) * vanilla * L

        res = MonteCarloEngine.summarize(weighted, n_paths=n_paths, method="is_barrier")
        res.extra = {"theta": float(theta)}
        res.elapsed_seconds = time.perf_counter() - start
        return res
