"""Monte Carlo Greeks via pathwise, likelihood-ratio and finite-difference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from simulation.gbm import GBMSimulator

__all__ = ["MCGreeks", "GreekSet"]


@dataclass
class GreekSet:
    """Container for a set of Monte Carlo Greeks and their standard errors."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    method: str = "auto"
    stderr: Optional[dict] = None


class MCGreeks:
    """Monte Carlo sensitivities via three complementary estimators.

    * **Pathwise** differentiates the (Lipschitz) payoff through the path —
      low variance but requires a differentiable payoff.
    * **Likelihood ratio** differentiates the density instead, handling
      discontinuous payoffs (digitals, barriers).
    * **Finite difference** bumps a parameter and reprices — a universal
      fallback using common random numbers to control variance.

    Parameters
    ----------
    seed : int, optional
        Seed for the internal generator (shared across bumps for CRN).
    n_steps : int, default 252
        Time-discretisation used when generating paths internally.
    """

    def __init__(self, seed: Optional[int] = None, n_steps: int = 252) -> None:
        self.seed = seed
        self.n_steps = n_steps

    # ------------------------------------------------------------------
    # Pathwise (terminal European-style payoffs)
    # ------------------------------------------------------------------
    def delta_pathwise(
        self,
        paths: np.ndarray,
        S0: float,
        K: float,
        r: float,
        T: float,
        option_type: str = "call",
    ):
        """Pathwise delta for a terminal-value payoff.

        For a call, ``d/dS0 E[e^{-rT}(S_T-K)^+] = e^{-rT} E[1{S_T>K} S_T/S0]``.

        Returns
        -------
        (float, float)
            Delta estimate and its standard error.
        """
        S_T = paths[:, -1]
        disc = np.exp(-r * T)
        if option_type == "call":
            est = disc * (S_T > K) * S_T / S0
        else:
            est = -disc * (S_T < K) * S_T / S0
        return float(est.mean()), float(est.std(ddof=1) / np.sqrt(est.size))

    def vega_pathwise(
        self,
        paths: np.ndarray,
        S0: float,
        K: float,
        r: float,
        T: float,
        sigma: float,
        option_type: str = "call",
    ):
        """Pathwise vega for a terminal-value payoff.

        Returns
        -------
        (float, float)
        """
        S_T = paths[:, -1]
        disc = np.exp(-r * T)
        # dS_T/dsigma = S_T * (log(S_T/S0) - (r - 0.5 sigma^2) T) / sigma.
        dST_dsig = S_T * (np.log(S_T / S0) - (r - 0.5 * sigma**2) * T) / sigma
        if option_type == "call":
            est = disc * (S_T > K) * dST_dsig
        else:
            est = -disc * (S_T < K) * dST_dsig
        return float(est.mean()), float(est.std(ddof=1) / np.sqrt(est.size))

    # ------------------------------------------------------------------
    # Likelihood ratio (handles discontinuous payoffs)
    # ------------------------------------------------------------------
    def delta_likelihood_ratio(
        self,
        paths: np.ndarray,
        payoff: np.ndarray,
        S0: float,
        r: float,
        T: float,
        sigma: float,
    ):
        """Likelihood-ratio delta for an arbitrary (possibly discontinuous) payoff.

        Uses the score of the terminal log-normal:
        ``Delta_LR = E[disc * payoff * Z / (S0 sigma sqrt(T))]`` where ``Z`` is
        the standard normal driving the terminal value.

        Returns
        -------
        (float, float)
        """
        S_T = paths[:, -1]
        Z = (np.log(S_T / S0) - (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        disc = np.exp(-r * T)
        score = Z / (S0 * sigma * np.sqrt(T))
        est = disc * np.asarray(payoff, dtype=float) * score
        return float(est.mean()), float(est.std(ddof=1) / np.sqrt(est.size))

    # ------------------------------------------------------------------
    # Finite-difference fallback (universal, with common random numbers)
    # ------------------------------------------------------------------
    def finite_difference(
        self,
        price_fn: Callable[[float], float],
        x0: float,
        h: Optional[float] = None,
    ):
        """Central-difference derivative of ``price_fn`` at ``x0``.

        Parameters
        ----------
        price_fn : callable
            Maps a parameter value to a price (should reuse common random
            numbers internally for low variance).
        h : float, optional
            Bump size; defaults to ``max(1e-4, 1e-4 * |x0|)``.

        Returns
        -------
        float
        """
        if h is None:
            h = max(1e-4, 1e-4 * abs(x0))
        return float((price_fn(x0 + h) - price_fn(x0 - h)) / (2 * h))

    def all_greeks(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        n_paths: int = 100_000,
        method: str = "finite_difference",
    ) -> GreekSet:
        """Compute a full set of Greeks via finite differences with CRN.

        A single fixed seed is reused across all bumps so that common random
        numbers dramatically reduce the variance of the differences.

        Returns
        -------
        GreekSet
        """
        seed = self.seed if self.seed is not None else 12345

        def price_at(S0_=S0, sigma_=sigma, r_=r, T_=T):
            # Reuse the same seed across bumps for common random numbers, and
            # discount with the bumped rate so rho captures both drift and
            # discount sensitivity.
            gbm = GBMSimulator(seed=seed)
            paths = gbm.generate_paths(S0_, T_, r_, q, sigma_, n_paths, self.n_steps)
            return float((np.exp(-r_ * T_) * payoff_fn(paths)).mean())

        base = price_at()
        hS = max(1e-3 * S0, 1e-3)
        hsig = 1e-3
        hr = 1e-4
        hT = min(1e-3, T / 4) if T > 0 else 1e-3

        up_S, dn_S = price_at(S0_=S0 + hS), price_at(S0_=S0 - hS)
        delta = (up_S - dn_S) / (2 * hS)
        gamma = (up_S - 2 * base + dn_S) / hS**2
        vega = (price_at(sigma_=sigma + hsig) - price_at(sigma_=sigma - hsig)) / (
            2 * hsig
        )
        rho = (price_at(r_=r + hr) - price_at(r_=r - hr)) / (2 * hr)
        # Theta is the negative time derivative (value decays as T shrinks).
        theta = -(price_at(T_=T) - price_at(T_=T - hT)) / hT if T > hT else 0.0

        return GreekSet(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho),
            method=method,
        )
