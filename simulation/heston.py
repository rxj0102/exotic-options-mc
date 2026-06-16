"""Heston stochastic volatility path generator."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

from .base import MonteCarloEngine

__all__ = ["HestonSimulator"]


class HestonSimulator(MonteCarloEngine):
    """Heston stochastic-volatility simulator with full-truncation Euler.

    The dynamics are

    .. math::

        dS &= (r - q) S \\, dt + \\sqrt{V}\\, S \\, dW_S, \\\\
        dV &= \\kappa(\\theta - V)\\, dt + \\xi \\sqrt{V}\\, dW_V,

    with :math:`dW_S \\, dW_V = \\rho \\, dt`. Correlated Brownian increments
    are produced via a Cholesky factorisation, and the variance process uses
    the full-truncation scheme :math:`V \\leftarrow \\max(V, 0)`.

    Notes
    -----
    The Feller condition :math:`2\\kappa\\theta > \\xi^2` guarantees the
    variance process stays strictly positive in continuous time; a warning is
    raised when it is violated.
    """

    def generate_paths(  # type: ignore[override]
        self,
        S0: float,
        T: float,
        r: float,
        q: float,
        V0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        n_paths: Optional[int] = None,
        n_steps: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate correlated spot and variance paths.

        Parameters
        ----------
        S0 : float
            Initial spot.
        T : float
            Maturity in years.
        r, q : float
            Risk-free rate and dividend yield.
        V0 : float
            Initial instantaneous variance (``sigma0 ** 2``).
        kappa : float
            Mean-reversion speed.
        theta : float
            Long-run variance.
        xi : float
            Volatility of variance.
        rho : float
            Correlation between the two Brownian motions.
        n_paths, n_steps : int, optional
            Override engine defaults.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Tuple ``(S_paths, V_paths)`` each of shape
            ``(n_paths, n_steps + 1)``.
        """
        if S0 <= 0:
            raise ValueError(f"S0 must be positive, got {S0}")
        if V0 < 0 or theta < 0:
            raise ValueError("V0 and theta must be non-negative")
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"rho must lie in [-1, 1], got {rho}")

        if 2 * kappa * theta <= xi**2:
            warnings.warn(
                "Feller condition 2*kappa*theta > xi^2 is violated; the "
                "variance process may hit zero frequently.",
                RuntimeWarning,
                stacklevel=2,
            )

        n_paths = int(n_paths or self.n_paths)
        n_steps = int(n_steps or self.n_steps)
        dt = T / n_steps if n_steps > 0 else 0.0
        sqrt_dt = np.sqrt(dt)

        # Cholesky factor of the 2x2 correlation matrix.
        chol = np.array([[1.0, 0.0], [rho, np.sqrt(max(1.0 - rho**2, 0.0))]])

        S = np.empty((n_paths, n_steps + 1), dtype=float)
        V = np.empty((n_paths, n_steps + 1), dtype=float)
        S[:, 0] = S0
        V[:, 0] = V0

        log_S = np.full(n_paths, np.log(S0))
        v = np.full(n_paths, float(V0))
        for t in range(1, n_steps + 1):
            z = self._rng.standard_normal(size=(n_paths, 2))
            dW = z @ chol.T * sqrt_dt  # correlated increments
            dW_S, dW_V = dW[:, 0], dW[:, 1]

            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            # Variance update (full truncation).
            v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * dW_V
            # Log-spot update using the truncated variance.
            log_S = log_S + (r - q - 0.5 * v_pos) * dt + sqrt_v * dW_S

            S[:, t] = np.exp(log_S)
            V[:, t] = np.maximum(v, 0.0)

        return S, V

    def feller_condition(
        self, kappa: float, theta: float, xi: float
    ) -> bool:
        """Return ``True`` when the Feller condition holds.

        Parameters
        ----------
        kappa, theta, xi : float
            Heston parameters.

        Returns
        -------
        bool
            ``2 * kappa * theta > xi ** 2``.
        """
        return 2 * kappa * theta > xi**2
