"""Merton jump-diffusion path generator."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import MonteCarloEngine

__all__ = ["MertonJumpDiffusion"]


class MertonJumpDiffusion(MonteCarloEngine):
    """Merton (1976) jump-diffusion simulator.

    The dynamics are

    .. math::

        \\frac{dS}{S} = (r - q - \\lambda \\bar k)\\, dt + \\sigma \\, dW
        + (J - 1)\\, dN,

    where :math:`N` is a Poisson process with intensity :math:`\\lambda`,
    jump sizes are log-normal :math:`\\log J \\sim N(\\mu_J, \\sigma_J^2)`, and
    the compensator is :math:`\\bar k = \\exp(\\mu_J + \\tfrac12 \\sigma_J^2) - 1`.

    Notes
    -----
    Over each step of length :math:`\\Delta t` the number of jumps is drawn
    from a Poisson distribution; the aggregate log jump is normal with mean
    :math:`N \\mu_J` and variance :math:`N \\sigma_J^2`, sampled in a single
    vectorised operation.
    """

    def generate_paths(  # type: ignore[override]
        self,
        S0: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        lambda_: float,
        mu_J: float,
        sigma_J: float,
        n_paths: Optional[int] = None,
        n_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Merton jump-diffusion price paths.

        Parameters
        ----------
        S0 : float
            Initial spot.
        T : float
            Maturity in years.
        r, q : float
            Risk-free rate and dividend yield.
        sigma : float
            Diffusive volatility.
        lambda_ : float
            Jump intensity (expected jumps per year).
        mu_J : float
            Mean of the log jump size.
        sigma_J : float
            Standard deviation of the log jump size.
        n_paths, n_steps : int, optional
            Override engine defaults.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_paths, n_steps + 1)`` with ``S0`` in the
            first column.
        """
        self._validate_inputs(S0, T, sigma)
        if lambda_ < 0 or sigma_J < 0:
            raise ValueError("lambda_ and sigma_J must be non-negative")

        n_paths = int(n_paths or self.n_paths)
        n_steps = int(n_steps or self.n_steps)
        dt = T / n_steps if n_steps > 0 else 0.0
        sqrt_dt = np.sqrt(dt)

        k_bar = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0
        drift = (r - q - lambda_ * k_bar - 0.5 * sigma**2) * dt
        vol = sigma * sqrt_dt

        Z = self._rng.standard_normal(size=(n_paths, n_steps))
        # Vectorised Poisson count of jumps per step.
        N = self._rng.poisson(lam=lambda_ * dt, size=(n_paths, n_steps))
        # Aggregate log jump per step: sum of N normals ~ N(N*mu_J, N*sigma_J^2).
        jump_normals = self._rng.standard_normal(size=(n_paths, n_steps))
        log_jumps = N * mu_J + np.sqrt(N) * sigma_J * jump_normals

        log_increments = drift + vol * Z + log_jumps
        log_paths = np.cumsum(log_increments, axis=1)

        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = S0
        paths[:, 1:] = S0 * np.exp(log_paths)
        return paths
