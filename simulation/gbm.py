"""Geometric Brownian Motion path generator."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import MonteCarloEngine

__all__ = ["GBMSimulator"]


class GBMSimulator(MonteCarloEngine):
    """Exact-discretisation Geometric Brownian Motion simulator.

    The risk-neutral dynamics are

    .. math::

        dS = (r - q) S \\, dt + \\sigma S \\, dW,

    discretised exactly via the log-Euler scheme

    .. math::

        S_{t+\\Delta t} = S_t \\exp\\!\\Big((r - q - \\tfrac12 \\sigma^2)
        \\Delta t + \\sigma \\sqrt{\\Delta t}\\, Z\\Big), \\quad Z \\sim N(0, 1).

    Examples
    --------
    >>> sim = GBMSimulator(seed=0)
    >>> paths = sim.generate_paths(100.0, 1.0, 0.05, 0.0, 0.2, n_paths=10, n_steps=4)
    >>> paths.shape
    (10, 5)
    >>> bool(np.allclose(paths[:, 0], 100.0))
    True
    """

    def generate_paths(
        self,
        S0: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        n_paths: Optional[int] = None,
        n_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Generate GBM price paths.

        Parameters
        ----------
        S0 : float
            Initial spot price.
        T : float
            Time to maturity in years.
        r, q : float
            Risk-free rate and continuous dividend yield.
        sigma : float
            Volatility.
        n_paths, n_steps : int, optional
            Override the engine defaults.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_paths, n_steps + 1)`` with ``S0`` in the
            first column.
        """
        self._validate_inputs(S0, T, sigma)
        n_paths = int(n_paths or self.n_paths)
        n_steps = int(n_steps or self.n_steps)

        Z = self._rng.standard_normal(size=(n_paths, n_steps))
        return self._paths_from_normals(Z, S0, T, r, q, sigma)

    def generate_paths_antithetic(
        self,
        S0: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        n_paths: Optional[int] = None,
        n_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Generate antithetic GBM paths.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(2 * n_paths, n_steps + 1)``; the first
            ``n_paths`` rows use draws ``Z`` and the next ``n_paths`` rows
            use the antithetic draws ``-Z``.
        """
        self._validate_inputs(S0, T, sigma)
        n_paths = int(n_paths or self.n_paths)
        n_steps = int(n_steps or self.n_steps)

        Z = self._rng.standard_normal(size=(n_paths, n_steps))
        Z_full = np.concatenate([Z, -Z], axis=0)
        return self._paths_from_normals(Z_full, S0, T, r, q, sigma)

    def _paths_from_normals(
        self,
        Z: np.ndarray,
        S0: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> np.ndarray:
        """Construct price paths from a matrix of standard normals.

        Parameters
        ----------
        Z : numpy.ndarray
            Array of shape ``(n_paths, n_steps)`` of standard normals.

        Returns
        -------
        numpy.ndarray
            Price paths of shape ``(n_paths, n_steps + 1)``.
        """
        n_paths, n_steps = Z.shape
        dt = T / n_steps if n_steps > 0 else 0.0
        drift = (r - q - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)

        log_increments = drift + vol * Z
        log_paths = np.cumsum(log_increments, axis=1)
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = S0
        paths[:, 1:] = S0 * np.exp(log_paths)
        return paths

    def terminal_distribution(
        self,
        S0: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        n_paths: Optional[int] = None,
        n_bins: int = 60,
    ) -> dict:
        """Return histogram and theoretical log-normal density of ``S_T``.

        Parameters
        ----------
        n_bins : int, default 60
            Number of histogram bins.

        Returns
        -------
        dict
            Keys ``bin_centers``, ``hist_density``, ``pdf_x``, ``pdf_y`` and
            the theoretical moments ``mean`` and ``var`` of ``S_T``.
        """
        paths = self.generate_paths(S0, T, r, q, sigma, n_paths=n_paths, n_steps=1)
        S_T = paths[:, -1]

        counts, edges = np.histogram(S_T, bins=n_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Theoretical log-normal parameters of S_T.
        mu = np.log(S0) + (r - q - 0.5 * sigma**2) * T
        s = sigma * np.sqrt(T)
        pdf_x = np.linspace(edges[0], edges[-1], 400)
        with np.errstate(divide="ignore"):
            pdf_y = np.where(
                pdf_x > 0,
                np.exp(-((np.log(pdf_x) - mu) ** 2) / (2 * s**2))
                / (pdf_x * s * np.sqrt(2 * np.pi)),
                0.0,
            )
        return {
            "bin_centers": centers,
            "hist_density": counts,
            "pdf_x": pdf_x,
            "pdf_y": pdf_y,
            "mean": float(S0 * np.exp((r - q) * T)),
            "var": float(
                S0**2 * np.exp(2 * (r - q) * T) * (np.exp(sigma**2 * T) - 1.0)
            ),
        }
