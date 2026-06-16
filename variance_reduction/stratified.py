"""Stratified sampling and Latin hypercube sampling."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.stats import qmc

from simulation.base import MCResult, MonteCarloEngine, normal_ppf

__all__ = ["StratifiedSampling"]


class StratifiedSampling:
    """Stratified and Latin hypercube sampling of the terminal distribution.

    Stratification partitions ``[0, 1]`` into ``n`` equal bins and draws one
    uniform from each, guaranteeing even coverage and reducing the variance of
    estimators that depend smoothly on the terminal value.

    Parameters
    ----------
    seed : int, optional
        Seed for the internal generator.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate_stratified_normals(
        self, n_paths: int, n_strata: Optional[int] = None
    ) -> np.ndarray:
        """Return ``n_paths`` stratified standard normals.

        Parameters
        ----------
        n_strata : int, optional
            Number of strata; defaults to ``n_paths`` (one sample per
            stratum).

        Returns
        -------
        numpy.ndarray
            Array of standard normal variates of length ``n_paths``.
        """
        n_strata = int(n_strata or n_paths)
        # Allocate samples to strata as evenly as possible.
        base = n_paths // n_strata
        rem = n_paths - base * n_strata
        counts = np.full(n_strata, base, dtype=int)
        counts[:rem] += 1

        edges = np.arange(n_strata) / n_strata
        u = np.concatenate(
            [
                (edges[i] + self._rng.random(counts[i]) / n_strata)
                for i in range(n_strata)
                if counts[i] > 0
            ]
        )
        self._rng.shuffle(u)
        return normal_ppf(u)

    def latin_hypercube_normals(self, n_paths: int, n_dims: int) -> np.ndarray:
        """Return a Latin hypercube sample mapped to standard normals.

        Parameters
        ----------
        n_dims : int
            Dimensionality (e.g. number of time steps).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_paths, n_dims)``.
        """
        sampler = qmc.LatinHypercube(d=n_dims, seed=self._rng)
        u = sampler.random(n=n_paths)
        return normal_ppf(u)

    def price_terminal(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        n_paths: int,
        option_type: str = "call",
        n_strata: Optional[int] = None,
    ) -> MCResult:
        """Price a European option using stratified terminal sampling.

        Returns
        -------
        MCResult
        """
        start = time.perf_counter()
        Z = self.generate_stratified_normals(n_paths, n_strata)
        S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        if option_type == "call":
            payoff = np.maximum(S_T - K, 0.0)
        else:
            payoff = np.maximum(K - S_T, 0.0)
        disc = np.exp(-r * T) * payoff
        res = MonteCarloEngine.summarize(disc, n_paths=n_paths, method="stratified")
        res.elapsed_seconds = time.perf_counter() - start
        return res
