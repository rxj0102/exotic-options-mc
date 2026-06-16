"""Quasi-Monte Carlo path generation (Sobol and Halton)."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.stats import qmc

from simulation.base import MCResult, MonteCarloEngine, normal_ppf

__all__ = ["QuasiMonteCarlo"]


class QuasiMonteCarlo:
    """Low-discrepancy path generation using Sobol and Halton sequences.

    Quasi-random sequences fill the unit hypercube more evenly than
    pseudo-random draws, yielding an asymptotic error of
    :math:`O((\\log N)^d / N)` versus :math:`O(N^{-1/2})` for crude MC. A
    Brownian-bridge construction is offered so that the most significant
    dimensions of the sequence drive the coarsest features of the path.

    Parameters
    ----------
    seed : int, optional
        Seed for scrambling.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------
    def generate_sobol(self, n_paths: int, n_dims: int) -> np.ndarray:
        """Return a scrambled Sobol sample mapped to standard normals.

        Parameters
        ----------
        n_paths : int
            Number of points (rounded up internally to a power of two for
            balance, then truncated).
        n_dims : int
            Sequence dimensionality.

        Returns
        -------
        numpy.ndarray
            Standard normal array of shape ``(n_paths, n_dims)``.
        """
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=self._rng)
        u = sampler.random(n=n_paths)
        return normal_ppf(u)

    def generate_halton(self, n_paths: int, n_dims: int) -> np.ndarray:
        """Return a scrambled Halton sample mapped to standard normals."""
        sampler = qmc.Halton(d=n_dims, scramble=True, seed=self._rng)
        u = sampler.random(n=n_paths)
        return normal_ppf(u)

    # ------------------------------------------------------------------
    # Path construction
    # ------------------------------------------------------------------
    def _gbm_from_normals(
        self, Z: np.ndarray, S0, T, r, q, sigma, bridge: bool
    ) -> np.ndarray:
        n_paths, n_steps = Z.shape
        dt = T / n_steps
        if bridge:
            increments = self.brownian_bridge_increments(Z, T)
        else:
            increments = np.sqrt(dt) * Z
        W = np.cumsum(increments, axis=1)
        t = np.arange(1, n_steps + 1) * dt
        log_paths = (r - q - 0.5 * sigma**2) * t + sigma * W
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = S0
        paths[:, 1:] = S0 * np.exp(log_paths)
        return paths

    def paths_from_sobol(
        self,
        S0: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        n_paths: int,
        n_steps: int,
        bridge: bool = True,
    ) -> np.ndarray:
        """Generate GBM paths from a Sobol sequence.

        Parameters
        ----------
        bridge : bool, default True
            Use the Brownian-bridge construction to align the leading Sobol
            dimensions with the path's low-frequency structure.

        Returns
        -------
        numpy.ndarray
            Price paths of shape ``(n_paths, n_steps + 1)``.
        """
        Z = self.generate_sobol(n_paths, n_steps)
        return self._gbm_from_normals(Z, S0, T, r, q, sigma, bridge)

    def paths_from_halton(
        self,
        S0: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        n_paths: int,
        n_steps: int,
        bridge: bool = True,
    ) -> np.ndarray:
        """Generate GBM paths from a Halton sequence."""
        Z = self.generate_halton(n_paths, n_steps)
        return self._gbm_from_normals(Z, S0, T, r, q, sigma, bridge)

    def brownian_bridge_increments(self, Z: np.ndarray, T: float) -> np.ndarray:
        """Construct Brownian increments via the Brownian-bridge ordering.

        The terminal value is set first (using the lowest-discrepancy
        dimension), then successively bisected. This places the variance of
        the path into the leading sequence dimensions where the sequence is
        most uniform.

        Parameters
        ----------
        Z : numpy.ndarray
            Standard normals of shape ``(n_paths, n_steps)``.
        T : float
            Maturity.

        Returns
        -------
        numpy.ndarray
            Brownian increments ``dW`` of shape ``(n_paths, n_steps)``.
        """
        n_paths, n_steps = Z.shape
        dt = T / n_steps
        W = np.zeros((n_paths, n_steps + 1))
        # Times indexed 0..n_steps; W[0] = 0.
        W[:, n_steps] = np.sqrt(T) * Z[:, 0]

        # Iterative bisection schedule.
        # left/right indices and the dimension counter.
        idx = 1
        schedule = [(0, n_steps)]
        while schedule:
            left, right = schedule.pop(0)
            if right - left <= 1:
                continue
            mid = (left + right) // 2
            t_l, t_m, t_r = left * dt, mid * dt, right * dt
            mean = W[:, left] + (W[:, right] - W[:, left]) * (t_m - t_l) / (t_r - t_l)
            std = np.sqrt((t_m - t_l) * (t_r - t_m) / (t_r - t_l))
            if idx < n_steps:
                W[:, mid] = mean + std * Z[:, idx]
                idx += 1
            else:
                W[:, mid] = mean
            schedule.append((left, mid))
            schedule.append((mid, right))

        return np.diff(W, axis=1)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def discrepancy_comparison(self, n_paths: int, n_dims: int = 2) -> dict:
        """Compare the star-discrepancy of Sobol, Halton and pseudo-random.

        Returns
        -------
        dict
            Discrepancy for each sampler; lower is better.
        """
        rng = np.random.default_rng(self.seed)
        sob = qmc.Sobol(d=n_dims, scramble=True, seed=rng).random(n_paths)
        hal = qmc.Halton(d=n_dims, scramble=True, seed=rng).random(n_paths)
        prng = rng.random((n_paths, n_dims))
        return {
            "sobol": float(qmc.discrepancy(sob)),
            "halton": float(qmc.discrepancy(hal)),
            "pseudo_random": float(qmc.discrepancy(prng)),
        }

    def price(
        self,
        payoff_fn,
        paths: np.ndarray,
        r: float,
        T: float,
        method: str = "sobol_qmc",
    ) -> MCResult:
        """Discount a payoff over a single QMC point set.

        Note that QMC point sets are deterministic, so the reported standard
        error (which assumes i.i.d. samples) is only a rough proxy; use
        :meth:`price_rqmc` for an unbiased randomized-QMC error estimate.
        """
        start = time.perf_counter()
        disc = np.exp(-r * T)
        payoffs = np.asarray(payoff_fn(paths), dtype=float) * disc
        res = MonteCarloEngine.summarize(payoffs, n_paths=paths.shape[0], method=method)
        res.elapsed_seconds = time.perf_counter() - start
        return res

    def price_rqmc(
        self,
        payoff_fn,
        S0: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        n_paths: int,
        n_steps: int,
        n_replications: int = 16,
        sequence: str = "sobol",
        bridge: bool = True,
        method: str = "sobol_qmc",
    ) -> MCResult:
        """Randomized-QMC price with an unbiased standard error.

        The estimator averages ``n_replications`` independently scrambled
        point sets. The standard error is the sample standard deviation of the
        per-replication means divided by ``sqrt(n_replications)`` — the proper
        way to quantify QMC's variance reduction.

        Parameters
        ----------
        n_replications : int, default 16
            Number of independent scrambles.
        sequence : {"sobol", "halton"}
            Low-discrepancy sequence to randomize.

        Returns
        -------
        MCResult
        """
        start = time.perf_counter()
        disc = np.exp(-r * T)
        gen = self.generate_sobol if sequence == "sobol" else self.generate_halton
        rep_means = np.empty(n_replications)
        for i in range(n_replications):
            Z = gen(n_paths, n_steps)
            paths = self._gbm_from_normals(Z, S0, T, r, q, sigma, bridge)
            rep_means[i] = np.mean(disc * np.asarray(payoff_fn(paths), dtype=float))

        price = float(np.mean(rep_means))
        stderr = (
            float(np.std(rep_means, ddof=1) / np.sqrt(n_replications))
            if n_replications > 1
            else 0.0
        )
        half = MonteCarloEngine.Z_95 * stderr
        res = MCResult(
            price=price,
            stderr=stderr,
            ci_lower=price - half,
            ci_upper=price + half,
            n_paths=n_paths * n_replications,
            method=method,
            extra={"n_replications": n_replications},
        )
        res.elapsed_seconds = time.perf_counter() - start
        return res
