"""Abstract Monte Carlo engine and shared result container.

This module defines the common interface that every path generator and
pricer in the library adheres to. The :class:`MCResult` dataclass is the
uniform return type used across all pricing routines so that downstream
diagnostics, risk and reporting code can consume any pricer identically.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import stats

__all__ = ["MCResult", "MonteCarloEngine"]


@dataclass
class MCResult:
    """Container for the output of a Monte Carlo pricing routine.

    Attributes
    ----------
    price : float
        Discounted Monte Carlo estimate of the contract value.
    stderr : float
        Standard error of the Monte Carlo estimate.
    ci_lower, ci_upper : float
        Lower and upper bounds of the 95% confidence interval.
    n_paths : int
        Number of simulated paths used for the estimate.
    elapsed_seconds : float
        Wall-clock time spent producing the estimate.
    method : str
        Human-readable label of the estimator (e.g. ``"crude_mc"``).
    extra : dict
        Optional free-form payload for method-specific diagnostics.
    """

    price: float
    stderr: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    elapsed_seconds: float = 0.0
    method: str = "crude_mc"
    extra: dict = field(default_factory=dict)

    @property
    def ci_95(self) -> Tuple[float, float]:
        """Return the 95% confidence interval as a ``(lower, upper)`` tuple."""
        return (self.ci_lower, self.ci_upper)

    @property
    def ci_width(self) -> float:
        """Return the width of the 95% confidence interval."""
        return self.ci_upper - self.ci_lower

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.method}: {self.price:.6f} +/- {self.stderr:.6f} "
            f"(95% CI [{self.ci_lower:.6f}, {self.ci_upper:.6f}], "
            f"n={self.n_paths}, {self.elapsed_seconds:.3f}s)"
        )


class MonteCarloEngine(ABC):
    """Abstract base class for all Monte Carlo path generators.

    Parameters
    ----------
    seed : int, optional
        Seed for the underlying random number generator. When provided the
        engine is fully reproducible.
    n_paths : int, default 100_000
        Default number of paths used when not otherwise specified.
    n_steps : int, default 252
        Default number of time steps (one trading year by default).

    Notes
    -----
    Concrete subclasses must implement :meth:`generate_paths`. The
    :meth:`price` method is provided as a concrete helper that takes a payoff
    callable and a path array; subclasses may override it for specialised
    behaviour.
    """

    Z_95 = 1.959963984540054  # two-sided 95% normal quantile

    def __init__(
        self,
        seed: Optional[int] = None,
        n_paths: int = 100_000,
        n_steps: int = 252,
    ) -> None:
        self.seed = seed
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self._rng = np.random.default_rng(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        """Reseed the engine's random number generator.

        Parameters
        ----------
        seed : int or None
            New seed value. ``None`` draws fresh entropy from the OS.
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def rng(self) -> np.random.Generator:
        """Return the engine's :class:`numpy.random.Generator`."""
        return self._rng

    @abstractmethod
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
        """Generate simulated price paths.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_paths, n_steps + 1)`` whose first column is
            ``S0``.
        """
        raise NotImplementedError

    def price(
        self,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        paths: np.ndarray,
        r: float,
        T: float,
        method: str = "crude_mc",
    ) -> MCResult:
        """Discount a payoff over simulated paths into an :class:`MCResult`.

        Parameters
        ----------
        payoff_fn : callable
            Maps a ``(n_paths, n_steps + 1)`` path array to a
            ``(n_paths,)`` array of undiscounted terminal payoffs.
        paths : numpy.ndarray
            Simulated price paths.
        r : float
            Continuously compounded risk-free rate used for discounting.
        T : float
            Time to maturity in years.
        method : str, default ``"crude_mc"``
            Label stored on the returned result.

        Returns
        -------
        MCResult
        """
        start = time.perf_counter()
        discount = np.exp(-r * T)
        payoffs = np.asarray(payoff_fn(paths), dtype=float) * discount
        result = self.summarize(payoffs, n_paths=paths.shape[0], method=method)
        result.elapsed_seconds = time.perf_counter() - start
        return result

    @classmethod
    def summarize(
        cls,
        discounted_payoffs: np.ndarray,
        n_paths: Optional[int] = None,
        method: str = "crude_mc",
    ) -> MCResult:
        """Summarise an array of already-discounted payoffs.

        Parameters
        ----------
        discounted_payoffs : numpy.ndarray
            One-dimensional array of discounted payoffs.
        n_paths : int, optional
            Number of paths; defaults to the length of the payoff array.
        method : str
            Estimator label.

        Returns
        -------
        MCResult
        """
        x = np.asarray(discounted_payoffs, dtype=float).ravel()
        n = int(n_paths if n_paths is not None else x.size)
        price = float(np.mean(x))
        # ddof=1 sample standard deviation; guard against n == 1.
        sample_std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        stderr = sample_std / np.sqrt(x.size) if x.size > 0 else 0.0
        half = cls.Z_95 * stderr
        return MCResult(
            price=price,
            stderr=stderr,
            ci_lower=price - half,
            ci_upper=price + half,
            n_paths=n,
            method=method,
        )

    @staticmethod
    def _validate_inputs(S0: float, T: float, sigma: float) -> None:
        """Validate common simulation inputs, raising on invalid values."""
        if S0 <= 0:
            raise ValueError(f"S0 must be positive, got {S0}")
        if T < 0:
            raise ValueError(f"T must be non-negative, got {T}")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")


def normal_ppf(u: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF, clipping inputs away from {0, 1}.

    Parameters
    ----------
    u : numpy.ndarray
        Array of uniforms in ``(0, 1)``.

    Returns
    -------
    numpy.ndarray
        Standard normal variates.
    """
    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1.0 - eps)
    return stats.norm.ppf(u)
