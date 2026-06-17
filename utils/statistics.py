"""Monte Carlo statistics helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "standard_error",
    "confidence_interval",
    "variance_reduction_ratio",
    "running_mean",
    "running_stderr",
]

_Z95 = 1.959963984540054


def standard_error(x: np.ndarray) -> float:
    """Standard error of the mean of ``x``."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def confidence_interval(x: np.ndarray, z: float = _Z95) -> Tuple[float, float]:
    """Normal-approximation confidence interval for the mean.

    Returns
    -------
    (float, float)
    """
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    half = z * standard_error(x)
    return mean - half, mean + half


def variance_reduction_ratio(crude_var: float, method_var: float) -> float:
    """Variance-reduction ratio ``crude_var / method_var``."""
    if method_var <= 0:
        return float("inf")
    return float(crude_var / method_var)


def running_mean(x: np.ndarray) -> np.ndarray:
    """Cumulative running mean of ``x`` (useful for convergence plots)."""
    x = np.asarray(x, dtype=float)
    return np.cumsum(x) / np.arange(1, x.size + 1)


def running_stderr(x: np.ndarray) -> np.ndarray:
    """Cumulative running standard error of the mean of ``x``.

    Returns
    -------
    numpy.ndarray
        Running standard error; the first element is ``0`` by convention.
    """
    x = np.asarray(x, dtype=float)
    n = np.arange(1, x.size + 1)
    csum = np.cumsum(x)
    csum2 = np.cumsum(x**2)
    mean = csum / n
    # Sample variance via the cumulative sums; clip tiny negatives from
    # floating-point cancellation.
    var = np.clip((csum2 - n * mean**2) / np.maximum(n - 1, 1), 0.0, None)
    return np.sqrt(var / n)
