"""Confidence-interval construction for Monte Carlo estimates."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["CIEngine"]

_Z95 = 1.959963984540054


class CIEngine:
    """Construct and compare confidence intervals for MC payoff samples.

    Parameters
    ----------
    seed : int, optional
        Seed for the bootstrap resampler.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def standard_ci(self, payoffs: np.ndarray, confidence: float = 0.95):
        """Normal-approximation confidence interval.

        Returns
        -------
        (float, float)
            Lower and upper bounds.
        """
        x = np.asarray(payoffs, dtype=float)
        mean = x.mean()
        stderr = x.std(ddof=1) / np.sqrt(x.size)
        z = _z_for(confidence)
        return float(mean - z * stderr), float(mean + z * stderr)

    def bootstrap_ci(
        self,
        payoffs: np.ndarray,
        confidence: float = 0.95,
        n_boot: int = 1000,
    ):
        """Percentile bootstrap confidence interval.

        Parameters
        ----------
        n_boot : int, default 1000
            Number of bootstrap resamples.

        Returns
        -------
        (float, float)
        """
        x = np.asarray(payoffs, dtype=float)
        n = x.size
        idx = self._rng.integers(0, n, size=(n_boot, n))
        boot_means = x[idx].mean(axis=1)
        alpha = 1.0 - confidence
        lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)

    def batch_means_ci(
        self,
        payoffs: np.ndarray,
        confidence: float = 0.95,
        n_batches: int = 30,
    ):
        """Batch-means confidence interval.

        Splits the sample into ``n_batches`` contiguous batches and forms the
        interval from the variance of the batch means.

        Returns
        -------
        (float, float)
        """
        x = np.asarray(payoffs, dtype=float)
        n = x.size
        batch_size = n // n_batches
        trimmed = x[: batch_size * n_batches].reshape(n_batches, batch_size)
        batch_means = trimmed.mean(axis=1)
        grand = batch_means.mean()
        se = batch_means.std(ddof=1) / np.sqrt(n_batches)
        z = _z_for(confidence)
        return float(grand - z * se), float(grand + z * se)

    def ci_comparison(
        self,
        payoffs: np.ndarray,
        method_list: Optional[Sequence[str]] = None,
        confidence: float = 0.95,
    ) -> pd.DataFrame:
        """Tabulate intervals from several construction methods.

        Returns
        -------
        pandas.DataFrame
            Columns ``Method, CI_Lower, CI_Upper, CI_Width``.
        """
        methods = method_list or ["standard", "bootstrap", "batch_means"]
        funcs = {
            "standard": self.standard_ci,
            "bootstrap": self.bootstrap_ci,
            "batch_means": self.batch_means_ci,
        }
        rows = []
        for m in methods:
            lo, hi = funcs[m](payoffs, confidence)
            rows.append(
                {"Method": m, "CI_Lower": lo, "CI_Upper": hi, "CI_Width": hi - lo}
            )
        return pd.DataFrame(rows)

    def coverage_test(
        self,
        sampler,
        true_value: float,
        n_experiments: int = 200,
        confidence: float = 0.95,
    ) -> float:
        """Empirical coverage of the standard CI over repeated experiments.

        Parameters
        ----------
        sampler : callable
            Zero-argument callable returning a fresh payoff array.
        true_value : float
            The known expectation.

        Returns
        -------
        float
            Fraction of experiments whose CI contained ``true_value`` (should
            be close to ``confidence``).
        """
        hits = 0
        for _ in range(n_experiments):
            lo, hi = self.standard_ci(sampler(), confidence)
            if lo <= true_value <= hi:
                hits += 1
        return hits / n_experiments


def _z_for(confidence: float) -> float:
    """Return the two-sided normal quantile for ``confidence``."""
    from scipy.stats import norm

    if abs(confidence - 0.95) < 1e-9:
        return _Z95
    return float(norm.ppf(0.5 + confidence / 2))
