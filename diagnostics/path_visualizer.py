"""Path visualisation data builders."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import stats

__all__ = ["PathVisualizer"]


class PathVisualizer:
    """Produce structured data for path fan charts, histograms and statistics.

    The methods return plain dictionaries / arrays so they can be consumed by
    any plotting backend (the Streamlit app uses Plotly).
    """

    def fan_chart_data(
        self,
        paths: np.ndarray,
        percentiles: Sequence[int] = (5, 25, 50, 75, 95),
    ) -> dict:
        """Percentile bands of the path ensemble over time.

        Returns
        -------
        dict
            Keys ``time_index`` and one ``p{level}`` series per percentile.
        """
        out = {"time_index": np.arange(paths.shape[1])}
        for p in percentiles:
            out[f"p{p}"] = np.percentile(paths, p, axis=0)
        return out

    def path_sample_data(self, paths: np.ndarray, n_display: int = 50) -> np.ndarray:
        """Return a random subsample of paths for overlay plotting.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(min(n_display, n_paths), n_steps + 1)``.
        """
        n = min(n_display, paths.shape[0])
        idx = np.linspace(0, paths.shape[0] - 1, n).astype(int)
        return paths[idx]

    def terminal_histogram_data(self, paths: np.ndarray, n_bins: int = 60) -> dict:
        """Histogram of terminal values with a fitted log-normal overlay.

        Returns
        -------
        dict
            Keys ``bin_centers``, ``density``, ``fit_x`` and ``fit_y``.
        """
        S_T = paths[:, -1]
        counts, edges = np.histogram(S_T, bins=n_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        shape, loc, scale = stats.lognorm.fit(S_T, floc=0)
        fit_x = np.linspace(edges[0], edges[-1], 400)
        fit_y = stats.lognorm.pdf(fit_x, shape, loc=loc, scale=scale)
        return {
            "bin_centers": centers,
            "density": counts,
            "fit_x": fit_x,
            "fit_y": fit_y,
        }

    def path_statistics(self, paths: np.ndarray) -> dict:
        """Summary statistics of the terminal distribution.

        Returns
        -------
        dict
            Keys ``mean_terminal``, ``std_terminal``, ``skewness``,
            ``kurtosis`` (excess), ``min`` and ``max``.
        """
        S_T = paths[:, -1]
        return {
            "mean_terminal": float(np.mean(S_T)),
            "std_terminal": float(np.std(S_T, ddof=1)),
            "skewness": float(stats.skew(S_T)),
            "kurtosis": float(stats.kurtosis(S_T)),  # excess kurtosis
            "min": float(np.min(S_T)),
            "max": float(np.max(S_T)),
        }

    def barrier_breach_visualization(
        self, paths: np.ndarray, H: float, barrier_type: str
    ) -> dict:
        """Split paths into breaching and surviving sets for highlighting.

        Returns
        -------
        dict
            Keys ``breached`` and ``survived`` (arrays of paths) plus
            ``breach_fraction``.
        """
        if "up" in barrier_type.lower():
            breached_mask = paths.max(axis=1) >= H
        else:
            breached_mask = paths.min(axis=1) <= H
        return {
            "breached": paths[breached_mask],
            "survived": paths[~breached_mask],
            "breach_fraction": float(breached_mask.mean()),
            "barrier": H,
        }
