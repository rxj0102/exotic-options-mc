"""Monte Carlo convergence analysis."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["ConvergenceAnalyzer"]

_DEFAULT_GRID = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]


class ConvergenceAnalyzer:
    """Study how a Monte Carlo estimate converges with the path count.

    The pricer callable must accept an integer ``n_paths`` and return an
    object exposing ``price`` and ``stderr`` attributes (e.g. an
    :class:`~simulation.base.MCResult`).
    """

    def price_vs_paths(
        self,
        pricer: Callable[[int], object],
        n_paths_list: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        """Run the pricer across a grid of path counts.

        Parameters
        ----------
        pricer : callable
            ``pricer(n_paths) -> MCResult``.
        n_paths_list : sequence of int, optional
            Path counts; defaults to ``[1e3, 5e3, 1e4, 5e4, 1e5, 5e5]``.

        Returns
        -------
        pandas.DataFrame
            Columns ``n_paths, price, stderr, ci_lower, ci_upper``.
        """
        grid = [int(n) for n in (n_paths_list or _DEFAULT_GRID)]
        rows = []
        for n in grid:
            res = pricer(n)
            rows.append(
                {
                    "n_paths": n,
                    "price": res.price,
                    "stderr": res.stderr,
                    "ci_lower": res.ci_lower,
                    "ci_upper": res.ci_upper,
                }
            )
        return pd.DataFrame(rows)

    def convergence_rate(self, df: pd.DataFrame) -> float:
        """Fit ``log(stderr) ~ slope * log(n) + c`` and return the slope.

        Returns
        -------
        float
            The fitted slope; Monte Carlo theory predicts ``-0.5``.
        """
        n = np.log(df["n_paths"].to_numpy(dtype=float))
        se = df["stderr"].to_numpy(dtype=float)
        mask = se > 0
        slope, _ = np.polyfit(n[mask], np.log(se[mask]), 1)
        return float(slope)

    def rmse_vs_analytical(
        self,
        pricer: Callable[[int], object],
        analytical_price: float,
        n_paths_list: Optional[Sequence[int]] = None,
        n_trials: int = 20,
    ) -> pd.DataFrame:
        """Estimate RMSE versus an analytical price at each path count.

        Parameters
        ----------
        analytical_price : float
            Ground-truth value.
        n_trials : int, default 20
            Independent repetitions used to estimate the RMSE.

        Returns
        -------
        pandas.DataFrame
            Columns ``n_paths`` and ``rmse``.
        """
        grid = [int(n) for n in (n_paths_list or _DEFAULT_GRID)]
        rows = []
        for n in grid:
            errs = np.array([pricer(n).price - analytical_price for _ in range(n_trials)])
            rows.append({"n_paths": n, "rmse": float(np.sqrt(np.mean(errs**2)))})
        return pd.DataFrame(rows)

    def path_count_for_precision(
        self,
        pricer: Callable[[int], object],
        target_stderr: float = 0.01,
        pilot_paths: int = 10_000,
    ) -> int:
        """Estimate the path count needed to reach a target standard error.

        Uses the :math:`O(N^{-1/2})` scaling: from a pilot run with standard
        error :math:`s_0` at :math:`N_0` paths, the required count is
        :math:`N_0 (s_0 / s^\\star)^2`.

        Returns
        -------
        int
        """
        pilot = pricer(pilot_paths)
        if pilot.stderr <= 0:
            return pilot_paths
        needed = pilot_paths * (pilot.stderr / target_stderr) ** 2
        return int(np.ceil(needed))
