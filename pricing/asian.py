"""Asian (average-price / average-strike) option pricing."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from benchmarks.asian_geometric import kemna_vorst_geometric_asian
from simulation.base import MCResult, MonteCarloEngine

__all__ = ["AsianOptionPricer"]

_FREQ_STEPS = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
    "continuous": None,  # use every simulated step
}


class AsianOptionPricer:
    """Monte Carlo pricer for arithmetic and geometric Asian options.

    Arithmetic Asian options have no closed form and are priced by simulation;
    geometric Asian options admit the Kemna-Vorst formula, which is also used
    as a control variate to slash the variance of the arithmetic estimate.

    Parameters
    ----------
    avg_frequency : {"daily", "weekly", "monthly", "continuous"}
        Frequency at which the running average is sampled. ``"continuous"``
        uses every simulated time step.
    """

    def __init__(self, avg_frequency: str = "continuous") -> None:
        if avg_frequency not in _FREQ_STEPS:
            raise ValueError(f"avg_frequency must be one of {sorted(_FREQ_STEPS)}")
        self.avg_frequency = avg_frequency

    # ------------------------------------------------------------------
    # Averaging helpers
    # ------------------------------------------------------------------
    def _averaging_columns(self, n_steps: int) -> np.ndarray:
        """Return the column indices (excluding ``S0``) used for averaging."""
        target = _FREQ_STEPS[self.avg_frequency]
        if target is None or target >= n_steps:
            return np.arange(1, n_steps + 1)
        # Evenly spaced observation columns over (0, T].
        idx = np.linspace(1, n_steps, target).round().astype(int)
        return np.unique(idx)

    def _arithmetic_average(self, paths: np.ndarray) -> np.ndarray:
        cols = self._averaging_columns(paths.shape[1] - 1)
        return paths[:, cols].mean(axis=1)

    def _geometric_average(self, paths: np.ndarray) -> np.ndarray:
        cols = self._averaging_columns(paths.shape[1] - 1)
        return np.exp(np.log(paths[:, cols]).mean(axis=1))

    @staticmethod
    def _payoff(avg: np.ndarray, S_T: np.ndarray, K, option_type, floating):
        option_type = option_type.lower()
        if floating:
            # Floating strike: strike is the realised average.
            if option_type == "call":
                return np.maximum(S_T - avg, 0.0)
            return np.maximum(avg - S_T, 0.0)
        if option_type == "call":
            return np.maximum(avg - K, 0.0)
        return np.maximum(K - avg, 0.0)

    # ------------------------------------------------------------------
    # Pricing routines
    # ------------------------------------------------------------------
    def price_arithmetic(
        self,
        paths: np.ndarray,
        K: float,
        r: float,
        T: float,
        option_type: str = "call",
        floating: bool = False,
    ) -> MCResult:
        """Price an arithmetic-average Asian option by Monte Carlo.

        Parameters
        ----------
        paths : numpy.ndarray
            Simulated price paths, shape ``(n_paths, n_steps + 1)``.
        K : float
            Strike (ignored when ``floating`` is ``True``).
        r, T : float
            Discount rate and maturity.
        option_type : {"call", "put"}
        floating : bool, default False
            If ``True`` price a floating-strike (average-strike) option.

        Returns
        -------
        MCResult
        """
        start = time.perf_counter()
        avg = self._arithmetic_average(paths)
        payoff = self._payoff(avg, paths[:, -1], K, option_type, floating)
        disc = np.exp(-r * T) * payoff
        res = MonteCarloEngine.summarize(
            disc, n_paths=paths.shape[0], method="arithmetic_asian"
        )
        res.elapsed_seconds = time.perf_counter() - start
        return res

    def price_geometric(
        self,
        paths: np.ndarray,
        K: float,
        r: float,
        T: float,
        option_type: str = "call",
        floating: bool = False,
    ) -> MCResult:
        """Price a geometric-average Asian option by Monte Carlo."""
        start = time.perf_counter()
        avg = self._geometric_average(paths)
        payoff = self._payoff(avg, paths[:, -1], K, option_type, floating)
        disc = np.exp(-r * T) * payoff
        res = MonteCarloEngine.summarize(
            disc, n_paths=paths.shape[0], method="geometric_asian"
        )
        res.elapsed_seconds = time.perf_counter() - start
        return res

    def price_geometric_analytical(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str = "call",
        n_obs: Optional[int] = None,
    ) -> float:
        """Kemna-Vorst closed-form price of the geometric Asian option.

        Parameters
        ----------
        n_obs : int, optional
            Number of averaging dates. Defaults to the number implied by
            ``avg_frequency`` (``252`` for continuous).

        Returns
        -------
        float
        """
        if n_obs is None:
            n_obs = _FREQ_STEPS[self.avg_frequency] or 252
        return kemna_vorst_geometric_asian(S0, K, T, r, q, sigma, n_obs, option_type)

    def price_with_control_variate(
        self,
        paths: np.ndarray,
        K: float,
        r: float,
        T: float,
        q: float,
        sigma: float,
        S0: float,
        option_type: str = "call",
    ) -> MCResult:
        """Price the arithmetic Asian using the geometric Asian as control.

        The control-variate estimator is

        .. math::

            \\hat P_{cv} = \\bar X_{arith}
                + \\beta\\,(P^{geo}_{analytic} - \\bar X_{geo}),

        with :math:`\\beta = \\operatorname{Cov}(X_{arith}, X_{geo}) /
        \\operatorname{Var}(X_{geo})` estimated from the same sample.

        Returns
        -------
        MCResult
            The ``extra`` payload reports the variance-reduction ratio versus
            the crude arithmetic estimate and the fitted ``beta``.
        """
        start = time.perf_counter()
        disc = np.exp(-r * T)
        arith = disc * self._payoff(
            self._arithmetic_average(paths), paths[:, -1], K, option_type, False
        )
        geo = disc * self._payoff(
            self._geometric_average(paths), paths[:, -1], K, option_type, False
        )

        n_obs = _FREQ_STEPS[self.avg_frequency] or (paths.shape[1] - 1)
        geo_analytic = kemna_vorst_geometric_asian(
            S0, K, T, r, q, sigma, n_obs, option_type
        )

        var_geo = np.var(geo, ddof=1)
        beta = np.cov(arith, geo, ddof=1)[0, 1] / var_geo if var_geo > 0 else 0.0
        cv = arith + beta * (geo_analytic - geo)

        res = MonteCarloEngine.summarize(
            cv, n_paths=paths.shape[0], method="arithmetic_asian_cv"
        )
        crude_var = np.var(arith, ddof=1)
        cv_var = np.var(cv, ddof=1)
        res.extra = {
            "beta": float(beta),
            "vr_ratio": float(crude_var / cv_var) if cv_var > 0 else float("inf"),
            "geo_analytic": float(geo_analytic),
        }
        res.elapsed_seconds = time.perf_counter() - start
        return res
