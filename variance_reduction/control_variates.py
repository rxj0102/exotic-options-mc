"""Control variates."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from benchmarks.analytical import bsm_price
from benchmarks.asian_geometric import kemna_vorst_geometric_asian
from simulation.base import MCResult, MonteCarloEngine

__all__ = ["ControlVariates"]


class ControlVariates:
    """Control-variate variance reduction.

    The estimator subtracts a scaled, mean-zero control from the payoff:

    .. math::

        \\hat P_{cv} = \\bar X - \\beta\\,(\\bar C - \\mathbb{E}[C]),

    with the variance-minimising coefficient
    :math:`\\beta = \\operatorname{Cov}(X, C)/\\operatorname{Var}(C)`.
    """

    @staticmethod
    def estimate_beta_pilot(
        payoffs: np.ndarray, cv_payoffs: np.ndarray, pilot_fraction: float = 0.1
    ) -> float:
        """Estimate the optimal coefficient from a pilot sub-sample.

        Parameters
        ----------
        pilot_fraction : float, default 0.1
            Fraction of the sample used to estimate ``beta``. Using a disjoint
            pilot avoids the slight bias from reusing the full sample.

        Returns
        -------
        float
        """
        n = len(payoffs)
        k = max(int(pilot_fraction * n), 2)
        x, c = payoffs[:k], cv_payoffs[:k]
        var_c = np.var(c, ddof=1)
        if var_c <= 0:
            return 0.0
        return float(np.cov(x, c, ddof=1)[0, 1] / var_c)

    def apply(
        self,
        payoffs: np.ndarray,
        cv_payoffs: np.ndarray,
        cv_analytical: float,
        beta: float = None,
    ) -> Tuple[float, float, float]:
        """Apply the control-variate correction to discounted payoffs.

        Parameters
        ----------
        payoffs : numpy.ndarray
            Discounted target payoffs.
        cv_payoffs : numpy.ndarray
            Discounted control payoffs (same length).
        cv_analytical : float
            Known expectation of the (discounted) control.
        beta : float, optional
            Coefficient; estimated from the full sample when omitted.

        Returns
        -------
        (price, stderr, vr_ratio)
        """
        payoffs = np.asarray(payoffs, dtype=float)
        cv_payoffs = np.asarray(cv_payoffs, dtype=float)
        var_c = np.var(cv_payoffs, ddof=1)
        if beta is None:
            beta = (
                np.cov(payoffs, cv_payoffs, ddof=1)[0, 1] / var_c if var_c > 0 else 0.0
            )

        adjusted = payoffs - beta * (cv_payoffs - cv_analytical)
        price = float(np.mean(adjusted))
        stderr = float(np.std(adjusted, ddof=1) / np.sqrt(len(adjusted)))
        crude_var = np.var(payoffs, ddof=1)
        adj_var = np.var(adjusted, ddof=1)
        vr = float(crude_var / adj_var) if adj_var > 0 else float("inf")
        return price, stderr, vr

    def _result(self, payoffs, cv_payoffs, cv_analytical, method: str) -> MCResult:
        start = time.perf_counter()
        beta = np.cov(payoffs, cv_payoffs, ddof=1)[0, 1] / np.var(cv_payoffs, ddof=1)
        adjusted = payoffs - beta * (cv_payoffs - cv_analytical)
        res = MonteCarloEngine.summarize(adjusted, n_paths=len(payoffs), method=method)
        crude_var = np.var(payoffs, ddof=1)
        adj_var = np.var(adjusted, ddof=1)
        res.extra = {
            "beta": float(beta),
            "vr_ratio": float(crude_var / adj_var) if adj_var > 0 else float("inf"),
        }
        res.elapsed_seconds = time.perf_counter() - start
        return res

    def geometric_asian_control(
        self,
        paths: np.ndarray,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str = "call",
    ) -> MCResult:
        """Price an arithmetic Asian using the geometric Asian as control."""
        disc = np.exp(-r * T)
        arith_avg = paths[:, 1:].mean(axis=1)
        geo_avg = np.exp(np.log(paths[:, 1:]).mean(axis=1))
        if option_type == "call":
            arith = disc * np.maximum(arith_avg - K, 0.0)
            geo = disc * np.maximum(geo_avg - K, 0.0)
        else:
            arith = disc * np.maximum(K - arith_avg, 0.0)
            geo = disc * np.maximum(K - geo_avg, 0.0)
        geo_analytic = kemna_vorst_geometric_asian(
            S0, K, T, r, q, sigma, paths.shape[1] - 1, option_type
        )
        return self._result(arith, geo, geo_analytic, "control_variate_geo_asian")

    def vanilla_bsm_control(
        self,
        paths: np.ndarray,
        payoffs: np.ndarray,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str = "call",
    ) -> MCResult:
        """Use the terminal European payoff (priced by BSM) as a control.

        Parameters
        ----------
        payoffs : numpy.ndarray
            Discounted target payoffs (e.g. of a barrier or lookback option).

        Returns
        -------
        MCResult
        """
        disc = np.exp(-r * T)
        S_T = paths[:, -1]
        if option_type == "call":
            cv = disc * np.maximum(S_T - K, 0.0)
        else:
            cv = disc * np.maximum(K - S_T, 0.0)
        cv_analytic = bsm_price(S0, K, T, r, q, sigma, option_type)
        return self._result(
            np.asarray(payoffs, dtype=float), cv, cv_analytic, "control_variate_bsm"
        )

    def delta_control(
        self,
        paths: np.ndarray,
        payoffs: np.ndarray,
        S0: float,
        r: float,
        T: float,
    ) -> MCResult:
        """Use a zero-mean delta control ``S_T - S0 e^{rT}``.

        Under the risk-neutral measure :math:`\\mathbb{E}[S_T] = S_0 e^{rT}`,
        so the discounted terminal price net of its forward is a mean-zero
        control correlated with directional payoffs.

        Returns
        -------
        MCResult
        """
        S_T = paths[:, -1]
        cv = S_T - S0 * np.exp(r * T)  # mean zero under Q
        return self._result(
            np.asarray(payoffs, dtype=float), cv, 0.0, "control_variate_delta"
        )
