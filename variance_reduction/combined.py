"""Combined variance-reduction pipelines and benchmarking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from benchmarks.asian_geometric import kemna_vorst_geometric_asian
from simulation.base import MCResult, MonteCarloEngine
from simulation.gbm import GBMSimulator

from .antithetic import AntitheticVariates
from .control_variates import ControlVariates
from .quasi_mc import QuasiMonteCarlo
from .stratified import StratifiedSampling

__all__ = ["CombinedVR", "Contract"]


@dataclass
class Contract:
    """Lightweight description of an option contract for benchmarking.

    Attributes
    ----------
    S0, K, T, r, q, sigma : float
        Market and contract parameters.
    option_type : {"call", "put"}
    kind : str
        Either ``"asian_arithmetic"`` (default) or ``"european"``; selects the
        payoff used by the benchmarking helpers.
    n_steps : int
        Number of monitoring steps.
    """

    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.05
    q: float = 0.0
    sigma: float = 0.2
    option_type: str = "call"
    kind: str = "asian_arithmetic"
    n_steps: int = 252

    def payoff_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the undiscounted payoff function for this contract."""
        K, ot, kind = self.K, self.option_type, self.kind

        def _payoff(paths: np.ndarray) -> np.ndarray:
            if kind == "asian_arithmetic":
                underlier = paths[:, 1:].mean(axis=1)
            else:
                underlier = paths[:, -1]
            if ot == "call":
                return np.maximum(underlier - K, 0.0)
            return np.maximum(K - underlier, 0.0)

        return _payoff


class CombinedVR:
    """Stack multiple variance-reduction techniques and benchmark them.

    Parameters
    ----------
    seed : int, optional
        Seed propagated to every sub-estimator for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    # ------------------------------------------------------------------
    # Stacked estimators
    # ------------------------------------------------------------------
    def antithetic_plus_control(
        self,
        paths: np.ndarray,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        cv_payoff_fn: Callable[[np.ndarray], np.ndarray],
        cv_analytical: float,
        r: float,
        T: float,
    ) -> MCResult:
        """Combine antithetic pairing with a control variate.

        Antithetic partners are averaged first, then the control correction is
        applied to the paired payoffs.

        Returns
        -------
        MCResult
        """
        start = time.perf_counter()
        n = paths.shape[0] // 2
        disc = np.exp(-r * T)
        f = disc * np.asarray(payoff_fn(paths), dtype=float)
        c = disc * np.asarray(cv_payoff_fn(paths), dtype=float)
        f_pair = 0.5 * (f[:n] + f[n:])
        c_pair = 0.5 * (c[:n] + c[n:])
        var_c = np.var(c_pair, ddof=1)
        beta = np.cov(f_pair, c_pair, ddof=1)[0, 1] / var_c if var_c > 0 else 0.0
        adjusted = f_pair - beta * (c_pair - cv_analytical)
        res = MonteCarloEngine.summarize(
            adjusted, n_paths=paths.shape[0], method="antithetic+control"
        )
        crude_var = np.var(f, ddof=1) / paths.shape[0]
        comb_var = np.var(adjusted, ddof=1) / n
        res.extra = {
            "beta": float(beta),
            "vr_ratio": float(crude_var / comb_var) if comb_var > 0 else float("inf"),
        }
        res.elapsed_seconds = time.perf_counter() - start
        return res

    def qmc_plus_antithetic(
        self,
        S0: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        n_paths: int,
        n_steps: int,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
    ) -> MCResult:
        """Sobol QMC paths augmented with antithetic reflection.

        Returns
        -------
        MCResult
        """
        start = time.perf_counter()
        qmc = QuasiMonteCarlo(seed=self.seed)
        Z = qmc.generate_sobol(n_paths, n_steps)
        Z_full = np.concatenate([Z, -Z], axis=0)
        paths = qmc._gbm_from_normals(Z_full, S0, T, r, q, sigma, bridge=True)
        disc = np.exp(-r * T)
        payoffs = disc * np.asarray(payoff_fn(paths), dtype=float)
        n = n_paths
        paired = 0.5 * (payoffs[:n] + payoffs[n:])
        res = MonteCarloEngine.summarize(paired, n_paths=2 * n_paths, method="qmc+antithetic")
        res.elapsed_seconds = time.perf_counter() - start
        return res

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------
    def efficiency_table(
        self, contract: Contract = None, n_paths: int = 50_000
    ) -> pd.DataFrame:
        """Benchmark every technique on a single contract.

        Parameters
        ----------
        contract : Contract, optional
            Defaults to an at-the-money arithmetic Asian call.
        n_paths : int, default 50_000
            Base number of paths per method.

        Returns
        -------
        pandas.DataFrame
            Columns ``Method, Price, StdErr, CI_Width, VR_Ratio,
            Computation_Time``.
        """
        c = contract or Contract()
        S0, K, T, r, q, sigma = c.S0, c.K, c.T, c.r, c.q, c.sigma
        payoff = c.payoff_fn()
        disc = np.exp(-r * T)
        rows: List[dict] = []

        # Reference crude MC for VR ratios.
        gbm = GBMSimulator(seed=self.seed)
        t0 = time.perf_counter()
        crude_paths = gbm.generate_paths(S0, T, r, q, sigma, n_paths, c.n_steps)
        crude_payoffs = disc * payoff(crude_paths)
        crude = MonteCarloEngine.summarize(crude_payoffs, n_paths=n_paths, method="Crude MC")
        crude.elapsed_seconds = time.perf_counter() - t0
        crude_var = np.var(crude_payoffs, ddof=1)
        rows.append(self._row(crude, crude_var, crude_var))

        # Antithetic.
        t0 = time.perf_counter()
        anti_paths = gbm.generate_paths_antithetic(S0, T, r, q, sigma, n_paths, c.n_steps)
        anti = AntitheticVariates().price(anti_paths, payoff, r, T)
        anti.method = "Antithetic"
        rows.append(self._row(anti, crude_var, anti.extra["antithetic_variance"] * (2 * n_paths)))

        # Control variate (geometric Asian) — only meaningful for Asian.
        t0 = time.perf_counter()
        if c.kind == "asian_arithmetic":
            cv = ControlVariates().geometric_asian_control(
                crude_paths, S0, K, T, r, q, sigma, c.option_type
            )
            cv.method = "Control Variate"
            cv_var = crude_var / cv.extra["vr_ratio"] if cv.extra["vr_ratio"] > 0 else crude_var
            rows.append(self._row(cv, crude_var, cv_var))

        # Stratified path sampling via Latin hypercube over the time steps.
        t0 = time.perf_counter()
        strat_sampler = StratifiedSampling(seed=self.seed)
        Z_lhs = strat_sampler.latin_hypercube_normals(n_paths, c.n_steps)
        strat_paths = gbm._paths_from_normals(Z_lhs, S0, T, r, q, sigma)
        strat_payoffs = disc * payoff(strat_paths)
        strat = MonteCarloEngine.summarize(strat_payoffs, n_paths=n_paths, method="Stratified")
        strat.elapsed_seconds = time.perf_counter() - t0
        rows.append(self._row(strat, crude_var, np.var(strat_payoffs, ddof=1)))

        # Sobol RQMC (variance measured across scrambled replications).
        qmc = QuasiMonteCarlo(seed=self.seed)
        sob = qmc.price_rqmc(
            payoff, S0, T, r, q, sigma, n_paths // 8, c.n_steps,
            n_replications=8, sequence="sobol", method="Sobol QMC",
        )
        sob_method_var = (sob.stderr**2) * (n_paths // 8) * 8
        rows.append(self._row(sob, crude_var, sob_method_var))

        # Halton RQMC.
        hal = qmc.price_rqmc(
            payoff, S0, T, r, q, sigma, n_paths // 8, c.n_steps,
            n_replications=8, sequence="halton", method="Halton QMC",
        )
        hal_method_var = (hal.stderr**2) * (n_paths // 8) * 8
        rows.append(self._row(hal, crude_var, hal_method_var))

        # Antithetic + control variate.
        if c.kind == "asian_arithmetic":
            geo_analytic = kemna_vorst_geometric_asian(
                S0, K, T, r, q, sigma, c.n_steps, c.option_type
            )

            def cv_payoff(paths):
                geo_avg = np.exp(np.log(paths[:, 1:]).mean(axis=1))
                if c.option_type == "call":
                    return np.maximum(geo_avg - K, 0.0)
                return np.maximum(K - geo_avg, 0.0)

            combo = self.antithetic_plus_control(
                anti_paths, payoff, cv_payoff, geo_analytic, r, T
            )
            combo.method = "Antithetic+CV"
            combo_var = crude_var / combo.extra["vr_ratio"] if combo.extra["vr_ratio"] > 0 else crude_var
            rows.append(self._row(combo, crude_var, combo_var))

        return pd.DataFrame(rows)

    @staticmethod
    def _row(res: MCResult, crude_var: float, method_var: float) -> dict:
        vr = crude_var / method_var if method_var > 0 else float("inf")
        return {
            "Method": res.method,
            "Price": round(res.price, 6),
            "StdErr": round(res.stderr, 6),
            "CI_Width": round(res.ci_width, 6),
            "VR_Ratio": round(vr, 2),
            "Computation_Time": round(res.elapsed_seconds, 4),
        }
