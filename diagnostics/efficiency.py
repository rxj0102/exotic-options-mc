"""Variance-reduction efficiency metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["EfficiencyAnalyzer"]


class EfficiencyAnalyzer:
    """Compute work-normalised variance-reduction efficiency metrics.

    Three complementary measures are provided:

    * **Variance reduction ratio** ``VRR = Var(crude) / Var(method)``.
    * **Effective sample size** ``ESS = n * VRR`` — equivalent crude paths.
    * **Work-normalised efficiency** ``VRR / (time_method / time_crude)`` —
      penalises techniques that buy variance reduction with extra compute.
    """

    @staticmethod
    def vrr(crude_var: float, method_var: float) -> float:
        """Variance-reduction ratio."""
        if method_var <= 0:
            return float("inf")
        return float(crude_var / method_var)

    @staticmethod
    def effective_sample_size(n: int, vrr: float) -> float:
        """Effective crude-MC sample size for a given VR ratio."""
        return float(n * vrr)

    @staticmethod
    def work_normalised_efficiency(
        vrr: float, time_method: float, time_crude: float
    ) -> float:
        """Efficiency adjusted for relative computational cost."""
        if time_crude <= 0 or time_method <= 0:
            return float(vrr)
        return float(vrr / (time_method / time_crude))

    def full_report(
        self,
        crude_var: float,
        method_var: float,
        n_paths: int,
        time_crude: float,
        time_method: float,
        method_name: str = "method",
    ) -> Dict[str, float]:
        """Assemble all efficiency metrics into a report dictionary.

        Returns
        -------
        dict
        """
        ratio = self.vrr(crude_var, method_var)
        return {
            "method": method_name,
            "vr_ratio": ratio,
            "effective_sample_size": self.effective_sample_size(n_paths, ratio),
            "work_normalised_efficiency": self.work_normalised_efficiency(
                ratio, time_method, time_crude
            ),
            "stderr_reduction": float(np.sqrt(ratio)),
        }

    def plot_data(self, reports: Dict[str, dict]) -> dict:
        """Structure several reports for a bar / radar chart.

        Parameters
        ----------
        reports : dict
            Mapping from method name to its :meth:`full_report` output.

        Returns
        -------
        dict
            Keys ``methods``, ``vr_ratios`` and ``efficiencies``.
        """
        methods = list(reports.keys())
        return {
            "methods": methods,
            "vr_ratios": [reports[m]["vr_ratio"] for m in methods],
            "efficiencies": [reports[m]["work_normalised_efficiency"] for m in methods],
        }
