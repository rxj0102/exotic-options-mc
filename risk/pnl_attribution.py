"""Greek-based P&L attribution for exotic books."""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["PnLAttribution"]


class PnLAttribution:
    """Decompose realised P&L into Greek contributions (Taylor expansion).

    The second-order attribution is

    .. math::

        \\Delta V \\approx \\delta\\, \\Delta S + \\tfrac12 \\Gamma\\,
        \\Delta S^2 + \\nu\\, \\Delta \\sigma + \\Theta\\, \\Delta t
        + \\rho\\, \\Delta r,

    with the residual capturing higher-order and cross terms.
    """

    def attribute(
        self,
        greeks: Dict[str, float],
        dS: float = 0.0,
        dSigma: float = 0.0,
        dt: float = 0.0,
        dr: float = 0.0,
        actual_pnl: float = None,
    ) -> dict:
        """Attribute P&L to first- and second-order Greek terms.

        Parameters
        ----------
        greeks : dict
            Keys ``delta``, ``gamma``, ``vega``, ``theta`` and ``rho``. Vega
            and rho are expected per unit (per 100%); pass ``dSigma`` / ``dr``
            in the same units.
        dS, dSigma, dt, dr : float
            Realised moves in spot, volatility, time and rate.
        actual_pnl : float, optional
            Observed P&L; when supplied the unexplained residual is reported.

        Returns
        -------
        dict
            Per-effect contributions, ``total_explained`` and (if available)
            ``residual``.
        """
        delta_pnl = greeks.get("delta", 0.0) * dS
        gamma_pnl = 0.5 * greeks.get("gamma", 0.0) * dS**2
        vega_pnl = greeks.get("vega", 0.0) * dSigma
        theta_pnl = greeks.get("theta", 0.0) * dt
        rho_pnl = greeks.get("rho", 0.0) * dr
        explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl

        out = {
            "delta_pnl": float(delta_pnl),
            "gamma_pnl": float(gamma_pnl),
            "vega_pnl": float(vega_pnl),
            "theta_pnl": float(theta_pnl),
            "rho_pnl": float(rho_pnl),
            "total_explained": float(explained),
        }
        if actual_pnl is not None:
            out["actual_pnl"] = float(actual_pnl)
            out["residual"] = float(actual_pnl - explained)
        return out

    def waterfall_data(self, attribution: dict) -> dict:
        """Reshape an attribution dict into waterfall-chart data.

        Returns
        -------
        dict
            Keys ``labels`` and ``values`` ready for a waterfall plot.
        """
        keys = ["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl", "rho_pnl"]
        labels = [k.replace("_pnl", "").capitalize() for k in keys]
        values = [attribution.get(k, 0.0) for k in keys]
        if "residual" in attribution:
            labels.append("Residual")
            values.append(attribution["residual"])
        return {"labels": labels, "values": values}
