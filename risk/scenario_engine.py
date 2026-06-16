"""Stress-scenario engine for exotic positions."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np

__all__ = ["ExoticScenarioEngine"]


class ExoticScenarioEngine:
    """Apply pre-built and custom stress scenarios to an exotic position.

    Parameters
    ----------
    value_fn : callable
        ``value_fn(S0, sigma, r, T) -> float`` returns the present value of
        the position under the given market state.
    base : dict
        Base-case market state with keys ``S0``, ``sigma``, ``r`` and ``T``.
    """

    #: Built-in scenario transforms (applied to the base state).
    SCENARIOS: Dict[str, Callable[[dict], dict]] = {
        "Spot Crash -30%": lambda s: {**s, "S0": s["S0"] * 0.70},
        "Spot Rally +20%": lambda s: {**s, "S0": s["S0"] * 1.20},
        "Vol Spike +20pp": lambda s: {**s, "sigma": s["sigma"] + 0.20},
        "Vol Crush -10pp": lambda s: {**s, "sigma": max(s["sigma"] - 0.10, 1e-4)},
        "Time Decay 30d": lambda s: {**s, "T": max(s["T"] - 30 / 365, 1e-6)},
        "Combined Stress": lambda s: {
            **s,
            "S0": s["S0"] * 0.80,
            "sigma": s["sigma"] + 0.15,
            "r": s["r"] + 0.01,
        },
    }

    def __init__(self, value_fn: Callable[..., float], base: dict) -> None:
        self.value_fn = value_fn
        self.base = dict(base)

    def _value(self, state: dict) -> float:
        return float(
            self.value_fn(
                S0=state["S0"], sigma=state["sigma"], r=state["r"], T=state["T"]
            )
        )

    def run_scenario(self, scenario_name: str) -> dict:
        """Evaluate a single named scenario.

        Returns
        -------
        dict
            Keys ``original_price``, ``stressed_price``, ``pnl`` and
            ``pct_change``.
        """
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario_name!r}")
        original = self._value(self.base)
        stressed_state = self.SCENARIOS[scenario_name](self.base)
        stressed = self._value(stressed_state)
        pnl = stressed - original
        return {
            "scenario": scenario_name,
            "original_price": original,
            "stressed_price": stressed,
            "pnl": pnl,
            "pct_change": (pnl / original * 100.0) if original != 0 else float("nan"),
        }

    def run_all(self) -> list:
        """Run every built-in scenario.

        Returns
        -------
        list of dict
        """
        return [self.run_scenario(name) for name in self.SCENARIOS]

    def spot_vol_grid(
        self,
        S_shocks_pct: Sequence[float],
        vol_shocks_abs: Sequence[float],
    ) -> dict:
        """P&L grid across spot and volatility shocks for a heatmap.

        Parameters
        ----------
        S_shocks_pct : sequence of float
            Relative spot shocks (e.g. ``-0.2`` for ``-20%``).
        vol_shocks_abs : sequence of float
            Absolute volatility shocks in vol points (e.g. ``0.05``).

        Returns
        -------
        dict
            Keys ``S_shocks``, ``vol_shocks`` and ``pnl_grid`` (2-D array,
            rows index vol shocks, columns index spot shocks).
        """
        base_value = self._value(self.base)
        grid = np.empty((len(vol_shocks_abs), len(S_shocks_pct)))
        for i, dv in enumerate(vol_shocks_abs):
            for j, ds in enumerate(S_shocks_pct):
                state = {
                    **self.base,
                    "S0": self.base["S0"] * (1 + ds),
                    "sigma": max(self.base["sigma"] + dv, 1e-4),
                }
                grid[i, j] = self._value(state) - base_value
        return {
            "S_shocks": np.asarray(S_shocks_pct),
            "vol_shocks": np.asarray(vol_shocks_abs),
            "pnl_grid": grid,
        }
