"""Structured product assembler built from component pricers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from benchmarks.analytical import bsm_price
from simulation.gbm import GBMSimulator

from .barrier import BarrierOptionPricer

__all__ = ["StructuredProductPricer"]


class StructuredProductPricer:
    """Assemble simple structured products from vanilla and barrier legs.

    Each method returns a dictionary with the fair value, a breakdown of the
    component values, and payoff-diagram data suitable for plotting.

    Parameters
    ----------
    seed : int, optional
        Seed forwarded to the internal GBM simulator.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._gbm = GBMSimulator(seed=seed)
        self._barrier = BarrierOptionPricer()

    @staticmethod
    def _payoff_grid(S0: float, lo: float = 0.4, hi: float = 1.8, n: int = 121):
        return np.linspace(S0 * lo, S0 * hi, n)

    def price_capital_protected_note(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        participation_rate: float = 1.0,
        notional: float = 100.0,
        n_paths: int = 100_000,
    ) -> dict:
        """Price a capital-protected note.

        Value = zero-coupon bond (principal protection) + participation in a
        vanilla call.

        Returns
        -------
        dict
            Keys ``price``, ``components`` and ``payoff_diagram``.
        """
        bond = notional * np.exp(-r * T)
        call = bsm_price(S0, K, T, r, q, sigma, "call")
        # Number of call units per note: participation * notional / S0.
        units = participation_rate * notional / S0
        option_value = units * call
        price = bond + option_value

        ST = self._payoff_grid(S0)
        payoff = notional + units * np.maximum(ST - K, 0.0)
        return {
            "price": float(price),
            "components": {
                "zero_coupon_bond": float(bond),
                "call_participation": float(option_value),
            },
            "payoff_diagram": {"S_T": ST, "payoff": payoff},
        }

    def price_shark_fin(
        self,
        S0: float,
        K: float,
        barrier: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        rebate: float = 0.0,
        notional: float = 100.0,
        n_paths: int = 100_000,
    ) -> dict:
        """Price a shark-fin note (up-and-out call with rebate).

        Returns
        -------
        dict
            Keys ``price``, ``components`` and ``payoff_diagram``.
        """
        paths = self._gbm.generate_paths(
            S0, T, r, q, sigma, n_paths=n_paths, n_steps=252
        )
        uo = self._barrier.price_mc(
            paths, K, barrier, r, T, "up-and-out", "call", rebate=rebate
        )
        bond = notional * np.exp(-r * T)
        units = notional / S0
        price = bond + units * uo.price

        ST = self._payoff_grid(S0)
        capped = np.where(ST < barrier, np.maximum(ST - K, 0.0), rebate)
        payoff = notional + units * capped
        return {
            "price": float(price),
            "components": {
                "zero_coupon_bond": float(bond),
                "up_and_out_call": float(units * uo.price),
            },
            "mc_result": uo,
            "payoff_diagram": {"S_T": ST, "payoff": payoff},
        }

    def price_reverse_convertible(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        coupon: float = 0.08,
        notional: float = 100.0,
        n_paths: int = 100_000,
    ) -> dict:
        """Price a reverse convertible note.

        Value = bond + present-valued coupon - short put option.

        Returns
        -------
        dict
            Keys ``price``, ``components`` and ``payoff_diagram``.
        """
        bond = notional * np.exp(-r * T)
        coupon_pv = notional * coupon * np.exp(-r * T)
        put = bsm_price(S0, K, T, r, q, sigma, "put")
        units = notional / S0
        short_put = units * put
        price = bond + coupon_pv - short_put

        ST = self._payoff_grid(S0)
        payoff = notional + notional * coupon - units * np.maximum(K - ST, 0.0)
        return {
            "price": float(price),
            "components": {
                "zero_coupon_bond": float(bond),
                "coupon_pv": float(coupon_pv),
                "short_put": float(-short_put),
            },
            "payoff_diagram": {"S_T": ST, "payoff": payoff},
        }
