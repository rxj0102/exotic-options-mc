"""Generate benchmark results consumed by the paper and README.

Running this module writes ``results/benchmark_results.json`` with actual
Monte Carlo prices, analytical benchmarks and variance-reduction efficiency
figures produced by the library.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from benchmarks.analytical import bsm_price
from pricing.asian import AsianOptionPricer
from pricing.barrier import BarrierOptionPricer
from pricing.lookback import LookbackOptionPricer
from simulation.gbm import GBMSimulator
from variance_reduction.combined import CombinedVR, Contract

warnings.filterwarnings("ignore")

RESULTS_PATH = (
    Path(__file__).resolve().parent.parent / "results" / "benchmark_results.json"
)


def generate(seed: int = 2024, n_paths: int = 100_000, n_steps: int = 252) -> dict:
    """Run the validation and efficiency suites and return a results dict."""
    S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
    g = GBMSimulator(seed=seed)
    paths = g.generate_paths(S0, T, r, q, sigma, n_paths, n_steps)

    # --- Validation: MC vs analytical ---------------------------------
    ap = AsianOptionPricer("continuous")
    geo_mc = ap.price_geometric(paths, K, r, T, "call")
    geo_an = ap.price_geometric_analytical(S0, K, T, r, q, sigma, "call")

    bp = BarrierOptionPricer()
    uo_mc = bp.price_mc(
        paths, K, 120.0, r, T, "up-and-out", "call", sigma=sigma, brownian_bridge=True
    )
    uo_an = bp.price_analytical(S0, K, 120.0, T, r, q, sigma, "up-and-out", "call")

    lb = LookbackOptionPricer()
    lc_mc = lb.price_floating_strike(
        paths, r, T, "call", sigma=sigma, continuity_correction=True
    )
    lc_an = lb.floating_call_analytical(S0, T, r, q, sigma)

    validation = [
        _val_row("Geometric Asian call", geo_mc, geo_an),
        _val_row("Up-and-out barrier call", uo_mc, uo_an),
        _val_row("Floating lookback call", lc_mc, lc_an),
    ]

    # --- Variance-reduction efficiency table --------------------------
    contract = Contract(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma)
    table = CombinedVR(seed=seed).efficiency_table(contract, n_paths=50_000)

    return {
        "market": {"S0": S0, "K": K, "T": T, "r": r, "q": q, "sigma": sigma},
        "n_paths": n_paths,
        "n_steps": n_steps,
        "validation": validation,
        "efficiency_table": table.to_dict(orient="records"),
        "bsm_call": bsm_price(S0, K, T, r, q, sigma, "call"),
    }


def _val_row(name: str, mc, analytic: float) -> dict:
    dev_bps = (mc.price - analytic) / analytic * 1e4 if analytic else 0.0
    z = (mc.price - analytic) / mc.stderr if mc.stderr > 0 else 0.0
    return {
        "instrument": name,
        "mc_price": round(mc.price, 4),
        "stderr": round(mc.stderr, 4),
        "analytical": round(analytic, 4),
        "deviation_bps": round(dev_bps, 1),
        "z_score": round(z, 2),
    }


def main() -> None:
    """Generate results and write them to ``results/benchmark_results.json``."""
    results = generate()
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
