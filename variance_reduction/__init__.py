"""Variance-reduction techniques for Monte Carlo option pricing."""

from .antithetic import AntitheticVariates
from .combined import CombinedVR, Contract
from .control_variates import ControlVariates
from .importance_sampling import ImportanceSampling
from .quasi_mc import QuasiMonteCarlo
from .stratified import StratifiedSampling

__all__ = [
    "AntitheticVariates",
    "ControlVariates",
    "StratifiedSampling",
    "ImportanceSampling",
    "QuasiMonteCarlo",
    "CombinedVR",
    "Contract",
]
