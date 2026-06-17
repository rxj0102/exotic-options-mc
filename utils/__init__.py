"""Shared utilities: RNG management, statistics and plotting."""

from .random import antithetic_normals, make_rng, spawn_rngs
from .statistics import (confidence_interval, running_mean, running_stderr,
                         standard_error, variance_reduction_ratio)

__all__ = [
    "make_rng",
    "spawn_rngs",
    "antithetic_normals",
    "standard_error",
    "confidence_interval",
    "variance_reduction_ratio",
    "running_mean",
    "running_stderr",
]
