"""Random-number-generation utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["make_rng", "spawn_rngs", "antithetic_normals"]


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a seeded :class:`numpy.random.Generator`.

    Parameters
    ----------
    seed : int or None
        Seed; ``None`` draws fresh OS entropy.

    Returns
    -------
    numpy.random.Generator
    """
    return np.random.default_rng(seed)


def spawn_rngs(seed: Optional[int], n: int) -> list:
    """Spawn ``n`` independent child generators from a root seed.

    Independent streams are produced via :class:`numpy.random.SeedSequence`
    so parallel replications do not overlap.

    Returns
    -------
    list of numpy.random.Generator
    """
    ss = np.random.SeedSequence(seed)
    return [np.random.default_rng(child) for child in ss.spawn(n)]


def antithetic_normals(
    rng: np.random.Generator, n_paths: int, n_steps: int
) -> np.ndarray:
    """Draw antithetic standard normals.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2 * n_paths, n_steps)`` whose second half is the
        negation of the first.
    """
    Z = rng.standard_normal((n_paths, n_steps))
    return np.concatenate([Z, -Z], axis=0)
