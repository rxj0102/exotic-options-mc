"""Unified path generation interface across simulation models."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .base import MonteCarloEngine
from .gbm import GBMSimulator
from .heston import HestonSimulator
from .jump_diffusion import MertonJumpDiffusion

__all__ = ["PathManager"]

_MODELS = {
    "gbm": GBMSimulator,
    "heston": HestonSimulator,
    "jump_diffusion": MertonJumpDiffusion,
    "merton": MertonJumpDiffusion,
}


class PathManager:
    """Factory and dispatcher for the supported simulation models.

    Parameters
    ----------
    model : {"gbm", "heston", "jump_diffusion"}
        Name of the model to instantiate.
    seed : int, optional
        Seed forwarded to the underlying engine.
    n_paths, n_steps : int
        Default simulation dimensions.

    Examples
    --------
    >>> pm = PathManager("gbm", seed=0)
    >>> paths = pm.generate(S0=100, T=1.0, r=0.05, q=0.0, sigma=0.2,
    ...                      n_paths=8, n_steps=4)
    >>> paths.shape
    (8, 5)
    """

    def __init__(
        self,
        model: str = "gbm",
        seed: Optional[int] = None,
        n_paths: int = 100_000,
        n_steps: int = 252,
    ) -> None:
        key = model.lower()
        if key not in _MODELS:
            raise ValueError(
                f"Unknown model '{model}'. Choose from {sorted(_MODELS)}."
            )
        self.model = key
        self.engine: MonteCarloEngine = _MODELS[key](
            seed=seed, n_paths=n_paths, n_steps=n_steps
        )

    def set_seed(self, seed: Optional[int]) -> None:
        """Reseed the underlying engine."""
        self.engine.set_seed(seed)

    def generate(
        self, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Dispatch to the underlying engine's ``generate_paths``.

        All keyword arguments are forwarded verbatim. For the Heston model
        the return value is the ``(S_paths, V_paths)`` tuple; otherwise a
        single price-path array is returned.
        """
        return self.engine.generate_paths(**kwargs)

    def generate_spot(self, **kwargs) -> np.ndarray:
        """Return only the spot price paths, regardless of model.

        Convenience wrapper that discards the Heston variance paths so that
        downstream pricers can treat every model uniformly.
        """
        out = self.engine.generate_paths(**kwargs)
        if isinstance(out, tuple):
            return out[0]
        return out

    @staticmethod
    def available_models() -> list:
        """Return the list of supported model identifiers."""
        return sorted(set(_MODELS))
