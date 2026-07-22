"""Base class for probability-path interpolants."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseInterpolant(ABC):
    """Base class for a probability path between noise ``x0`` and data ``x1``.

    An interpolant defines, for each ``t in [0, 1]``, an intermediate point
    ``x_t`` and its conditional target velocity ``u_t`` such that regressing
    a model onto ``u_t`` (in expectation over the path) yields the marginal
    velocity field of the flow-matching ODE.

    Convention used throughout DeltaFlow: ``t = 0`` corresponds to noise
    (``x_t = x_0``) and ``t = 1`` corresponds to data (``x_t = x_1``).
    """

    @abstractmethod
    def interpolate(
        self, x1: torch.Tensor, t: torch.Tensor, x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_t, target_velocity)`` for the given data/time (and optional noise)."""
        raise NotImplementedError
