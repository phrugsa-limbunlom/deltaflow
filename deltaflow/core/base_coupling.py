"""Base class for train-time coupling strategies between ``x0`` and ``x1``.

Coupling is the choice of which ``(x0, x1)`` pairs the flow-matching
regression is computed on, kept deliberately separate from the choice of
probability path (`deltaflow.interpolants`). Concrete strategies live in
`deltaflow.trainer.coupling`, this base sits in ``core`` alongside the other
drop-in abstractions so a new coupling is a subclass rather than a rewrite of
the surrounding training loop.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class BaseCoupling(ABC):
    """Given a batch ``x1`` of data samples, return a paired ``(x0, x1)``."""

    @abstractmethod
    def sample_pair(self, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
