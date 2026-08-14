"""Base class for velocity fields consumed by DeltaFlow losses and solvers."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseVelocityField(nn.Module, ABC):
    """Base class for the time-conditioned velocity field ``v_theta(x, t)``.

    Subclasses must implement `forward` and return a tensor with the
    same shape as ``x``. Any additional conditioning (e.g. a guidance flag,
    class label, or cross-attention context) can be passed as keyword
    arguments and is forwarded unchanged by the losses and solvers.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, **cond) -> torch.Tensor:
        raise NotImplementedError
