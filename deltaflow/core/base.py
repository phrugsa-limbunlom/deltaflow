"""Base class for velocity fields consumed by DeltaFlow losses and samplers."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseVelocityField(nn.Module, ABC):
    """Base class for the time-conditioned velocity field ``v_theta(x, t)``.

    Subclasses must implement :meth:`forward` and return a tensor with the
    same shape as ``x``. Any additional conditioning (e.g. a guidance flag,
    class label, or cross-attention context) can be passed as keyword
    arguments and is forwarded unchanged by :class:`~deltaflow.losses.flow_matching.FlowMatchingLoss`
    and :class:`~deltaflow.samplers.euler.FlowSampler`.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, **cond) -> torch.Tensor:
        raise NotImplementedError
