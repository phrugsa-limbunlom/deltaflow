"""Base class for DeltaFlow training objectives."""

from abc import ABC, abstractmethod

import torch


class BaseLoss(ABC):
    """Base class for a callable training loss.

    Subclasses must implement `__call__` and return a scalar tensor
    with ``requires_grad=True`` (assuming the model has trainable
    parameters). The signature is intentionally flexible - individual losses
    define which positional and keyword arguments they consume.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
