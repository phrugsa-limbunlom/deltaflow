"""Train-time coupling strategies between source noise ``x0`` and data ``x1``.

Coupling is intentionally kept separate from the choice of probability
path (:mod:`deltaflow.interpolants`). The two decisions are independent
and swappable:

- Path choice (linear vs VP) is what the model regresses against.
- Coupling choice (independent vs OT) is which ``(x0, x1)`` pairs the
  regression is computed on.

Reference for OT coupling: Tong et al., "Improving and generalizing
flow-based generative models with minibatch optimal transport"
(arXiv:2302.00482). The OT-vs-independent ablation in Flower
("Flower: Flow-based inverse problems solver", 2024) recommends OT
coupling as the safer default when the trained model is later reused for
posterior sampling.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch

from ..interpolants.ot import _batch_ot_permutation


class BaseCoupling(ABC):
    """Given a batch ``x1`` of data samples, return a paired ``(x0, x1)``."""

    @abstractmethod
    def sample_pair(self, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class IndependentCoupling(BaseCoupling):
    """``x0 ~ N(0, I)`` drawn independently of ``x1``. Default flow-matching setup."""

    def sample_pair(self, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.randn_like(x1), x1


class OTCoupling(BaseCoupling):
    """Mini-batch optimal-transport coupling on squared-L2 cost.

    Draws ``x0 ~ N(0, I)`` and then permutes it within the batch so that
    each ``(x0, x1)`` pair minimises the batch's total transport cost. Uses
    ``scipy.optimize.linear_sum_assignment`` if available; otherwise falls
    back to a deterministic greedy nearest-neighbour matching.
    """

    def sample_pair(self, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = torch.randn_like(x1)
        perm = _batch_ot_permutation(x0, x1)
        return x0[perm], x1


__all__ = ["BaseCoupling", "IndependentCoupling", "OTCoupling"]
