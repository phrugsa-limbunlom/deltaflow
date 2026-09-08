"""Base class for time-invariant Equilibrium Matching fields ``f(x)``."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEquilibriumField(nn.Module, ABC):
    r"""Base class for a time-invariant Equilibrium Matching field \(f(x)\).

    Unlike a flow-matching velocity field \(v_\theta(x, t)\), an EqM field
    carries no time argument. It approximates the equilibrium gradient of an
    implicit energy, \(f_\theta(x) \approx -\nabla E(x)\), pointing from noise
    toward data. The interpolation coefficient \(\gamma\) is implicit and (per
    the EqM paper) never seen by the model, so ``forward`` takes only ``x`` and
    any extra conditioning, never a time or \(\gamma\).

    Subclasses implement `forward` and return a tensor with the same
    shape as ``x``. Additional conditioning is passed as keyword arguments and
    forwarded unchanged by `EquilibriumMatchingLoss` and
    `EquilibriumSolver`.

    References:
        Wang and Du, "Equilibrium Matching: Generative Modeling with Implicit
        Energy-Based Models" (2025), https://arxiv.org/abs/2510.02300.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, **cond) -> torch.Tensor:
        raise NotImplementedError
