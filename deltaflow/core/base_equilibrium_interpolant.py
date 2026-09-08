"""Base class for Equilibrium Matching interpolants (parametrised by gamma)."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseEquilibriumInterpolant(ABC):
    r"""Base class for an energy-compatible path indexed by a coefficient gamma.

    This base is deliberately separate from `BaseInterpolant`. A flow-matching
    interpolant is parametrised by a dynamical time ``t`` that a sampler
    integrates over. An Equilibrium Matching interpolant is parametrised by an
    interpolation coefficient \(\gamma \in [0, 1]\), a noise level rather than a
    time. The learned field is trained to be time-invariant, so \(\gamma\) only
    indexes where along the noise-to-data path a training point sits, it is
    never integrated and (per the EqM paper) is not seen by the model.

    Convention: \(\gamma = 0\) is noise (\(x_\gamma = x_0\)) and \(\gamma = 1\)
    is data (\(x_\gamma = x_1\)).

    References:
        Wang and Du, "Equilibrium Matching: Generative Modeling with Implicit
        Energy-Based Models" (2025), https://arxiv.org/abs/2510.02300.
    """

    @abstractmethod
    def interpolate(
        self, x1: torch.Tensor, gamma: torch.Tensor, x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return ``(x_gamma, target)`` for the given data and coefficient.

        Args:
            x1: data sample (the \(\gamma = 1\) endpoint).
            gamma: interpolation coefficient (noise level) in \([0, 1]\), not a
                dynamical time.
            x0: optional noise sample (the \(\gamma = 0\) endpoint). Drawn from
                a standard normal when omitted.
        """
        raise NotImplementedError
