"""Flow-matching Tweedie decomposition.

Given a state ``x_t`` on a probability path and the corresponding velocity
``v_t = v_theta(x_t, t)``, invert the path definition to recover
estimates of the two endpoints:

- ``x_clean_hat`` -- an estimate of the data endpoint (``t = 1`` in
  DeltaFlow's convention).
- ``x_noise_hat`` -- an estimate of the noise endpoint (``t = 0``).

This is the flow-matching analogue of Tweedie's formula for diffusion
models, and is what FlowDPS uses to compute a differentiable measurement
likelihood on the clean-signal estimate at every sampling step.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union

import torch


def _broadcast_time(t: Union[float, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        if t.dim() == 0:
            return t
        return t.view(-1, *([1] * (x.dim() - 1)))
    return torch.tensor(t, device=x.device, dtype=x.dtype)


class BaseTweedie(ABC):
    """Base class for path-specific ``(x_t, v_t, t) -> (x_clean, x_noise)``."""

    @abstractmethod
    def decompose(
        self,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        t: Union[float, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class LinearTweedie(BaseTweedie):
    """Tweedie decomposition for the linear (rectified-flow) path.

    Path: ``x_t = (1 - t) * x0 + t * x1``, ``v_t = x1 - x0``. Solving::

        x_clean = x1_hat = x_t + (1 - t) * v_t
        x_noise = x0_hat = x_t - t * v_t
    """

    def decompose(self, x_t, v_t, t):
        t_ = _broadcast_time(t, x_t)
        x_clean = x_t + (1.0 - t_) * v_t
        x_noise = x_t - t_ * v_t
        return x_clean, x_noise


class VPTweedie(BaseTweedie):
    """Tweedie decomposition for the trigonometric variance-preserving path.

    Path: ``alpha = sin(pi/2 * t)``, ``sigma = cos(pi/2 * t)``,
    ``x_t = alpha * x1 + sigma * x0``, ``v_t = (pi/2)(cos * x1 - sin * x0)``.
    Inverting the 2x2 system yields::

        x_clean = alpha * x_t + (2/pi) * sigma * v_t
        x_noise = sigma * x_t - (2/pi) * alpha * v_t
    """

    def decompose(self, x_t, v_t, t):
        t_ = _broadcast_time(t, x_t)
        half_pi = 0.5 * math.pi
        alpha = torch.sin(half_pi * t_) if isinstance(t_, torch.Tensor) else math.sin(half_pi * float(t_))
        sigma = torch.cos(half_pi * t_) if isinstance(t_, torch.Tensor) else math.cos(half_pi * float(t_))
        two_over_pi = 2.0 / math.pi
        x_clean = alpha * x_t + two_over_pi * sigma * v_t
        x_noise = sigma * x_t - two_over_pi * alpha * v_t
        return x_clean, x_noise


@dataclass
class TweedieEstimates:
    """Bundle of the two endpoint estimates for convenience."""

    x_clean: torch.Tensor
    x_noise: torch.Tensor


__all__ = ["BaseTweedie", "LinearTweedie", "TweedieEstimates", "VPTweedie"]
