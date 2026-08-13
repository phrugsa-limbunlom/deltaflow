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
    r"""Tweedie decomposition for the linear (rectified-flow) path.

    On the linear path \(x_t = (1-t)x_0 + t x_1\) with velocity \(v_t = x_1 -
    x_0\), the two endpoints are recovered in closed form by solving the
    \(2\times 2\) linear system:

    \[
    \hat{x}_1 = x_t + (1 - t)\,v_t, \qquad
    \hat{x}_0 = x_t - t\,v_t,
    \]

    where \(\hat{x}_1\) is the clean-data estimate and \(\hat{x}_0\) the noise
    estimate.
    """

    def decompose(self, x_t, v_t, t):
        t_ = _broadcast_time(t, x_t)
        x_clean = x_t + (1.0 - t_) * v_t
        x_noise = x_t - t_ * v_t
        return x_clean, x_noise


class VPTweedie(BaseTweedie):
    r"""Tweedie decomposition for the trigonometric variance-preserving path.

    With \(\alpha_t = \sin(\tfrac{\pi}{2}t)\), \(\sigma_t =
    \cos(\tfrac{\pi}{2}t)\), the path is \(x_t = \alpha_t x_1 + \sigma_t x_0\)
    and its velocity \(v_t = \tfrac{\pi}{2}(\sigma_t x_1 - \alpha_t x_0)\).
    Inverting this \(2\times 2\) system (using \(\alpha_t^2 + \sigma_t^2 = 1\))
    gives

    \[
    \hat{x}_1 = \alpha_t\,x_t + \frac{2}{\pi}\,\sigma_t\,v_t, \qquad
    \hat{x}_0 = \sigma_t\,x_t - \frac{2}{\pi}\,\alpha_t\,v_t.
    \]
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
