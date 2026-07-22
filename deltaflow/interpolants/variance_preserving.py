"""Variance-preserving (VP) probability path with a trigonometric schedule.

``x_t = alpha_t * x1 + sigma_t * x0`` with

    alpha_t = sin(pi/2 * t),   sigma_t = cos(pi/2 * t),

so that ``alpha^2 + sigma^2 = 1`` at every ``t`` (variance-preserving). The
conditional target velocity is

    u_t = alpha'_t * x1 + sigma'_t * x0
        = (pi/2) * [cos(pi/2 * t) * x1 - sin(pi/2 * t) * x0].

Reference: Lipman et al., "Flow Matching for Generative Modeling" (2023),
Sec. 3.2; Ma et al., "SiT: Exploring Flow and Diffusion-Based Generative
Models" (2024).
"""

import math
from typing import Optional, Tuple

import torch

from ..core.base_interpolant import BaseInterpolant


class VariancePreservingInterpolant(BaseInterpolant):
    """Trigonometric variance-preserving probability path."""

    def interpolate(
        self, x1: torch.Tensor, t: torch.Tensor, x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x0 is None:
            x0 = torch.randn_like(x1)
        t_ = t.view(-1, *([1] * (x1.dim() - 1)))
        half_pi = 0.5 * math.pi
        alpha = torch.sin(half_pi * t_)
        sigma = torch.cos(half_pi * t_)
        x_t = alpha * x1 + sigma * x0
        target_v = half_pi * (torch.cos(half_pi * t_) * x1 - torch.sin(half_pi * t_) * x0)
        return x_t, target_v
