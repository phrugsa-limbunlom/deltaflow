"""Entropic Schrödinger-bridge probability path.

The static Schrödinger bridge between two marginals (here, the empirical
noise and data distributions) is the entropy-regularised optimal-transport
plan. Its dynamic counterpart is a diffusion process pinned at both ends;
conditioned on an endpoint pair ``(x0, x1)`` the bridge reduces to a
Brownian bridge

    x_t = (1 - t) * x0 + t * x1 + sigma * sqrt(t * (1 - t)) * z,   z ~ N(0, I)

whose conditional target velocity (the drift of the associated
probability-flow ODE, obtained by differentiating the reparameterised path
w.r.t. ``t`` at fixed ``z``) is

    u_t = (x1 - x0) + sigma * (1 - 2t) / (2 * sqrt(t * (1 - t))) * z.

As ``sigma -> 0`` this collapses onto the straight-line path of
:class:`~deltaflow.interpolants.linear.LinearInterpolant`. The
``(1 - 2t) / sqrt(t(1-t))`` term is unbounded as ``t`` approaches the
endpoints; it is stabilised here with a small denominator floor (``eps``),
which is the standard simulation-free training treatment near the
boundary (see references below).

This interpolant only defines the *path*; it is agnostic to how ``(x0,
x1)`` pairs are formed. Pairing ``x0``/``x1`` via mini-batch optimal
transport (:class:`~deltaflow.trainer.coupling.OTCoupling`) rather than
drawing them independently is what makes the discretised process converge
to the actual Schrödinger bridge rather than an arbitrary diffusion mixture
- see Tong et al. (2024) - and is the recommended way to combine the two.

References:
    De Bortoli et al., "Diffusion Schrödinger Bridge with Applications to
    Score-Based Generative Modeling" (2021), https://arxiv.org/abs/2106.01357.
    Tong et al., "Simulation-Free Schrödinger Bridges via Score and Flow
    Matching" (SF2M, 2024), https://arxiv.org/abs/2307.03672.
"""

from typing import Optional, Tuple

import torch

from ..core.base_interpolant import BaseInterpolant


class SchrodingerBridgeInterpolant(BaseInterpolant):
    """Brownian-bridge probability path with tunable diffusivity ``sigma``.

    Args:
        sigma: bridge diffusivity. ``sigma=0`` recovers the deterministic
            straight-line path (equivalent to :class:`LinearInterpolant`).
            Larger values inject more stochastic "wiggle" around the
            straight-line mean, matching the entropic-OT interpretation of
            the Schrödinger bridge (higher entropy regularisation -> more
            diffusive bridge).
        eps: numerical floor on the ``sqrt(t(1-t))`` denominator used when
            computing the target velocity, to keep it finite as ``t``
            approaches 0 or 1.
    """

    def __init__(self, sigma: float = 1.0, eps: float = 1e-4):
        self.sigma = sigma
        self.eps = eps

    def interpolate(
        self, x1: torch.Tensor, t: torch.Tensor, x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x0 is None:
            x0 = torch.randn_like(x1)
        t_ = t.view(-1, *([1] * (x1.dim() - 1)))

        mean = (1 - t_) * x0 + t_ * x1
        var = t_ * (1 - t_)
        std = self.sigma * torch.sqrt(var)

        z = torch.randn_like(x1)
        x_t = mean + std * z

        denom = torch.clamp(2 * torch.sqrt(var), min=self.eps)
        target_v = (x1 - x0) + self.sigma * (1 - 2 * t_) / denom * z
        return x_t, target_v
