"""Conditional flow matching loss."""

from typing import Optional

import torch
import torch.nn.functional as F

from ..interpolants.base import BaseInterpolant
from ..interpolants.linear import LinearInterpolant


class FlowMatchingLoss:
    """Regress a model's predicted velocity onto the conditional target
    velocity of a probability path.

    Reference: Lipman et al., "Flow Matching for Generative Modeling" (2023),
    https://arxiv.org/abs/2210.02747.

    Args:
        interpolant: the probability path to regress against. Defaults to
            :class:`~deltaflow.interpolants.linear.LinearInterpolant`.
        loss_type: one of ``"l2"``, ``"l1"``, ``"huber"``.
        time_scale: scales the continuous ``t in [0, 1]`` before it reaches
            the model, e.g. to match a diffusion-style time embedding.
    """

    def __init__(
        self,
        interpolant: Optional[BaseInterpolant] = None,
        loss_type: str = "l2",
        time_scale: float = 1.0,
    ):
        self.interpolant = interpolant or LinearInterpolant()
        self.loss_type = loss_type
        self.time_scale = time_scale

    def __call__(self, model, x1: torch.Tensor, **cond) -> torch.Tensor:
        t = torch.rand(x1.shape[0], device=x1.device)
        x_t, target_v = self.interpolant.interpolate(x1, t)
        pred_v = model(x_t, t * self.time_scale, **cond)

        if self.loss_type == "l1":
            return F.l1_loss(target_v, pred_v)
        if self.loss_type == "l2":
            return F.mse_loss(target_v, pred_v)
        if self.loss_type == "huber":
            return F.smooth_l1_loss(target_v, pred_v)
        raise NotImplementedError(f"Unknown loss_type: {self.loss_type!r}")
