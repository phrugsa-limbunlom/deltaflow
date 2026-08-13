"""Conditional flow matching loss with pluggable train-time coupling."""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from ..core.base_interpolant import BaseInterpolant
from ..core.base_loss import BaseLoss
from ..interpolants.linear import LinearInterpolant

if TYPE_CHECKING:
    from ..trainer.coupling import BaseCoupling


class ConditionalFlowMatchingLoss(BaseLoss):
    r"""Regress a velocity field onto the conditional target velocity of a path.

    Conditional flow matching trains \(v_\theta\) to match the per-pair target
    velocity \(u_t\) of a probability path by minimising

    \[
    \mathcal{L} = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_0,\, x_1}
        \bigl\| v_\theta(x_t, t) - u_t \bigr\|^2,
    \]

    where \((x_t, u_t)\) are produced by the chosen ``interpolant`` (for the
    linear path, \(x_t = (1-t)x_0 + t x_1\) and \(u_t = x_1 - x_0\)). Although
    the target is only defined *conditionally* on \((x_0, x_1)\), its
    regression minimiser is the marginal velocity field that transports noise
    onto data, which is exactly the field the sampler integrates.

    Reference: Lipman et al., "Flow Matching for Generative Modeling" (2023),
    https://arxiv.org/abs/2210.02747.

    Args:
        interpolant: the probability path to regress against. Defaults to
            :class:`~deltaflow.interpolants.linear.LinearInterpolant`.
        coupling: optional train-time coupling that produces \((x_0, x_1)\)
            pairs from a batch of \(x_1\). See
            :mod:`deltaflow.trainer.coupling`. Kept separate from
            ``interpolant`` on purpose, since the two decisions (which path to
            use, and how to pair noise with data) are independent and
            swappable.
        loss_type: one of ``"l2"``, ``"l1"``, ``"huber"``.
        time_scale: scales the continuous \(t \in [0, 1]\) before it reaches
            the model, e.g. to match a diffusion-style time embedding.
    """

    def __init__(
        self,
        interpolant: Optional[BaseInterpolant] = None,
        coupling: Optional["BaseCoupling"] = None,
        loss_type: str = "l2",
        time_scale: float = 1.0,
    ):
        self.interpolant = interpolant or LinearInterpolant()
        self.coupling = coupling
        self.loss_type = loss_type
        self.time_scale = time_scale

    def _reduce(self, target_v: torch.Tensor, pred_v: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(target_v, pred_v)
        if self.loss_type == "l2":
            return F.mse_loss(target_v, pred_v)
        if self.loss_type == "huber":
            return F.smooth_l1_loss(target_v, pred_v)
        raise NotImplementedError(f"Unknown loss_type: {self.loss_type!r}")

    def __call__(self, model, x1: torch.Tensor, **cond) -> torch.Tensor:
        if self.coupling is not None:
            x0, x1 = self.coupling.sample_pair(x1)
        else:
            x0 = None
        t = torch.rand(x1.shape[0], device=x1.device)
        x_t, target_v = self.interpolant.interpolate(x1, t, x0=x0)
        pred_v = model(x_t, t * self.time_scale, **cond)
        return self._reduce(target_v, pred_v)


# Backwards-compatible alias: the earlier public name for this loss.
FlowMatchingLoss = ConditionalFlowMatchingLoss

__all__ = ["ConditionalFlowMatchingLoss", "FlowMatchingLoss"]
