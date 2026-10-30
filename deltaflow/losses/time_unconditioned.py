"""Time-unconditioned (time-blind) flow matching loss."""

from typing import Optional

import torch

from ..core.base_coupling import BaseCoupling
from ..core.base_interpolant import BaseInterpolant
from ..core.base_loss import BaseLoss
from ..interpolants.linear import LinearInterpolant
from ._reduction import regression_loss


class TimeUnconditionedFlowMatchingLoss(BaseLoss):
    r"""Flow matching where the model is not given the interpolation time.

    Trains \(v_\theta(x_t)\) (no \(t\)) to match the conditional target
    velocity:

    \[
    \mathcal{L} = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_0,\, x_1}
        \bigl\| v_\theta(x_t) - u_t \bigr\|^2 .
    \]

    The time-blind loss exceeds the conditioned one by a "time-blindness gap",
    but in high dimensions time is identifiable from the noisy interpolant
    itself (data concentrates near a low-dimensional subspace), so the gap is
    asymptotically negligible next to the coupling variance. The choice of
    ``coupling`` is therefore the main practical lever.

    References:
        "What Time Is It? How Data Geometry Makes Time Conditioning Optional
        for Flow Matching" (2026), https://arxiv.org/abs/2605.08344.

    Args:
        interpolant: probability path. Defaults to `LinearInterpolant`.
        coupling: optional train-time coupling producing \((x_0, x_1)\) pairs.
        loss_type: one of ``"l2"``, ``"l1"``, ``"huber"``.
        pass_time: if ``False`` (default) the model is called as
            ``model(x_t, **cond)``. If ``True`` it is called as
            ``model(x_t, t, **cond)`` with a constant zero ``t``, for
            models whose signature requires a time argument (e.g.
            `TinyVelocityField`); the zero carries no information.
    """

    def __init__(
        self,
        interpolant: Optional[BaseInterpolant] = None,
        coupling: Optional[BaseCoupling] = None,
        loss_type: str = "l2",
        pass_time: bool = False,
    ):
        self.interpolant = interpolant or LinearInterpolant()
        self.coupling = coupling
        self.loss_type = loss_type
        self.pass_time = pass_time

    def __call__(self, model, x1: torch.Tensor, **cond) -> torch.Tensor:
        x0 = None
        if self.coupling is not None:
            x0, x1 = self.coupling.sample_pair(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        x_t, target_v = self.interpolant.interpolate(x1, t, x0=x0)
        if self.pass_time:
            pred_v = model(x_t, torch.zeros_like(t), **cond)
        else:
            pred_v = model(x_t, **cond)
        return regression_loss(pred_v, target_v, self.loss_type)


__all__ = ["TimeUnconditionedFlowMatchingLoss"]
