"""Linear (rectified-flow) probability path."""

from typing import Optional, Tuple

import torch

from ..core.base_interpolant import BaseInterpolant


class LinearInterpolant(BaseInterpolant):
    """Straight-line probability path between noise and data.

    ``x_t = (1 - t) * x0 + t * x1``, with conditional target velocity
    ``u_t = x1 - x0`` (constant along the path). This is the "independent
    coupling" variant: if ``x0`` is not supplied, it is drawn from a
    standard normal independently of ``x1``. For OT-coupled linear paths
    see :class:`~deltaflow.interpolants.ot.OTInterpolant` or use
    :class:`~deltaflow.trainer.coupling.OTCoupling` on the training side.

    Reference: Lipman et al., "Flow Matching for Generative Modeling" (2023),
    https://arxiv.org/abs/2210.02747.
    """

    def interpolate(
        self, x1: torch.Tensor, t: torch.Tensor, x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x0 is None:
            x0 = torch.randn_like(x1)
        t_ = t.view(-1, *([1] * (x1.dim() - 1)))
        x_t = (1 - t_) * x0 + t_ * x1
        target_v = x1 - x0
        return x_t, target_v
