"""Linear (rectified-flow) probability path."""

from typing import Optional, Tuple

import torch

from ..core.base_interpolant import BaseInterpolant


class LinearInterpolant(BaseInterpolant):
    r"""Straight-line (rectified-flow) probability path between noise and data.

    The path linearly interpolates a noise sample \(x_0\) and a data sample
    \(x_1\),

    \[
    x_t = (1 - t)\,x_0 + t\,x_1, \qquad t \in [0, 1],
    \]

    whose conditional target velocity is the constant displacement

    \[
    u_t = \frac{\mathrm{d} x_t}{\mathrm{d} t} = x_1 - x_0.
    \]

    Because \(u_t\) does not depend on \(t\), the learned field regresses onto
    a single displacement vector per pair, which is what makes rectified-flow
    trajectories straight and cheap to integrate.

    **Coupling.** This is the *independent-coupling* variant: if \(x_0\) is not
    supplied it is drawn from a standard normal \(\mathcal{N}(0, I)\)
    independently of \(x_1\). For OT-coupled linear paths see
    `OTInterpolant` or use
    `OTCoupling` on the training side.

    References:
        Lipman et al., "Flow Matching for Generative Modeling" (2023),
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
