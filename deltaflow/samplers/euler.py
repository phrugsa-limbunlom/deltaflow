"""Euler ODE sampler for flow-matching generation."""

from typing import Optional

import torch
from tqdm import tqdm


class FlowSampler:
    """Integrates ``dx/dt = v_theta(x, t)`` from ``t=0`` to ``t=1`` with the
    explicit Euler method.

    Args:
        model: a callable ``model(x, t, **cond) -> velocity`` (e.g. a
            :class:`~deltaflow.core.base.BaseVelocityField`).
        time_scale: multiplies the continuous ``t in [0, 1]`` before it is
            passed to the model, useful when the backbone's time embedding
            was trained on a larger numeric range (e.g. diffusion-style
            integer timesteps).
    """

    def __init__(self, model, time_scale: float = 1.0):
        self.model = model
        self.time_scale = time_scale

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        n_steps: int = 50,
        x_cond: Optional[torch.Tensor] = None,
        t_start: float = 0.0,
        show_progress: bool = True,
        **cond,
    ) -> torch.Tensor:
        """Generate samples by integrating the velocity field.

        Args:
            x: initial state. Ignored (but used for shape/device/dtype) if
                ``x_cond`` is given, since the starting point is then built
                by partially noising ``x_cond``.
            n_steps: number of Euler integration steps.
            x_cond: optional partially-noised starting point, for
                reconstruction-style sampling instead of sampling from pure
                noise.
            t_start: starting time; used together with ``x_cond``.
            **cond: extra keyword arguments forwarded to ``model``.
        """
        if x_cond is not None:
            x0 = torch.randn_like(x_cond)
            x = (1 - t_start) * x0 + t_start * x_cond

        dt = (1.0 - t_start) / n_steps
        steps = range(n_steps)
        if show_progress:
            steps = tqdm(steps, desc="FlowSampler (Euler)", total=n_steps, leave=False)

        for i in steps:
            t_val = t_start + i * dt
            t_tensor = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
            v = self.model(x, t_tensor * self.time_scale, **cond)
            x = x + v * dt

        return x
