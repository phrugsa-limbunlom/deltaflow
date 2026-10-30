"""Base class for ODE/SDE solvers that integrate a learned velocity field."""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from tqdm import tqdm


class BaseSolver(ABC):
    """Base class for numerical integrators of ``dx/dt = v_theta(x, t)``.

    A solver holds a reference to a velocity model and exposes two methods:

    - `step` performs a single integration step from ``(x, t)`` to
      ``(x', t + dt)``. Subclasses implement the actual stepping rule.
    - `sample` drives `step` in a loop from ``t_start`` to
      ``t_end`` and returns the final state.

    `sample` runs under ``torch.no_grad()``: sampling never needs gradients
    through the trajectory. Solvers that need autograd inside a step (e.g.
    `PosteriorSolver`, which wraps a `BaseSolver` and adds a likelihood
    gradient) enable it locally with ``torch.enable_grad()``.

    Args:
        model: callable ``model(x, t, **cond) -> velocity`` with the same
            signature as `BaseVelocityField`.
        time_scale: multiplies the continuous ``t in [0, 1]`` before it is
            passed to the model, useful when the backbone was trained with a
            different numeric time convention (e.g. diffusion timesteps).
    """

    def __init__(self, model: Callable, time_scale: float = 1.0):
        self.model = model
        self.time_scale = time_scale

    def velocity(self, x: torch.Tensor, t: float, **cond) -> torch.Tensor:
        """Evaluate the model at scalar time ``t`` (scaled by ``time_scale``)."""
        t_batch = torch.full((x.shape[0],), t * self.time_scale, device=x.device, dtype=x.dtype)
        return self.model(x, t_batch, **cond)

    @abstractmethod
    def step(self, x: torch.Tensor, t: float, dt: float, **cond) -> torch.Tensor:
        """Advance the state ``x`` from time ``t`` to ``t + dt``."""
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        n_steps: int = 50,
        t_start: float = 0.0,
        t_end: float = 1.0,
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
        **cond,
    ) -> torch.Tensor:
        """Integrate from ``t_start`` to ``t_end`` in ``n_steps`` uniform steps."""
        dt = (t_end - t_start) / n_steps
        steps = tqdm(
            range(n_steps),
            desc=progress_desc or type(self).__name__,
            leave=False,
            disable=not show_progress,
        )
        for i in steps:
            t_val = t_start + i * dt
            x = self.step(x, t_val, dt, **cond)
        return x
