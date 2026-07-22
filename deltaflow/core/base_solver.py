"""Base class for ODE/SDE solvers that integrate a learned velocity field."""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch


class BaseSolver(ABC):
    """Base class for numerical integrators of ``dx/dt = v_theta(x, t)``.

    A solver holds a reference to a velocity model and exposes two methods:

    - :meth:`step` performs a single integration step from ``(x, t)`` to
      ``(x', t + dt)``. Subclasses implement the actual stepping rule.
    - :meth:`sample` drives :meth:`step` in a loop from ``t_start`` to
      ``t_end`` and returns the final state.

    Design note: :class:`~deltaflow.solvers.posterior_solver.PosteriorSolver`
    wraps a :class:`BaseSolver` and hooks the likelihood gradient into every
    call to :meth:`step`, so the base stepping logic is never duplicated.

    Args:
        model: callable ``model(x, t, **cond) -> velocity`` with the same
            signature as :class:`~deltaflow.core.BaseVelocityField`.
        time_scale: multiplies the continuous ``t in [0, 1]`` before it is
            passed to the model; useful when the backbone was trained with a
            different numeric time convention (e.g. diffusion timesteps).
    """

    def __init__(self, model: Callable, time_scale: float = 1.0):
        self.model = model
        self.time_scale = time_scale

    def _time_tensor(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)

    def _eval_velocity(self, x: torch.Tensor, t: float, **cond) -> torch.Tensor:
        t_tensor = self._time_tensor(x, t) * self.time_scale
        return self.model(x, t_tensor, **cond)

    @abstractmethod
    def step(self, x: torch.Tensor, t: float, dt: float, **cond) -> torch.Tensor:
        """Advance the state ``x`` from time ``t`` to ``t + dt``."""
        raise NotImplementedError

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
        steps = range(n_steps)
        if show_progress:
            try:
                from tqdm import tqdm

                steps = tqdm(
                    steps,
                    desc=progress_desc or type(self).__name__,
                    total=n_steps,
                    leave=False,
                )
            except ImportError:
                pass

        for i in steps:
            t_val = t_start + i * dt
            x = self.step(x, t_val, dt, **cond)
        return x
