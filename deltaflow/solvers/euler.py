"""Euler ODE solver for flow-matching generation."""

from typing import Any, Optional

import torch

from ..core.base_solver import BaseSolver


class EulerSolver(BaseSolver):
    r"""Explicit (forward) Euler integrator for the flow-matching ODE.

    Integrates \(\mathrm{d}x/\mathrm{d}t = v_\theta(x, t)\) with the
    first-order update

    \[
    x_{n+1} = x_n + \Delta t\; v_\theta(x_n, t_n),
    \]

    using a single velocity evaluation per step. It is cheap but incurs
    \(\mathcal{O}(\Delta t^2)\) local truncation error (\(\mathcal{O}(\Delta
    t)\) global), so prefer `HeunSolver` when
    accuracy at low step counts matters.
    """

    def step(self, x: torch.Tensor, t: float, dt: float, **cond) -> torch.Tensor:
        v = self._eval_velocity(x, t, **cond)
        return x + dt * v

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        n_steps: int = 50,
        x_cond: Optional[torch.Tensor] = None,
        t_start: float = 0.0,
        t_end: float = 1.0,
        show_progress: bool = True,
        **cond: Any,
    ) -> torch.Tensor:
        """Generate samples by integrating the velocity field.

        Args:
            x: initial state (shape/device/dtype template if ``x_cond`` is set).
            n_steps: number of Euler integration steps.
            x_cond: optional partially-noised starting point, if given the
                initial state becomes ``(1 - t_start) * noise + t_start * x_cond``.
            t_start: starting time, used together with ``x_cond``.
            t_end: end time, defaults to ``1.0``.
            **cond: extra keyword arguments forwarded to the velocity model.
        """
        if x_cond is not None:
            x0 = torch.randn_like(x_cond)
            x = (1 - t_start) * x0 + t_start * x_cond
        return super().sample(
            x,
            n_steps=n_steps,
            t_start=t_start,
            t_end=t_end,
            show_progress=show_progress,
            progress_desc="EulerSolver",
            **cond,
        )


# Backwards-compatible alias: earlier public name for this solver.
FlowSampler = EulerSolver

__all__ = ["EulerSolver", "FlowSampler"]
