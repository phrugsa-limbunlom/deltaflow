"""Heun's method (improved Euler) ODE solver."""

import torch

from ..core.base_solver import BaseSolver


class HeunSolver(BaseSolver):
    """Second-order predictor-corrector integrator.

    ::

        k1 = v(x_n,        t_n)
        k2 = v(x_n + dt*k1, t_n + dt)
        x_{n+1} = x_n + (dt/2) * (k1 + k2)

    Two velocity evaluations per step, but ``O(dt^3)`` local truncation
    error - typically matches Euler quality at half the number of steps.
    Widely used in EDM-style samplers.
    """

    @torch.no_grad()
    def step(self, x: torch.Tensor, t: float, dt: float, **cond) -> torch.Tensor:
        k1 = self._eval_velocity(x, t, **cond)
        x_pred = x + dt * k1
        k2 = self._eval_velocity(x_pred, t + dt, **cond)
        return x + 0.5 * dt * (k1 + k2)

    def sample(self, *args, **kwargs) -> torch.Tensor:
        kwargs.setdefault("progress_desc", "HeunSolver")
        return super().sample(*args, **kwargs)


__all__ = ["HeunSolver"]
