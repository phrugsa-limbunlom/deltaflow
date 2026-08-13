"""Heun's method (improved Euler) ODE solver."""

import torch

from ..core.base_solver import BaseSolver


class HeunSolver(BaseSolver):
    r"""Heun (improved-Euler) second-order predictor-corrector integrator.

    Each step takes an Euler predictor and averages the velocity at the
    current and predicted states,

    \[
    \begin{aligned}
    k_1 &= v_\theta(x_n, t_n), \\
    k_2 &= v_\theta\bigl(x_n + \Delta t\,k_1,\; t_n + \Delta t\bigr), \\
    x_{n+1} &= x_n + \tfrac{\Delta t}{2}\,(k_1 + k_2).
    \end{aligned}
    \]

    This costs two velocity evaluations per step but has
    \(\mathcal{O}(\Delta t^3)\) local truncation error (\(\mathcal{O}(\Delta
    t^2)\) global), so it typically matches Euler's quality at half the number
    of steps. It is the trapezoidal-rule integrator widely used in EDM-style
    samplers.
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
