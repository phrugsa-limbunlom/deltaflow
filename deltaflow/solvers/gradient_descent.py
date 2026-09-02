"""Optimization-based sampler for Equilibrium Matching fields."""

from typing import Any, Optional

import torch

from ..core.base_solver import BaseSolver


class EquilibriumSolver(BaseSolver):
    r"""Gradient-descent sampler for an Equilibrium Matching landscape.

    A field trained with `EquilibriumInterpolant`
    approximates the *time-invariant* equilibrium gradient of an implicit
    energy \(E\), pointing from noise toward data, \(f(x) \approx -\nabla
    E(x)\). Unlike flow/diffusion models, samples are not obtained by
    integrating a velocity over a fixed time horizon. Instead they are found by
    optimising a noise draw down the landscape,

    \[
    x_{k+1} = x_k + \eta\, f(x_k),
    \]

    which is gradient descent on \(E\) (ascent along \(f\)). Because sampling
    is decoupled from any prescribed trajectory, the step size ``step_size``
    and the number of steps ``n_steps`` are free knobs, and adaptive compute is
    just running more iterations.

    **Nesterov (NAG-GD).** With ``momentum > 0`` the solver uses the
    accelerated look-ahead update from the paper,

    \[
    x_k' = x_k + \eta\,\mu\, m_k,\quad
    m_{k+1} = f(x_k'),\quad
    x_{k+1} = x_k + \eta\, m_{k+1},
    \]

    which reaches high-quality samples in fewer steps than vanilla gradient
    descent.

    **Time invariance.** The trained field ignores its time argument, so the
    solver evaluates it at a single fixed ``eval_time`` (its value does not
    affect a properly EqM-trained, noise-unconditional model). ``eval_time`` is
    still routed through ``time_scale`` for backbones that expect a rescaled
    time input.

    References:
        Wang and Du, "Equilibrium Matching: Generative Modeling with Implicit
        Energy-Based Models" (2025), https://arxiv.org/abs/2510.02300.

    Args:
        model: callable ``model(x, t, **cond) -> gradient`` (same signature as
            `BaseVelocityField`),
            returning the equilibrium gradient with the shape of ``x``.
        step_size: gradient-descent step \(\eta\).
        momentum: Nesterov coefficient \(\mu\). ``0`` gives vanilla gradient
            descent, positive values give NAG-GD.
        eval_time: fixed time at which the time-invariant field is queried.
        time_scale: multiplies ``eval_time`` before it reaches the model.
    """

    def __init__(
        self,
        model,
        step_size: float = 0.05,
        momentum: float = 0.0,
        eval_time: float = 1.0,
        time_scale: float = 1.0,
    ):
        super().__init__(model, time_scale=time_scale)
        self.step_size = step_size
        self.momentum = momentum
        self.eval_time = eval_time

    def _gradient(self, x: torch.Tensor, **cond) -> torch.Tensor:
        """Equilibrium gradient ``f(x)``, pointing from noise toward data."""
        return self._eval_velocity(x, self.eval_time, **cond)

    def step(self, x: torch.Tensor, t: float, dt: float, **cond) -> torch.Tensor:
        """Single vanilla gradient-descent step (``t``/``dt`` are ignored).

        Provided for `BaseSolver` compatibility. The momentum-aware loop lives
        in `sample`.
        """
        return x + self.step_size * self._gradient(x, **cond)

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        n_steps: int = 100,
        step_size: Optional[float] = None,
        momentum: Optional[float] = None,
        show_progress: bool = True,
        **cond: Any,
    ) -> torch.Tensor:
        """Sample by gradient descent on the learned equilibrium landscape.

        Args:
            x: initial noise state (shape/device/dtype template).
            n_steps: number of gradient-descent iterations.
            step_size: overrides ``self.step_size`` for this call.
            momentum: overrides ``self.momentum`` for this call (Nesterov
                look-ahead when positive).
            show_progress: display a tqdm progress bar if available.
            **cond: extra keyword arguments forwarded to the model unchanged.
        """
        eta = self.step_size if step_size is None else step_size
        mu = self.momentum if momentum is None else momentum

        steps = range(n_steps)
        if show_progress:
            try:
                from tqdm import tqdm

                steps = tqdm(steps, desc=type(self).__name__, total=n_steps, leave=False)
            except ImportError:
                pass

        m = torch.zeros_like(x)
        for _ in steps:
            if mu != 0.0:
                x_look = x + eta * mu * m
                m = self._gradient(x_look, **cond)
                grad = m
            else:
                grad = self._gradient(x, **cond)
            x = x + eta * grad
        return x


# Descriptive alias for the vanilla (momentum-free) sampler.
GradientDescentSolver = EquilibriumSolver

__all__ = ["EquilibriumSolver", "GradientDescentSolver"]
