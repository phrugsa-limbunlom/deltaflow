"""Posterior sampler for inverse problems.

The :class:`PosteriorSolver` wraps an existing ODE solver (Euler, Heun,
...) and injects a measurement-likelihood gradient at every step, so the
underlying integrator is reused rather than re-implemented. This follows
the same design as FlowDPS (Chung et al., "FlowDPS: Flow-Driven Posterior
Sampling for Inverse Problems", 2024) and Flower ("Flower: Flow-based
inverse problem solver", 2024) - both modify the sampling-time ODE and do
not touch the pretrained velocity field.

Per step, given the current state ``x_t`` and the velocity ``v_theta(x_t, t)``,
the update is::

    x_clean_hat = Tweedie.decompose(x_t, v_t, t).x_clean
    g = grad_{x_t} -log p( y | x_clean_hat )
    x_{t+dt} = BaseSolver.step(x_t, t, dt) - eta * g

with ``eta`` a step-size (``guidance_scale``). The likelihood gradient is
computed by autograd, so if the velocity field operates on VAE latents and
the measurement operator is defined on pixels, the user just passes a
decoder to the likelihood object and the gradient is pulled back through
the decoder automatically.
"""

from typing import Optional

import torch

from ..core.base_solver import BaseSolver
from ..inverse.tweedie import BaseTweedie, LinearTweedie


class PosteriorSolver(BaseSolver):
    """Base-solver wrapper that adds a per-step measurement-likelihood gradient.

    Args:
        base_solver: any :class:`~deltaflow.core.base_solver.BaseSolver` that
            already integrates the unconditional flow (Euler, Heun, ...).
        likelihood: object with a ``.neg_log_prob(x_clean)`` method that
            returns a per-sample (or reducible) scalar tensor with
            ``requires_grad=True`` support. See
            :mod:`deltaflow.inverse.likelihood`.
        tweedie: flow-matching Tweedie decomposition to derive
            ``x_clean_hat`` from ``(x_t, v_t, t)``. Defaults to
            :class:`~deltaflow.inverse.tweedie.LinearTweedie`, matching a
            :class:`~deltaflow.interpolants.linear.LinearInterpolant`
            training path.
        guidance_scale: step size ``eta`` on the likelihood gradient. Larger
            values snap harder to the measurement but risk over-shooting.
        grad_normalize: if ``True``, the injected gradient is rescaled to
            match the norm of the base step; this is a stability trick used
            in some DPS variants when likelihood magnitudes vary wildly.
    """

    def __init__(
        self,
        base_solver: BaseSolver,
        likelihood,
        tweedie: Optional[BaseTweedie] = None,
        guidance_scale: float = 1.0,
        grad_normalize: bool = False,
    ):
        # Re-use the base solver's model reference and time_scale.
        super().__init__(model=base_solver.model, time_scale=base_solver.time_scale)
        self.base = base_solver
        self.likelihood = likelihood
        self.tweedie = tweedie or LinearTweedie()
        self.guidance_scale = guidance_scale
        self.grad_normalize = grad_normalize

    def _likelihood_grad(self, x: torch.Tensor, t: float, **cond) -> torch.Tensor:
        x_grad = x.detach().requires_grad_(True)
        with torch.enable_grad():
            v = self.base._eval_velocity(x_grad, t, **cond)
            x_clean_hat, _ = self.tweedie.decompose(x_grad, v, t)
            neg_log_lik = self.likelihood.neg_log_prob(x_clean_hat)
            if neg_log_lik.dim() > 0:
                neg_log_lik = neg_log_lik.sum()
            grad = torch.autograd.grad(neg_log_lik, x_grad)[0]
        return grad.detach()

    def step(self, x: torch.Tensor, t: float, dt: float, **cond) -> torch.Tensor:
        grad = self._likelihood_grad(x, t, **cond)
        with torch.no_grad():
            x_next = self.base.step(x, t, dt, **cond)
            correction = self.guidance_scale * grad
            if self.grad_normalize:
                base_step_norm = (x_next - x).flatten(1).norm(dim=1, keepdim=True)
                grad_norm = grad.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)
                scale = (base_step_norm / grad_norm).view(-1, *([1] * (grad.dim() - 1)))
                correction = self.guidance_scale * scale * grad
            return x_next - correction

    def sample(self, *args, **kwargs) -> torch.Tensor:
        kwargs.setdefault("progress_desc", "PosteriorSolver")
        # Autograd needs to run inside step(); do not use the enclosing no_grad
        # from the base implementation. BaseSolver.sample() does not wrap in
        # no_grad, so we can call through directly.
        return super().sample(*args, **kwargs)


__all__ = ["PosteriorSolver"]
