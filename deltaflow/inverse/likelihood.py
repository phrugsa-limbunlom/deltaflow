"""Measurement-likelihood objects for posterior sampling.

Given a measurement ``y = A(x_clean) + n`` with Gaussian noise ``n``, this
module supplies differentiable ``-log p(y | x_clean_hat)`` objects that
:class:`~deltaflow.solvers.posterior_solver.PosteriorSolver` differentiates
through to inject a likelihood gradient at every step.

**Latent-space case.** If the velocity field acts on VAE latents but the
measurement operator ``A`` is defined on pixels, pass a ``decoder``
callable (``latent -> pixel``). The likelihood becomes
``||y - A(decoder(x_clean_hat))||^2``, autograd walks back through the
decoder, and the resulting gradient lives in latent space where the
solver needs it. The decoder is *not* wrapped or trained here; it is just
a callable the user supplies (e.g. a frozen ``StableDiffusion`` VAE).
"""

from typing import Callable, Optional, Protocol, runtime_checkable

import torch


@runtime_checkable
class Likelihood(Protocol):
    """Structural type for measurement-likelihood objects.

    Any object exposing a differentiable ``neg_log_prob(x_clean_hat)`` method
    (returning a scalar or per-sample tensor that stays in ``x_clean_hat``'s
    autograd graph) satisfies this protocol and can be passed to
    :class:`~deltaflow.solvers.posterior_solver.PosteriorSolver`.
    """

    def neg_log_prob(self, x_clean_hat: torch.Tensor) -> torch.Tensor: ...


class GaussianLikelihood:
    """``-log p(y | x_clean_hat) = (1 / (2 sigma^2)) * || y - A(D(x_clean_hat)) ||^2``.

    Args:
        y: measured tensor, shape matches ``A``'s output.
        operator: linear measurement operator ``A`` (any callable, e.g.
            an instance from :mod:`deltaflow.inverse.operators`).
        sigma: measurement-noise standard deviation. Only affects the
            *scale* of the likelihood gradient - the direction is
            independent of ``sigma``.
        decoder: optional ``latent -> pixel`` callable, used when the
            velocity field is trained in a latent space but ``A`` is
            defined on pixels. Autograd flows through it automatically.
        reduction: ``"sum"`` (default, matches the log-density scaling) or
            ``"mean"``. The :class:`PosteriorSolver` sums per-sample
            values, so ``"sum"`` is usually what you want.
    """

    def __init__(
        self,
        y: torch.Tensor,
        operator: Callable[[torch.Tensor], torch.Tensor],
        sigma: float = 1.0,
        decoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        reduction: str = "sum",
    ):
        self.y = y
        self.operator = operator
        self.sigma = float(sigma)
        self.decoder = decoder
        if reduction not in ("sum", "mean"):
            raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")
        self.reduction = reduction

    def neg_log_prob(self, x_clean_hat: torch.Tensor) -> torch.Tensor:
        """Return ``-log p(y | x_clean_hat)`` as a scalar tensor.

        The result stays in the autograd graph of ``x_clean_hat`` so the
        posterior solver can backprop through it.
        """
        pixel = self.decoder(x_clean_hat) if self.decoder is not None else x_clean_hat
        y_pred = self.operator(pixel)
        diff = self.y.to(dtype=y_pred.dtype, device=y_pred.device) - y_pred
        sq = diff.pow(2)
        if self.reduction == "mean":
            sq = sq.mean()
        else:
            sq = sq.sum()
        return 0.5 * sq / (self.sigma * self.sigma)


__all__ = ["GaussianLikelihood", "Likelihood"]
