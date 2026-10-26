"""Uncertainty-tilted sampling, ActFlow's inference-time active exploration.

ActFlow self-generates queries from the KL-regularised objective (its Eq. 9)

    q* = argmax_q  E_{x ~ q}[ sigma_t(phi_s(x)) ] - beta * KL(q || p_theta),

whose maximiser is the exponential tilt of the current model density,

    q*(x)  proportional to  p_theta(x) * exp( sigma_t(phi_s(x)) / beta ).

Sampling a tilt of a flow model is exactly the problem posterior samplers
already solve, with ``exp(sigma / beta)`` in place of a measurement
likelihood. So instead of a new integrator, `UncertaintyGuidance` exposes the
tilt through the `Likelihood` protocol,

    neg_log_prob(x_clean_hat) = - sigma_t(phi_s(x_clean_hat)) / beta,

and `PosteriorSolver` applies its gradient per step on the Tweedie estimate
of the clean sample (a DPS-style approximation of the tilted path). As
``beta -> inf`` the gradient vanishes and plain sampling from ``p_theta`` is
recovered, as ``beta -> 0`` it becomes pure uncertainty maximisation.
"""

import math
from typing import Any, Dict, Optional

import torch

from .representation import FlowRepresentation
from .uncertainty import BaseUncertainty


class UncertaintyGuidance:
    r"""Likelihood adapter for the uncertainty tilt \(p_\theta \exp(\sigma_t / \beta)\).

    Implements ActFlow's active-exploration objective

    \[
    \tilde{p}_t \in \arg\max_q\;
        \mathbb{E}_{x \sim q}\bigl[\sigma_t(\phi_s(x))\bigr]
        - \beta\, \mathrm{KL}\bigl(q \,\Vert\, p_1^{\theta_t}\bigr),
    \qquad
    \tilde{p}_t(x) \propto p_1^{\theta_t}(x)\,
        \exp\!\bigl(\sigma_t(\phi_s(x)) / \beta\bigr),
    \]

    by returning \(-\sigma_t(\phi_s(\hat{x}_1)) / \beta\) as a negative
    log-likelihood. Pass it to `PosteriorSolver` as ``likelihood``. The
    solver differentiates it through `FlowRepresentation` and the Tweedie
    estimate \(\hat{x}_1\) back to the current state \(x_t\).

    References:
        De Santi et al., "Active Flow Expansion for Out-of-Distribution
        Discovery: from Theory to Molecules" (2026),
        https://arxiv.org/abs/2606.08802.

    Args:
        representation: the feature map \(\phi_s\).
        uncertainty: a fitted `BaseUncertainty` giving \(\sigma_t\).
        beta: KL weight \(\beta > 0\). ``math.inf`` disables the tilt.
        cond: conditioning forwarded unchanged to the velocity field inside
            \(\phi_s\) (the `Likelihood` protocol carries no ``**cond``).
    """

    def __init__(
        self,
        representation: FlowRepresentation,
        uncertainty: BaseUncertainty,
        beta: float = 0.1,
        cond: Optional[Dict[str, Any]] = None,
    ):
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.representation = representation
        self.uncertainty = uncertainty
        self.beta = float(beta)
        self.cond = dict(cond or {})

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample uncertainty \\(\\sigma_t(\\phi_s(x))\\), shape ``[B]``."""
        return self.uncertainty(self.representation(x, **self.cond))

    def neg_log_prob(self, x_clean_hat: torch.Tensor) -> torch.Tensor:
        if math.isinf(self.beta):
            return x_clean_hat.sum() * 0.0
        return -self.score(x_clean_hat).sum() / self.beta


__all__ = ["UncertaintyGuidance"]
