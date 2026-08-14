"""Posterior sampling machinery for linear inverse problems.

**v1 space decision.** DeltaFlow v1 targets the *pixel-space* case by default:
the velocity field ``v_theta(x, t)`` acts directly on images and the
measurement operator ``A`` is defined on the same space. However, all
components in `deltaflow.inverse` fully support the *latent-space*
case: `GaussianLikelihood` accepts an
optional ``decoder`` callable, and because the likelihood gradient is
computed by autograd, gradients are pulled back through the decoder into
the latent state automatically. See the docstring on ``GaussianLikelihood``
for the wiring.

Modules:

- `operators`, measurement operators ``A``: masks, blur, downsample.
- `tweedie`, flow-matching Tweedie decomposition of ``(x_t, v_t, t)``
  into a clean-signal estimate and a noise estimate (FlowDPS-style).
- `likelihood`, ``-log p(y | x_clean_hat)`` objects for the
  `PosteriorSolver`.
"""

from .likelihood import GaussianLikelihood, Likelihood
from .operators import BlurOperator, DownsampleOperator, IdentityOperator, MaskOperator
from .tweedie import BaseTweedie, LinearTweedie, VPTweedie

__all__ = [
    "BaseTweedie",
    "BlurOperator",
    "DownsampleOperator",
    "GaussianLikelihood",
    "IdentityOperator",
    "Likelihood",
    "LinearTweedie",
    "MaskOperator",
    "VPTweedie",
]
