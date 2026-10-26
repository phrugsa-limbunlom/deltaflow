"""Generable-set expansion, growing what a flow model can reach.

Distribution matching teaches a flow to reproduce its training density. Set
expansion asks a different question, which valid designs can the model
generate at all? This package implements Active Flow Expansion (ActFlow,
De Santi et al., 2026, arXiv:2606.08802), which fine-tunes a pre-trained flow
on self-generated, verifier-labelled samples chosen by active exploration in
the flow's own noised representation.

Modules:

- `representation`, the feature map ``phi_s`` (hidden activations of the
  velocity field at noise level ``s``).
- `uncertainty`, verifier-uncertainty estimators over ``phi_s``, including
  the closed-form kernel posterior standard deviation (Eq. 10).
- `guidance`, the uncertainty tilt ``p_theta * exp(sigma / beta)`` (Eq. 9)
  exposed as a `Likelihood` so `PosteriorSolver` can sample it.
- `actflow`, the round loop of Algorithm 1.
- `metrics`, histogram generable-set coverage for 2D problems.

The signed replay objective lives with the other losses as
`deltaflow.losses.SignedReplayFlowMatchingLoss`.
"""

from .actflow import ActFlow, ActFlowRound
from .guidance import UncertaintyGuidance
from .metrics import generable_coverage
from .representation import FlowRepresentation
from .uncertainty import BaseUncertainty, KernelUncertainty

__all__ = [
    "ActFlow",
    "ActFlowRound",
    "BaseUncertainty",
    "FlowRepresentation",
    "KernelUncertainty",
    "UncertaintyGuidance",
    "generable_coverage",
]
