"""Mini-batch optimal transport coupling for linear flow-matching paths.

This interpolant reuses the straight-line path from `linear`, but
permutes ``x0`` within the current batch so that each ``(x0, x1)`` pair
approximately minimises the batch's total squared-L2 transport cost. In the
limit of large batches this converges to a coupling from an OT plan and
generally produces straighter learned trajectories - see Tong et al.,
"Improving and generalizing flow-based generative models with minibatch
optimal transport" (arXiv:2302.00482), and the OT-vs-independent ablation
in "Flower: A Flow-Matching Solver for Inverse Problems" (arXiv:2509.26287)
for the sampling side.

Exact optimal assignment (Hungarian algorithm) is used by default, since
``scipy`` is a core dependency. A deterministic greedy nearest-neighbour
fallback is used only if ``scipy`` is unavailable in the environment. This
is a coupling strategy, not a new probability path, the same
straight-line ``x_t = (1-t) x0 + t x1`` interpolation is applied after
permuting.
"""

from typing import Optional, Tuple

import torch

from ..core.base_interpolant import BaseInterpolant
from ..utils.ot import batch_ot_permutation
from .linear import LinearInterpolant

# Backward-compat alias: the implementation now lives in ``deltaflow.utils.ot``.
_batch_ot_permutation = batch_ot_permutation


class OTInterpolant(BaseInterpolant):
    r"""Linear probability path with mini-batch optimal-transport coupling.

    This reuses the straight-line path of
    `LinearInterpolant`, but instead of
    pairing noise and data independently it re-orders \(x_0\) within the batch
    to approximately solve the discrete optimal-transport assignment. Given a
    batch of noise \(\{x_0^{(i)}\}\) and data \(\{x_1^{(j)}\}\), it seeks a
    permutation \(\pi\) minimising the total squared-\(L_2\) transport cost

    \[
    \pi^\star = \arg\min_{\pi \in S_B}
        \sum_{i=1}^{B} \bigl\| x_0^{(i)} - x_1^{(\pi(i))} \bigr\|_2^2,
    \]

    then applies the linear interpolant to the matched pairs
    \(\bigl(x_0^{(\pi^\star(i))}, x_1^{(i)}\bigr)\). Concretely, each call

    1. draws (or receives) a batch of noise samples \(x_0\),
    2. computes \(\pi^\star\) so each noise sample is paired with the data
       sample that minimises the batch transport cost,
    3. applies `LinearInterpolant` on the permuted pair.

    In the large-batch limit this converges to a coupling drawn from the true
    OT plan and yields straighter learned trajectories that sample in fewer
    steps. It is the zero-entropy limit of the static Schrödinger bridge.
    Because the coupling is purely a re-ordering of \(x_0\), the training
    objective is identical to standard conditional flow matching and no other
    component (loss, solver, model) needs to change.

    **Solver.** The exact assignment (Hungarian algorithm) is used by
    default, since ``scipy`` is a core dependency. A deterministic greedy
    nearest-neighbour fallback is used only if ``scipy`` is unavailable.

    References:
        Tong et al., "Improving and generalizing flow-based generative
        models with minibatch optimal transport" (2023),
        https://arxiv.org/abs/2302.00482. "Flower: A Flow-Matching Solver for
        Inverse Problems" (2025), https://arxiv.org/abs/2509.26287.
    """

    def __init__(self):
        self._linear = LinearInterpolant()

    def interpolate(
        self, x1: torch.Tensor, t: torch.Tensor, x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x0 is None:
            x0 = torch.randn_like(x1)
        perm = batch_ot_permutation(x0, x1)
        x0 = x0[perm]
        return self._linear.interpolate(x1, t, x0=x0)
