"""Mini-batch optimal transport coupling for linear flow-matching paths.

This interpolant reuses the straight-line path from :mod:`.linear`, but
permutes ``x0`` within the current batch so that each ``(x0, x1)`` pair
approximately minimises the batch's total squared-L2 transport cost. In the
limit of large batches this converges to a coupling from an OT plan and
generally produces straighter learned trajectories - see Tong et al.,
"Improving and generalizing flow-based generative models with minibatch
optimal transport" (arXiv:2302.00482), and the OT-vs-independent ablation
in "Flower: Flow-based inverse problems solver" for the sampling side.

Exact optimal assignment (Hungarian algorithm) is used when ``scipy`` is
installed; otherwise a deterministic greedy nearest-neighbour fallback is
used. This is a coupling strategy, not a new probability path - the same
straight-line ``x_t = (1-t) x0 + t x1`` interpolation is applied after
permuting.
"""

from typing import Optional, Tuple

import torch

from ..core.base_interpolant import BaseInterpolant
from .linear import LinearInterpolant


def _batch_ot_permutation(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Return a permutation ``perm`` such that ``x0[perm]`` is OT-coupled to ``x1``.

    Costs are squared L2 distances on the flattened per-sample tensors.
    """
    b = x0.shape[0]
    if b == 1:
        return torch.zeros(1, dtype=torch.long, device=x0.device)

    x0f = x0.reshape(b, -1).float()
    x1f = x1.reshape(b, -1).float()
    cost = torch.cdist(x0f, x1f) ** 2  # (B, B), cost[i, j] = |x0[i] - x1[j]|^2

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        # linear_sum_assignment guarantees row_ind == 0..B-1 in sorted order;
        # col_ind[i] is the x1 index paired with x0[i]. We want a permutation
        # of x0 aligned to x1's original order: for each x1[j], take x0[i] where col_ind[i] == j.
        col = torch.as_tensor(col_ind, dtype=torch.long, device=x0.device)
        perm = torch.argsort(col)
        return perm
    except ImportError:
        # Greedy fallback: for each x1[j] in order, pick the closest un-used x0[i].
        used = torch.zeros(b, dtype=torch.bool, device=x0.device)
        perm = torch.empty(b, dtype=torch.long, device=x0.device)
        for j in range(b):
            row = cost[:, j].clone()
            row[used] = float("inf")
            i = int(torch.argmin(row).item())
            perm[j] = i
            used[i] = True
        return perm


class OTInterpolant(BaseInterpolant):
    """Linear probability path with mini-batch OT coupling of ``(x0, x1)``.

    Concretely, on each call this interpolant

    1. draws (or receives) a batch of noise samples ``x0``,
    2. computes a permutation of ``x0`` that pairs each sample with a data
       sample ``x1`` so as to minimise the batch's total squared-L2
       transport cost,
    3. applies :class:`LinearInterpolant` on the permuted pair.

    Because the coupling is purely a re-ordering of ``x0``, the training
    objective is identical to standard conditional flow matching and no
    other component (loss, solver, model) needs to change.
    """

    def __init__(self):
        self._linear = LinearInterpolant()

    def interpolate(
        self, x1: torch.Tensor, t: torch.Tensor, x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x0 is None:
            x0 = torch.randn_like(x1)
        perm = _batch_ot_permutation(x0, x1)
        x0 = x0[perm]
        return self._linear.interpolate(x1, t, x0=x0)
