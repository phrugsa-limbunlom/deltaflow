"""Discrete optimal-transport assignment helpers.

The mini-batch OT coupling used by `deltaflow.interpolants.ot.OTInterpolant`
and `deltaflow.trainer.coupling` reduces to a single reusable primitive, a
squared-L2 assignment between two equal-size batches. It lives here so the
interpolant, the training-time coupling, and the visualization examples can
share one implementation.

Exact optimal assignment (Hungarian algorithm) is used by default, since
``scipy`` is a core dependency. A deterministic greedy nearest-neighbour
fallback is used only if ``scipy`` is unavailable in the environment.
"""

import torch

__all__ = ["batch_ot_permutation"]


def batch_ot_permutation(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
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
