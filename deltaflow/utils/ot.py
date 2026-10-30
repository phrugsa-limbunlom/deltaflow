"""Discrete optimal-transport assignment helpers.

The mini-batch OT coupling used by `deltaflow.interpolants.ot.OTInterpolant`
and `deltaflow.trainer.coupling` reduces to a single reusable primitive, a
squared-L2 assignment between two equal-size batches. It lives here so the
interpolant, the training-time coupling, and the visualization examples can
share one implementation.

Exact optimal assignment is computed with the Hungarian algorithm
(``scipy.optimize.linear_sum_assignment``).
"""

import torch
from scipy.optimize import linear_sum_assignment

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

    _, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    # Rows come back sorted, so col_ind[i] is the x1 index paired with x0[i].
    # Inverting it gives, for each x1[j], the x0 index assigned to it.
    col = torch.as_tensor(col_ind, dtype=torch.long, device=x0.device)
    return torch.argsort(col)
