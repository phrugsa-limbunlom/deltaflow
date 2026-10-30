"""Element-wise regression reductions shared by the flow-matching style losses."""

from typing import Callable, Dict

import torch
import torch.nn.functional as F

_REGRESSION_LOSSES: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "l1": F.l1_loss,
    "l2": F.mse_loss,
    "huber": F.smooth_l1_loss,
}


def regression_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Mean regression loss of ``pred`` against ``target`` for ``loss_type``."""
    loss_fn = _REGRESSION_LOSSES.get(loss_type)
    if loss_fn is None:
        raise NotImplementedError(f"Unknown loss_type: {loss_type!r}")
    return loss_fn(pred, target)
