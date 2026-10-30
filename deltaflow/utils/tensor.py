"""Small tensor-shape helpers shared by interpolants, solvers, and inverse problems."""

from typing import Union

import torch


def broadcast_time(t: Union[float, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Reshape a time (or any per-sample scalar) so it broadcasts against ``x``.

    - A per-sample tensor of shape ``(B,)`` becomes ``(B, 1, ..., 1)`` with
      ``x.dim()`` dimensions, e.g. ``(B,) -> (B, 1, 1, 1)`` for images.
    - A Python float or 0-d tensor becomes a 0-d tensor on ``x``'s device.
    """
    if not isinstance(t, torch.Tensor):
        return torch.tensor(t, device=x.device, dtype=x.dtype)
    if t.dim() == 0:
        return t
    return t.view(-1, *([1] * (x.dim() - 1)))


__all__ = ["broadcast_time"]
