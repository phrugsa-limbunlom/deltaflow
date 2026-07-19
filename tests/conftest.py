"""Shared test fixtures and a tiny dummy velocity field."""

import torch
import torch.nn as nn

from deltaflow.core.base import BaseVelocityField


class DummyVelocityField(BaseVelocityField):
    """Minimal velocity field for tests: a per-pixel linear map, time-conditioned."""

    def __init__(self, dim: int = 2):
        super().__init__()
        self.net = nn.Linear(dim + 1, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **cond) -> torch.Tensor:
        t_ = t.view(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_], dim=-1))
