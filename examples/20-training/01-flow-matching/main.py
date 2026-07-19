"""
20-training/01-flow-matching: the training loop from
examples/10-sampling/01-euler-flow, isolated to show the objective on its own.

Run: python examples/20-training/01-flow-matching/main.py
"""

import torch
import torch.nn as nn

from deltaflow.core.base import BaseVelocityField
from deltaflow.losses import FlowMatchingLoss


class MLPVelocityField(BaseVelocityField):
    def __init__(self, dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **cond) -> torch.Tensor:
        return self.net(torch.cat([x, t.view(-1, 1)], dim=-1))


def main():
    torch.manual_seed(0)
    field = MLPVelocityField(dim=2)
    loss_fn = FlowMatchingLoss(loss_type="l2")
    opt = torch.optim.Adam(field.parameters(), lr=1e-3)

    data = torch.randn(2000, 2) * torch.tensor([3.0, 1.0])

    for step in range(300):
        x1 = data[torch.randint(len(data), (128,))]
        loss = loss_fn(field, x1)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"final loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
