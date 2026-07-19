"""
10-sampling/01-euler-flow: sample a toy Gaussian-mixture target with
`FlowSampler`, after fitting a tiny MLP velocity field with flow matching.

Run: python examples/10-sampling/01-euler-flow/main.py
"""

import torch
import torch.nn as nn

from deltaflow.core.base import BaseVelocityField
from deltaflow.losses import FlowMatchingLoss
from deltaflow.samplers import FlowSampler


class MLPVelocityField(BaseVelocityField):
    def __init__(self, dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **cond) -> torch.Tensor:
        t_ = t.view(-1, 1)
        return self.net(torch.cat([x, t_], dim=-1))


def two_gaussians(n: int) -> torch.Tensor:
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
    idx = torch.randint(0, 2, (n,))
    return centers[idx] + 0.2 * torch.randn(n, 2)


def main():
    torch.manual_seed(0)
    field = MLPVelocityField(dim=2)
    loss_fn = FlowMatchingLoss()
    opt = torch.optim.Adam(field.parameters(), lr=2e-3)

    for step in range(500):
        x1 = two_gaussians(256)
        loss = loss_fn(field, x1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step={step:4d}  loss={loss.item():.4f}")

    sampler = FlowSampler(field)
    samples = sampler.sample(torch.randn(1000, 2), n_steps=50, show_progress=False)
    print("sample mean:", samples.mean(dim=0).tolist())
    print("sample std:", samples.std(dim=0).tolist())


if __name__ == "__main__":
    main()
