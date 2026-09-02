"""
10-sampling/02-equilibrium-matching: train an implicit energy landscape with
Equilibrium Matching and sample it by gradient descent, no ODE integration.

EqM keeps the straight-line path but regresses onto the energy-compatible
target ``c(t) * (x1 - x0)`` (via `EquilibriumInterpolant`), whose coefficient
vanishes at data so ground truths become minima of the learned landscape. The
field is noise-unconditional (it ignores ``t``), so sampling is plain gradient
descent ``x <- x + eta * f(x)`` with `EquilibriumSolver` (NAG-GD when
``momentum > 0``).

Run: python examples/10-sampling/02-equilibrium-matching/main.py
"""

import torch
import torch.nn as nn

from deltaflow.core.base import BaseVelocityField
from deltaflow.interpolants import EquilibriumInterpolant
from deltaflow.losses import ConditionalFlowMatchingLoss
from deltaflow.solvers import EquilibriumSolver


class EquilibriumField(BaseVelocityField):
    """Noise-unconditional gradient field ``f(x)`` (the time input is ignored)."""

    def __init__(self, dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **cond) -> torch.Tensor:
        return self.net(x)


def two_gaussians(n: int) -> torch.Tensor:
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
    idx = torch.randint(0, 2, (n,))
    return centers[idx] + 0.2 * torch.randn(n, 2)


def main():
    torch.manual_seed(0)
    field = EquilibriumField(dim=2)
    loss_fn = ConditionalFlowMatchingLoss(interpolant=EquilibriumInterpolant())
    opt = torch.optim.Adam(field.parameters(), lr=2e-3)

    for step in range(800):
        x1 = two_gaussians(256)
        loss = loss_fn(field, x1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"step={step:4d}  loss={loss.item():.4f}")

    # Optimisation-based sampling: gradient descent on the learned landscape.
    solver = EquilibriumSolver(field, step_size=0.05, momentum=0.3)
    samples = solver.sample(torch.randn(1000, 2), n_steps=200, show_progress=False)
    print("sample mean:", samples.mean(dim=0).tolist())
    print("sample std:", samples.std(dim=0).tolist())
    print("|x| near a mode:", samples.abs().mean(dim=0).tolist())


if __name__ == "__main__":
    main()
