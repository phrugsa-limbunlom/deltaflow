"""
20-training/04-active-flow-expansion: grow a flow's generable set with ActFlow.

A toy version of the 2D experiment in De Santi et al., "Active Flow Expansion
for Out-of-Distribution Discovery: from Theory to Molecules" (2026). The valid
design space is an annulus, but the pre-training data only covers one arc of
it. A flow fitted to that data can only generate the arc. ActFlow fine-tunes it
on self-generated samples labelled by a black-box verifier, steering sampling
toward the uncertain frontier of what the model can already reach, and the
generable set spreads around the ring.

The same loop with the tilt turned off (``beta=math.inf``) is the
self-training baseline (REC-F in the paper). Both runs start from the same
pre-trained weights and use the same budget.

Run: python examples/20-training/04-active-flow-expansion/main.py
"""

import copy
import math

import torch
import torch.nn as nn

from deltaflow.core import BaseVelocityField
from deltaflow.expansion import (
    ActFlow,
    FlowRepresentation,
    KernelUncertainty,
    generable_coverage,
)
from deltaflow.losses import FlowMatchingLoss, SignedReplayFlowMatchingLoss
from deltaflow.solvers import EulerSolver

R_IN, R_OUT = 1.5, 2.5
BOUNDS = (-3.5, 3.5)


def verifier(x: torch.Tensor) -> torch.Tensor:
    """Black-box validity oracle, the annulus R_IN <= |x| <= R_OUT."""
    r = x.norm(dim=-1)
    return (r >= R_IN) & (r <= R_OUT)


def arc_data(n: int, half_angle: float = math.pi / 6) -> torch.Tensor:
    """Pre-training data, only the arc |angle| <= half_angle of the annulus."""
    theta = (torch.rand(n) * 2 - 1) * half_angle
    r = R_IN + (R_OUT - R_IN) * torch.rand(n)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)


class MLPVelocityField(BaseVelocityField):
    def __init__(self, dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **cond) -> torch.Tensor:
        return self.net(torch.cat([x, t.view(-1, 1)], dim=-1))


def pretrain(steps: int = 1500) -> MLPVelocityField:
    field = MLPVelocityField()
    opt = torch.optim.Adam(field.parameters(), lr=2e-3)
    loss_fn = FlowMatchingLoss()
    data = arc_data(4000)
    for _ in range(steps):
        loss = loss_fn(field, data[torch.randint(len(data), (256,))])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return field


@torch.no_grad()
def evaluate(field: MLPVelocityField, n: int = 3000) -> dict:
    x = EulerSolver(field).sample(torch.randn(n, 2), n_steps=40, show_progress=False)
    return generable_coverage(x, verifier, bounds=BOUNDS, bins=50, tau=0.01)


def expand(field: MLPVelocityField, beta: float, rounds: int, label: str) -> dict:
    act = ActFlow(
        field,
        verifier,
        torch.optim.Adam(field.parameters(), lr=1e-3),
        sample_shape=(2,),
        # phi_s, the last hidden layer of the velocity field at noise level s = 0.9
        representation=FlowRepresentation(field, s=0.9, layer="net.3"),
        uncertainty=KernelUncertainty(kernel="linear", reg=1e-2),
        loss=SignedReplayFlowMatchingLoss(alpha=0.05, alpha_mode="grad_norm"),
        beta=beta,
        guidance_scale=1.0,
        n_steps=30,
        finetune_steps=60,
        batch_size=128,
    )
    for r in act.run(rounds, n_samples=128):
        if (r.round + 1) % 5 == 0:
            cov = evaluate(field)
            print(
                f"  {label:<13} round {r.round + 1:>2}  accept {r.acceptance_rate:5.2f}  "
                f"sigma {r.mean_uncertainty:6.3f}  coverage {cov['coverage']:5.2f}  "
                f"validity {cov['validity']:5.2f}"
            )
    return evaluate(field)


def main():
    torch.manual_seed(0)
    pretrained = pretrain()
    base = evaluate(pretrained)
    print(f"pre-trained   coverage {base['coverage']:5.2f}  validity {base['validity']:5.2f}")

    rounds = 20
    torch.manual_seed(1)
    baseline = expand(copy.deepcopy(pretrained), beta=math.inf, rounds=rounds, label="self-train")
    torch.manual_seed(1)
    actflow = expand(copy.deepcopy(pretrained), beta=0.1, rounds=rounds, label="ActFlow")

    print()
    print(
        f"final coverage, self-training {baseline['coverage']:.2f}, ActFlow {actflow['coverage']:.2f}"
    )


if __name__ == "__main__":
    main()
