# Quickstart

Fit a toy flow-matching model and draw samples in a few lines.

```python
import torch
import torch.nn as nn
from deltaflow.core import BaseVelocityField
from deltaflow.losses import FlowMatchingLoss
from deltaflow.samplers import FlowSampler


class MLPVelocityField(BaseVelocityField):
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(), nn.Linear(hidden, dim)
        )

    def forward(self, x, t, **cond):
        return self.net(torch.cat([x, t.view(-1, 1)], dim=-1))


field = MLPVelocityField()
loss_fn = FlowMatchingLoss()
opt = torch.optim.Adam(field.parameters(), lr=1e-3)

data = torch.randn(2000, 2)
for _ in range(300):
    x1 = data[torch.randint(len(data), (128,))]
    loss = loss_fn(field, x1)          # (1)!
    opt.zero_grad()
    loss.backward()
    opt.step()

samples = FlowSampler(field).sample(torch.randn(1000, 2), n_steps=50)  # (2)!
```

1. `FlowMatchingLoss` samples a time `t`, builds `x_t` from the interpolant,
   and regresses `field(x_t, t)` onto the conditional target velocity.
2. `FlowSampler` integrates `dx/dt = field(x, t)` with Euler from `t=0`
   (noise) to `t=1` (data).

!!! tip "Time convention"
    Throughout DeltaFlow, `t=0` is **noise** and `t=1` is **data**.

## Where to next

- [Flow Matching](../guides/flow-matching.md), the objective in depth, plus
  interpolant choices.
- [Delta Alignment](../guides/delta-alignment.md), the guidance-representation
  pretraining loss.
- [Examples](../examples.md), a runnable, tiered curriculum in the repo.
