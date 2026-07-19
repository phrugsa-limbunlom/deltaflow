# Getting Started

## Install

```bash
pip install -e ".[dev]"
```

## Fit a toy flow-matching model

```python
import torch
import torch.nn as nn
from deltaflow.core.base import BaseVelocityField
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
    loss = loss_fn(field, x1)
    opt.zero_grad(); loss.backward(); opt.step()

samples = FlowSampler(field).sample(torch.randn(1000, 2), n_steps=50)
```

See `examples/20-training/02-delta-alignment/main.py` in the repository
for the guidance-alignment loss on synthetic multi-level features.
