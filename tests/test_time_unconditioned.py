import torch
import torch.nn as nn

from deltaflow.losses import TimeUnconditionedFlowMatchingLoss
from deltaflow.models import TimeBlindVelocityField, TinyVelocityField


def test_time_unconditioned_loss_never_passes_time():
    torch.manual_seed(0)
    calls = []

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x, **cond):
            calls.append(cond)
            return self.lin(x)

    loss = TimeUnconditionedFlowMatchingLoss()(Net(), torch.randn(6, 4))
    loss.backward()
    assert loss.dim() == 0 and calls == [{}]


def test_time_blind_field_ignores_t():
    field = TimeBlindVelocityField(nn.Linear(4, 4))
    x = torch.randn(3, 4)
    assert torch.equal(field(x, 0.1), field(x, 0.9))


def test_pass_time_uses_zero_time():
    torch.manual_seed(0)
    model = TinyVelocityField(channels=1, hidden=8)
    loss = TimeUnconditionedFlowMatchingLoss(pass_time=True)(model, torch.randn(2, 1, 8, 8))
    assert loss.dim() == 0
