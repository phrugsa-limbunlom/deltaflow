import torch

from deltaflow.losses import FlowMatchingLoss
from deltaflow.losses.delta_alignment import delta_alignment_loss
from deltaflow.models import MultiScaleProjector
from tests.conftest import DummyVelocityField


def test_flow_matching_loss_is_scalar_and_nonnegative():
    torch.manual_seed(0)
    model = DummyVelocityField(dim=4)
    loss_fn = FlowMatchingLoss()

    x1 = torch.randn(6, 4)
    loss = loss_fn(model, x1)

    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_delta_alignment_loss_zero_when_deltas_identical():
    torch.manual_seed(0)
    projector = MultiScaleProjector(feature_dims={"bottleneck": 16}, hidden_dim=8, out_dim=8)

    feats_u = {"bottleneck": torch.randn(2, 16, 4, 4)}
    feats_c = {"bottleneck": feats_u["bottleneck"] + torch.randn(2, 16, 4, 4)}

    # Identical deltas across both "views" => cosine similarity is exactly 1.
    loss = delta_alignment_loss(feats_u, feats_c, feats_u, feats_c, projector)

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5)
