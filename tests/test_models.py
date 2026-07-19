import torch

from deltaflow.models import EMA, MultiScaleProjector


def test_multiscale_projector_project_delta_h_shapes():
    projector = MultiScaleProjector(
        feature_dims={"enc_1_4": 32, "bottleneck": 64},
        hidden_dim=16,
        out_dim=8,
    )

    feats_cond = {
        "enc_1_4": torch.randn(2, 32, 8, 8),
        "bottleneck": torch.randn(2, 64, 2, 2),
    }
    feats_uncond = {
        "enc_1_4": torch.randn(2, 32, 8, 8),
        "bottleneck": torch.randn(2, 64, 2, 2),
    }

    projected = projector.project_delta_h(feats_cond, feats_uncond)

    assert set(projected.keys()) == {"enc_1_4", "bottleneck"}
    for z in projected.values():
        assert z.shape == (2, 8)
        norms = z.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_ema_update_moves_toward_new_weights():
    ema = EMA(beta=0.9)
    old = torch.zeros(4)
    new = torch.ones(4)

    updated = ema.update_average(old, new)

    assert torch.allclose(updated, torch.full((4,), 0.1))
