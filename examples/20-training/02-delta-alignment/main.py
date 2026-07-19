"""
20-training/02-delta-alignment: fit `DeltaAlignmentLoss` on synthetic
multi-level feature maps, to show the API without needing a full
conditional backbone.

Run: python examples/20-training/02-delta-alignment/main.py
"""

import torch

from deltaflow.losses.delta_alignment import DeltaAlignmentLoss
from deltaflow.models import MultiScaleProjector


def random_feats(batch: int, dims: dict, spatial: int = 4) -> dict:
    return {name: torch.randn(batch, c, spatial, spatial) for name, c in dims.items()}


def main():
    torch.manual_seed(0)
    feature_dims = {"enc_1_4": 32, "enc_1_8": 64, "bottleneck": 128, "dec_1_8": 64}
    projector = MultiScaleProjector(feature_dims, hidden_dim=64, out_dim=32)
    loss_fn = DeltaAlignmentLoss(projector, lambda_flow=1.0, lambda_align=5.0)

    opt = torch.optim.Adam(projector.parameters(), lr=1e-3)

    batch = 4
    # Two augmented "views" of the same batch, each with a conditional and
    # unconditional feature pass at every hierarchy level.
    feats_u1 = random_feats(batch, feature_dims)
    feats_c1 = {k: v + torch.randn_like(v) for k, v in feats_u1.items()}
    feats_u2 = random_feats(batch, feature_dims)
    feats_c2 = {k: v + torch.randn_like(v) for k, v in feats_u2.items()}

    v_c1 = v_u1 = v_c2 = v_u2 = target_v1 = target_v2 = torch.zeros(batch, 2)

    for step in range(50):
        total, loss_dict = loss_fn(
            v_c1, v_u1, target_v1,
            v_c2, v_u2, target_v2,
            feats_u1, feats_c1, feats_u2, feats_c2,
        )
        opt.zero_grad()
        total.backward(retain_graph=True)
        opt.step()
        if step % 10 == 0:
            print(f"step={step:3d}  {loss_dict}")


if __name__ == "__main__":
    main()
