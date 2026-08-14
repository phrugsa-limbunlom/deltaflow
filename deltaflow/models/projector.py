"""
Multi-scale projection heads for DeltaFlow's guidance-alignment loss.

Each backbone hierarchy level gets its own projector: global-average-pool ->
MLP -> L2-normalize. The projected, normalized embeddings are what
`DeltaAlignmentLoss` compares.
"""

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_LAYER_SET = ["enc_1_4", "enc_1_8", "bottleneck", "dec_1_8"]


class ProjectorHead(nn.Module):
    """Two-layer MLP projector (``in_dim -> hidden_dim -> out_dim``, GELU)."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MultiScaleProjector(nn.Module):
    """Collection of per-level projection heads.

    Args:
        feature_dims: ``{layer_name: channel_count}``, e.g.
            ``{"enc_1_4": 256, "bottleneck": 1024}``. Only layers present in
            ``layer_set`` (default: four backbone hierarchy levels) get a
            projector.
        hidden_dim: hidden width shared by every per-level projector.
        out_dim: shared output embedding dimension.
        layer_set: which layer names to project, defaults to
            ``["enc_1_4", "enc_1_8", "bottleneck", "dec_1_8"]``.
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        hidden_dim: int = 512,
        out_dim: int = 256,
        layer_set: Sequence[str] = _DEFAULT_LAYER_SET,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layer_set = list(layer_set)

        self.projectors = nn.ModuleDict(
            {
                layer: ProjectorHead(in_dim=feature_dims[layer], hidden_dim=hidden_dim, out_dim=out_dim)
                for layer in self.layer_set
                if layer in feature_dims
            }
        )

    @staticmethod
    def pool_feature(feature_map: torch.Tensor) -> torch.Tensor:
        """Global average pooling: ``[B, C, H, W] -> [B, C]``."""
        return feature_map.mean(dim=(2, 3))

    def project_delta_h(
        self,
        feats_cond: Dict[str, torch.Tensor],
        feats_uncond: Dict[str, torch.Tensor],
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute ``delta_h = h_cond - h_uncond`` and project each level.

        Returns a dict of (optionally L2-normalized) embeddings, one per
        layer present in both feature dicts and in ``self.projectors``.
        """
        projected = {}
        for layer in self.layer_set:
            if layer not in feats_cond or layer not in feats_uncond:
                continue
            if layer not in self.projectors:
                continue
            delta_h = feats_cond[layer] - feats_uncond[layer]
            pooled = self.pool_feature(delta_h)
            z = self.projectors[layer](pooled)
            if normalize:
                z = F.normalize(z, dim=-1)
            projected[layer] = z
        return projected
