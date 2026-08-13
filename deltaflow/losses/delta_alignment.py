"""
Delta alignment: a multi-scale, anatomy-cancelling guidance-alignment loss.

The core idea: for a conditionally-generated backbone, the guidance-difference
feature ``delta_h = h_cond - h_uncond`` isolates *what the conditioning
signal changed* at each hierarchy level, largely cancelling out the
anatomy-specific content that both the conditional and unconditional passes
share. Aligning ``delta_h`` across two augmented views of the same input
(rather than aligning the raw features) encourages the model to encode a
guidance representation that is consistent regardless of the anatomy it is
applied to.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.projector import MultiScaleProjector
from ..utils.numerical import clamp_cosine_similarity, clamp_loss, clamp_prediction, safe_normalize


def _cosine_dissimilarity(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine dissimilarity (``1 - cosine_sim``), averaged over the batch.

    Inputs are assumed to already be L2-normalized.
    """
    sim = (z1 * z2).sum(dim=-1)
    sim = clamp_cosine_similarity(sim)
    return (1.0 - sim).mean()


def delta_alignment_loss(
    feats_u1: Dict[str, torch.Tensor],
    feats_c1: Dict[str, torch.Tensor],
    feats_u2: Dict[str, torch.Tensor],
    feats_c2: Dict[str, torch.Tensor],
    projector: MultiScaleProjector,
) -> torch.Tensor:
    """Multi-scale alignment loss on guidance-difference embeddings.

    For each hierarchy level ``l`` and each of two augmented views ``i in {1, 2}``::

        delta_h_l_i = h_cond_i[l] - h_uncond_i[l]
        z_l_i       = projector_l(GAP(delta_h_l_i)), L2-normalized
        loss_l      = 1 - cos(z_l_1, z_l_2)

    Returns the average of ``loss_l`` over all levels present in every
    feature dict and in ``projector``.
    """
    layer_set = projector.layer_set
    total_loss = torch.tensor(0.0, device=next(iter(feats_u1.values())).device)
    num_layers = 0

    for layer in layer_set:
        if not all(layer in feats for feats in (feats_u1, feats_c1, feats_u2, feats_c2)):
            continue
        if layer not in projector.projectors:
            continue

        delta_h1 = feats_c1[layer] - feats_u1[layer]
        delta_h2 = feats_c2[layer] - feats_u2[layer]

        z1 = projector.pool_feature(delta_h1)
        z2 = projector.pool_feature(delta_h2)

        z1 = projector.projectors[layer](z1)
        z2 = projector.projectors[layer](z2)

        z1 = safe_normalize(z1, dim=-1)
        z2 = safe_normalize(z2, dim=-1)

        layer_loss = clamp_loss(_cosine_dissimilarity(z1, z2), max_val=10.0)
        total_loss = total_loss + layer_loss
        num_layers += 1

    if num_layers == 0:
        return total_loss
    return total_loss / num_layers


def flow_matching_velocity_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """Standard MSE velocity-regression loss, with prediction clamping."""
    v_pred = clamp_prediction(v_pred)
    return clamp_loss(F.mse_loss(v_pred, v_target))


class DeltaAlignmentLoss(nn.Module):
    r"""Combined flow-matching and delta-alignment loss.

    The total objective linearly combines the velocity-regression term with
    the multi-scale guidance-alignment term,

    \[
    \mathcal{L} = \lambda_\text{flow}\,\mathcal{L}_\text{flow}
                + \lambda_\text{align}\,\mathcal{L}_\text{align},
    \]

    where, at each hierarchy level \(l\), the alignment term compares the
    guidance-difference embeddings \(z^{(i)}_l = g_l\bigl(\text{GAP}(\Delta
    h^{(i)}_l)\bigr)\) of two augmented views \(i \in \{1, 2\}\) via a cosine
    dissimilarity,

    \[
    \mathcal{L}_\text{align} = \frac{1}{L}\sum_{l=1}^{L}
        \Bigl(1 - \cos\bigl(z^{(1)}_l, z^{(2)}_l\bigr)\Bigr),
    \qquad
    \Delta h_l = h^\text{cond}_l - h^\text{uncond}_l.
    \]

    During the alignment phase, \(\mathcal{L}_\text{flow}\) is computed over
    all four velocity predictions (two views by two conditioning modes) to
    preserve both the guided and unguided generative pathways while the
    alignment term shapes the guidance representation.
    """

    def __init__(
        self,
        projector: MultiScaleProjector,
        lambda_flow: float = 1.0,
        lambda_align: float = 5.0,
    ):
        super().__init__()
        self.projector = projector
        self.lambda_flow = lambda_flow
        self.lambda_align = lambda_align

    def forward(
        self,
        v_c1: torch.Tensor, v_u1: torch.Tensor, target_v1: torch.Tensor,
        v_c2: torch.Tensor, v_u2: torch.Tensor, target_v2: torch.Tensor,
        feats_u1: Dict[str, torch.Tensor],
        feats_c1: Dict[str, torch.Tensor],
        feats_u2: Dict[str, torch.Tensor],
        feats_c2: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Flow-matching loss over both views and both conditioning modes.
        loss_flow_c = (
            flow_matching_velocity_loss(v_c1, target_v1) + flow_matching_velocity_loss(v_c2, target_v2)
        ) / 2
        loss_flow_u = (
            flow_matching_velocity_loss(v_u1, target_v1) + flow_matching_velocity_loss(v_u2, target_v2)
        ) / 2
        loss_flow = loss_flow_c + loss_flow_u

        # Multi-scale delta-alignment loss.
        loss_align = delta_alignment_loss(feats_u1, feats_c1, feats_u2, feats_c2, self.projector)

        total = self.lambda_flow * loss_flow + self.lambda_align * loss_align

        loss_dict = {
            "loss_total": total.detach(),
            "loss_flow": loss_flow.detach(),
            "loss_align": loss_align.detach(),
        }
        return total, loss_dict
