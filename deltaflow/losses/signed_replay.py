"""Signed replay flow matching, ActFlow's continued pre-training objective.

After each query round the verifier splits the self-generated samples into
an accepted buffer ``D+`` and a rejected buffer ``D-``. ActFlow fine-tunes on
both,

    L_signed(theta) = L_{D+}(theta) - alpha_t * L_{D-}(theta),

where ``L_S`` is the ordinary flow-matching loss on ``S``. Accepted samples
attract probability mass (standard flow matching), rejected samples repel it.
This is the retain-minus-forget objective of SISS data unlearning (Alberti et
al., ICLR 2025) with the forget set produced online by the verifier, in its
non-importance-sampled form (separate minibatches from each buffer).

The repulsive term is unbounded below, so its magnitude must be controlled.
Following the paper, ``alpha_t`` can be calibrated per step as a gradient-norm
fraction, ``alpha_t = alpha * ||grad L+|| / ||grad L-||``, which makes the
rejected-sample update a fixed fraction ``alpha`` of the accepted one. With
``alpha = 0`` the loss is plain flow matching on verifier-accepted samples,
which the paper reports is often sufficient.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..core.base_loss import BaseLoss
from ..utils.numerical import get_eps
from .conditional_flow_matching import ConditionalFlowMatchingLoss


class SignedReplayFlowMatchingLoss(BaseLoss):
    r"""Accept-attract, reject-repel flow matching on verifier-labelled replay.

    \[
    \mathcal{L}^{\text{signed}}_t(\theta)
        = \hat{\mathcal{L}}^{+}_t(\theta) - \alpha_t\,\hat{\mathcal{L}}^{-}_t(\theta),
    \qquad
    \hat{\mathcal{L}}^{\pm}_t(\theta)
        = \frac{1}{|U^{\pm}_t|} \sum_{x \in U^{\pm}_t} \ell(\theta; x),
    \]

    where \(U^{+}_t\) and \(U^{-}_t\) are minibatches from the accepted and
    rejected buffers and \(\ell\) is the ``base_loss`` (conditional flow
    matching by default). Its gradient is the signed update
    \(g_t = \nabla \hat{\mathcal{L}}^{+}_t - \alpha_t \nabla
    \hat{\mathcal{L}}^{-}_t\).

    With ``alpha_mode="grad_norm"`` the weight is recalibrated every call,

    \[
    \alpha_t = \alpha\,
        \frac{\lVert \nabla_\theta \hat{\mathcal{L}}^{+}_t \rVert_2}
             {\lVert \nabla_\theta \hat{\mathcal{L}}^{-}_t \rVert_2},
    \]

    so that \(\lVert \alpha_t \nabla \hat{\mathcal{L}}^{-}_t \rVert = \alpha
    \lVert \nabla \hat{\mathcal{L}}^{+}_t \rVert\). This costs two extra
    backward passes and requires ``model`` to be an ``nn.Module``. The
    calibrated \(\alpha_t\) is detached and stored in ``last_alpha``.

    References:
        De Santi et al., "Active Flow Expansion for Out-of-Distribution
        Discovery: from Theory to Molecules" (2026), Apx. D.1,
        https://arxiv.org/abs/2606.08802. Alberti et al., "Data Unlearning in
        Diffusion Models" (ICLR 2025).

    Args:
        base_loss: per-sample objective \(\ell\), any callable
            ``base_loss(model, x1, **cond) -> scalar``. Defaults to
            `ConditionalFlowMatchingLoss`.
        alpha: target weight. A loss weight for ``"fixed"``, a gradient-norm
            fraction in \([0, 1]\) for ``"grad_norm"``. ``0`` disables the
            repulsive term.
        alpha_mode: ``"fixed"`` or ``"grad_norm"``.
    """

    def __init__(
        self,
        base_loss: Optional[BaseLoss] = None,
        alpha: float = 0.0,
        alpha_mode: str = "fixed",
    ):
        if alpha_mode not in ("fixed", "grad_norm"):
            raise ValueError(f"alpha_mode must be 'fixed' or 'grad_norm', got {alpha_mode!r}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.base_loss = base_loss or ConditionalFlowMatchingLoss()
        self.alpha = float(alpha)
        self.alpha_mode = alpha_mode
        self.last_alpha: float = 0.0

    @staticmethod
    def _grad_norm(loss: torch.Tensor, params) -> torch.Tensor:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        sq = [g.detach().pow(2).sum() for g in grads if g is not None]
        if not sq:
            return loss.new_zeros(())
        return torch.stack(sq).sum().sqrt()

    def __call__(
        self,
        model,
        x_pos: torch.Tensor,
        x_neg: Optional[torch.Tensor] = None,
        neg_cond: Optional[Dict[str, Any]] = None,
        **cond,
    ) -> torch.Tensor:
        r"""Return the signed loss.

        Args:
            model: velocity field.
            x_pos: accepted minibatch \(U^{+}_t\).
            x_neg: rejected minibatch \(U^{-}_t\), or ``None``.
            neg_cond: conditioning for ``x_neg`` if it differs from ``cond``
                (e.g. per-sample labels of a different batch). Defaults to
                ``cond``.
            **cond: conditioning forwarded unchanged to the model for
                ``x_pos``.
        """
        loss_pos = self.base_loss(model, x_pos, **cond)
        if x_neg is None or x_neg.shape[0] == 0 or self.alpha == 0.0:
            self.last_alpha = 0.0
            return loss_pos

        loss_neg = self.base_loss(model, x_neg, **(cond if neg_cond is None else neg_cond))
        if self.alpha_mode == "fixed":
            alpha_t = loss_pos.new_tensor(self.alpha)
        else:
            if not isinstance(model, nn.Module):
                raise TypeError("alpha_mode='grad_norm' requires model to be an nn.Module")
            params = [p for p in model.parameters() if p.requires_grad]
            g_pos = self._grad_norm(loss_pos, params)
            g_neg = self._grad_norm(loss_neg, params)
            alpha_t = self.alpha * g_pos / g_neg.clamp_min(get_eps(loss_pos.dtype))
        alpha_t = alpha_t.detach()
        self.last_alpha = float(alpha_t)
        return loss_pos - alpha_t * loss_neg


__all__ = ["SignedReplayFlowMatchingLoss"]
