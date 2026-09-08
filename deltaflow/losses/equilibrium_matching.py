"""Equilibrium Matching loss (energy-compatible target, time-invariant field)."""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from ..core.base_equilibrium_interpolant import BaseEquilibriumInterpolant
from ..core.base_loss import BaseLoss
from ..interpolants.equilibrium import EquilibriumInterpolant

if TYPE_CHECKING:
    from ..trainer.coupling import BaseCoupling


class EquilibriumMatchingLoss(BaseLoss):
    r"""Regress a time-invariant field onto the Equilibrium Matching target.

    Equilibrium Matching (EqM) trains a field \(f_\theta\) to match an
    energy-compatible target along the straight noise-to-data path. For an
    interpolation coefficient \(\gamma \in [0, 1]\) (a noise level), noise
    \(x_0\) and data \(x_1\), the corrupted sample and target are produced by
    an `EquilibriumInterpolant`,

    \[
    x_\gamma = (1 - \gamma)\,x_0 + \gamma\,x_1, \qquad
    u_\gamma = c(\gamma)\,(x_1 - x_0),
    \]

    and the objective is (the paper's Eq. 3, up to the target-sign convention
    below)

    \[
    \mathcal{L}_{\text{EqM}} =
        \mathbb{E}_{\gamma,\, x_0,\, x_1}
        \bigl\| f_\theta(x_\gamma) - u_\gamma \bigr\|^2 .
    \]

    **\(\gamma\) is implicit, and there is no time at all.** Unlike flow
    matching, where the model is conditioned on time \(t\), an EqM field is
    time-invariant and noise-unconditional, \(f_\theta(x)\). The coefficient
    \(\gamma\) is *not seen by the model*, and neither is any surrogate time.
    This loss therefore queries the model as ``model(x, **cond)``, passing no
    time (and no \(\gamma\)) at all, matching `BaseEquilibriumField`.

    **Target sign.** The interpolant returns \(u_\gamma = c(\gamma)(x_1 - x_0)\)
    (data minus noise), so the field points from noise toward data and is
    sampled by gradient *ascent* \(x \leftarrow x + \eta f(x)\) in
    `EquilibriumSolver`. This matches the official EqM code. The
    paper writes the mirror-image \((\epsilon - x)c(\gamma)\) with descent, the
    two conventions are equivalent under a global sign flip.

    References:
        Wang and Du, "Equilibrium Matching: Generative Modeling with Implicit
        Energy-Based Models" (2025), https://arxiv.org/abs/2510.02300.

    Args:
        interpolant: the energy-compatible path. Defaults to
            `EquilibriumInterpolant`.
        coupling: optional train-time coupling that produces \((x_0, x_1)\)
            pairs from a batch of \(x_1\). See
            `deltaflow.trainer.coupling`.
        loss_type: one of ``"l2"``, ``"l1"``, ``"huber"``.
    """

    def __init__(
        self,
        interpolant: Optional[BaseEquilibriumInterpolant] = None,
        coupling: Optional["BaseCoupling"] = None,
        loss_type: str = "l2",
    ):
        self.interpolant = interpolant or EquilibriumInterpolant()
        self.coupling = coupling
        self.loss_type = loss_type

    def _reduce(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(target, pred)
        if self.loss_type == "l2":
            return F.mse_loss(target, pred)
        if self.loss_type == "huber":
            return F.smooth_l1_loss(target, pred)
        raise NotImplementedError(f"Unknown loss_type: {self.loss_type!r}")

    def __call__(self, model, x1: torch.Tensor, **cond) -> torch.Tensor:
        if self.coupling is not None:
            x0, x1 = self.coupling.sample_pair(x1)
        else:
            x0 = None
        gamma = torch.rand(x1.shape[0], device=x1.device)
        x_gamma, target = self.interpolant.interpolate(x1, gamma, x0=x0)
        # gamma is implicit and there is no time: the field is queried as f(x).
        pred = model(x_gamma, **cond)
        return self._reduce(target, pred)


__all__ = ["EquilibriumMatchingLoss"]
