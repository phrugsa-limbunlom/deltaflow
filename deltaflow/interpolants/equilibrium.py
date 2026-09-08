"""Equilibrium Matching probability path (energy-compatible target)."""

from typing import Optional, Tuple

import torch

from ..core.base_equilibrium_interpolant import BaseEquilibriumInterpolant


class EquilibriumInterpolant(BaseEquilibriumInterpolant):
    r"""Straight-line path with an energy-compatible (equilibrium) target.

    Equilibrium Matching (EqM) keeps the rectified-flow straight-line path but
    parametrises it by an interpolation coefficient \(\gamma \in [0, 1]\), a
    noise level rather than a dynamical time. The learned field is trained to be
    *time-invariant* (it approximates an equilibrium gradient), so \(\gamma\)
    only indexes where along the noise-to-data path a training point sits, it is
    not integrated over. In the paper's notation, for data \(x\) and Gaussian
    noise \(\epsilon\),

    \[
    x_\gamma = \gamma\,x + (1 - \gamma)\,\epsilon,
    \]

    which, with DeltaFlow's \((x_0, x_1)\) = (noise, data) naming, is the same
    straight line

    \[
    x_\gamma = (1 - \gamma)\,x_0 + \gamma\,x_1, \qquad \gamma \in [0, 1].
    \]

    EqM reshapes the regression target so the learned field becomes the
    gradient of an implicit energy landscape rather than a time-conditional
    velocity. Instead of regressing onto the constant displacement
    \(x_1 - x_0\), it regresses onto the scaled displacement (the paper's
    \((x - \epsilon)\,c(\gamma)\)),

    \[
    u_\gamma = c(\gamma)\,(x_1 - x_0),
    \]

    where \(c(\gamma)\) is the equilibrium coefficient. The key design
    constraint is \(c(1) = 0\): the target vanishes at data, so ground-truth
    samples become stationary points (local minima) of the landscape whose
    gradient the field learns. Away from data the coefficient is held on a
    constant plateau, so the field points from noise toward data with roughly
    constant magnitude.

    Concretely the coefficient is the minimum of two lines, rescaled by
    ``scale``,

    \[
    c(\gamma) = \text{scale}\cdot\min\!\Bigl(
        \text{start} - \tfrac{\text{start} - 1}{p}\,\gamma,\;
        \tfrac{1 - \gamma}{1 - p}
    \Bigr),
    \]

    with plateau fraction \(p\) (``plateau``). With the defaults
    (``start = 1``, ``plateau = 0.8``, ``scale = 4``) the first line is flat at
    \(1\), so \(c(\gamma) = 4\,\min(1, 5(1 - \gamma))\): a plateau of \(4\) for
    \(\gamma \le 0.8\) that then ramps linearly down to \(0\) at \(\gamma = 1\).

    **Interpolation coefficient.** \(\gamma = 0\) is noise (\(x_\gamma = x_0\))
    and \(\gamma = 1\) is data (\(x_\gamma = x_1\)). This interpolant subclasses
    `BaseEquilibriumInterpolant`, whose ``interpolate`` takes ``gamma`` rather
    than the dynamical time ``t`` of `BaseInterpolant`, because \(\gamma\) is a
    noise level and not a time to be integrated.

    **Coupling.** This interpolant defines only the *path* and *target*. It is
    agnostic to how \((x_0, x_1)\) pairs are formed. If ``x0`` is not supplied
    it is drawn from \(\mathcal{N}(0, I)\) independently of ``x1``. Pass an
    `OTCoupling` on the training side for
    straighter, OT-coupled displacements.

    **Sampling.** A field trained with this target is *not* integrated over a
    fixed time horizon. Because it approximates an equilibrium gradient it is
    sampled by optimisation (gradient descent on the landscape), for which see
    `EquilibriumSolver`.

    References:
        Wang and Du, "Equilibrium Matching: Generative Modeling with Implicit
        Energy-Based Models" (2025), https://arxiv.org/abs/2510.02300.

    Args:
        plateau: fraction \(p \in (0, 1)\) of the path spent before the
            coefficient starts ramping down to zero. The ramp occupies the
            final ``1 - plateau`` of the \([0, 1]\) interval.
        scale: overall magnitude applied to the coefficient. Sets the typical
            norm of the learned equilibrium gradient away from data.
        start: value of the first (upper) line at ``gamma=0``. Left at ``1``
            this line stays flat so the coefficient plateaus at ``scale``,
            values other than ``1`` tilt the plateau.
    """

    def __init__(self, plateau: float = 0.8, scale: float = 4.0, start: float = 1.0):
        if not 0.0 < plateau < 1.0:
            raise ValueError(f"plateau must lie in (0, 1), got {plateau!r}")
        self.plateau = plateau
        self.scale = scale
        self.start = start

    def equilibrium_coefficient(self, gamma: torch.Tensor) -> torch.Tensor:
        r"""Return the scalar coefficient \(c(\gamma)\) applied to \(x_1 - x_0\).

        ``gamma`` is the interpolation coefficient (noise level), \(0\) at noise
        and \(1\) at data, not a dynamical time.
        """
        line_plateau = self.start - (self.start - 1.0) / self.plateau * gamma
        line_ramp = (1.0 - gamma) / (1.0 - self.plateau)
        return self.scale * torch.minimum(line_plateau, line_ramp)

    def interpolate(
        self, x1: torch.Tensor, gamma: torch.Tensor, x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x0 is None:
            x0 = torch.randn_like(x1)
        gamma = gamma.view(-1, *([1] * (x1.dim() - 1)))
        x_gamma = (1 - gamma) * x0 + gamma * x1
        target_v = self.equilibrium_coefficient(gamma) * (x1 - x0)
        return x_gamma, target_v
