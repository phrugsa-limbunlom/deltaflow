"""Tests for Equilibrium Matching: the energy-compatible interpolant and the
gradient-descent sampler."""

import torch

from deltaflow.core import (
    BaseEquilibriumField,
    BaseEquilibriumInterpolant,
    BaseInterpolant,
)
from deltaflow.interpolants import EquilibriumInterpolant, LinearInterpolant
from deltaflow.losses import EquilibriumMatchingLoss
from deltaflow.solvers import EquilibriumSolver, GradientDescentSolver


def test_equilibrium_coefficient_plateau_and_vanishes_at_data():
    """Defaults give a plateau of 4 up to gamma=0.8, then a linear ramp to 0 at gamma=1."""
    interp = EquilibriumInterpolant()  # plateau=0.8, scale=4, start=1
    gamma = torch.tensor([0.0, 0.4, 0.8, 0.9, 1.0])
    c = interp.equilibrium_coefficient(gamma)

    # Plateau region: constant scale.
    assert torch.allclose(c[:3], torch.full((3,), 4.0), atol=1e-6)
    # Ramp region: 4 * 5 * (1 - gamma).
    assert torch.allclose(c[3], torch.tensor(2.0), atol=1e-6)
    # Target must vanish at data so ground truths are stationary points.
    assert torch.allclose(c[4], torch.tensor(0.0), atol=1e-6)


def test_equilibrium_target_is_scaled_displacement():
    torch.manual_seed(0)
    interp = EquilibriumInterpolant()
    x1 = torch.randn(5, 3, 4, 4)
    x0 = torch.randn(5, 3, 4, 4)
    gamma = torch.rand(5)

    x_gamma, target = interp.interpolate(x1, gamma, x0=x0.clone())
    c = interp.equilibrium_coefficient(gamma).view(-1, 1, 1, 1)

    # Path is the plain straight line, target is c(gamma) * (x1 - x0).
    g_ = gamma.view(-1, 1, 1, 1)
    assert torch.allclose(x_gamma, (1 - g_) * x0 + g_ * x1, atol=1e-6)
    assert torch.allclose(target, c * (x1 - x0), atol=1e-6)


def test_equilibrium_target_zero_at_data():
    torch.manual_seed(0)
    interp = EquilibriumInterpolant()
    x1 = torch.randn(4, 2)
    x0 = torch.randn(4, 2)

    _, target = interp.interpolate(x1, torch.ones(4), x0=x0.clone())
    assert torch.allclose(target, torch.zeros_like(target), atol=1e-6)


def test_equilibrium_matches_scaled_linear_on_plateau():
    """On the plateau the EqM target is exactly the flow-matching target * scale."""
    torch.manual_seed(0)
    x1 = torch.randn(6, 4)
    x0 = torch.randn(6, 4)
    gamma = torch.full((6,), 0.3)  # inside the plateau

    eqm = EquilibriumInterpolant(scale=4.0)
    linear = LinearInterpolant()
    _, u_eqm = eqm.interpolate(x1, gamma, x0=x0.clone())
    _, u_lin = linear.interpolate(x1, gamma, x0=x0.clone())

    assert torch.allclose(u_eqm, 4.0 * u_lin, atol=1e-6)


def test_equilibrium_interpolant_is_gamma_based_and_decoupled():
    """EqM uses the gamma-based base, decoupled from the t-based BaseInterpolant."""
    interp = EquilibriumInterpolant()
    assert isinstance(interp, BaseEquilibriumInterpolant)
    assert not isinstance(interp, BaseInterpolant)

    # The interpolate signature exposes gamma, not t.
    x1 = torch.randn(4, 2)
    x_gamma, target = interp.interpolate(x1, gamma=torch.ones(4))
    assert x_gamma.shape == x1.shape
    assert torch.allclose(target, torch.zeros_like(target), atol=1e-6)


def test_equilibrium_matching_loss_calls_model_without_time():
    """The EqM loss must query the field as f(x): no time, no gamma."""

    seen_calls = []

    class RecordingField:
        def __init__(self):
            self.model = self

        def __call__(self, x, *args, **cond):
            seen_calls.append((args, cond))
            return torch.zeros_like(x)

    torch.manual_seed(0)
    field = RecordingField()
    loss_fn = EquilibriumMatchingLoss()
    x1 = torch.randn(16, 3)

    loss_fn(field, x1)

    assert len(seen_calls) == 1
    args, cond = seen_calls[0]
    # No time (or gamma) is passed positionally, and no extra conditioning.
    assert args == ()
    assert cond == {}


def test_equilibrium_matching_loss_trains_a_time_invariant_field():
    """A time-free field should receive gradients from the loss."""

    torch.manual_seed(0)

    class TinyField(BaseEquilibriumField):
        def __init__(self, dim: int):
            super().__init__()
            self.net = torch.nn.Linear(dim, dim)

        def forward(self, x, **cond):
            return self.net(x)

    model = TinyField(dim=4)
    loss_fn = EquilibriumMatchingLoss(interpolant=EquilibriumInterpolant())
    x1 = torch.randn(8, 4)

    loss = loss_fn(model, x1)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_equilibrium_solver_shape_and_finite():
    torch.manual_seed(0)

    class TinyField(BaseEquilibriumField):
        def __init__(self, dim: int):
            super().__init__()
            self.net = torch.nn.Linear(dim, dim)

        def forward(self, x, **cond):
            return self.net(x)

    model = TinyField(dim=4)
    x = torch.randn(3, 4)

    out = EquilibriumSolver(model, step_size=0.02).sample(x, n_steps=10, show_progress=False)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_equilibrium_solver_descends_toward_data_fixed_point():
    """A field pointing toward a fixed point x* should converge to it under GD."""

    class TowardPoint:
        # f(x) = x* - x, whose only stationary point is x*. Gradient descent
        # x <- x + eta f(x) contracts toward x* for small eta.
        def __init__(self, target):
            self.target = target
            self.model = self

        def __call__(self, x, **cond):
            return self.target - x

    target = torch.tensor([[3.0, -2.0]])
    field = TowardPoint(target)
    x0 = torch.zeros(1, 2)

    out = EquilibriumSolver(field, step_size=0.2).sample(
        x0.clone(), n_steps=200, show_progress=False
    )
    assert torch.allclose(out, target, atol=1e-3)


def test_nag_gd_converges_to_fixed_point():
    """NAG-GD (momentum > 0) must still converge to the landscape's minimum."""

    class TowardPoint:
        def __init__(self, target):
            self.target = target
            self.model = self

        def __call__(self, x, **cond):
            return self.target - x

    target = torch.tensor([[5.0]])
    x0 = torch.zeros(1, 1)

    nag = EquilibriumSolver(TowardPoint(target), step_size=0.1, momentum=0.3)
    out_nag = nag.sample(x0.clone(), n_steps=200, show_progress=False)
    assert torch.allclose(out_nag, target, atol=1e-2)


def test_momentum_changes_trajectory():
    """Momentum must actually alter the update, not silently no-op."""

    class TowardPoint:
        def __init__(self, target):
            self.target = target
            self.model = self

        def __call__(self, x, **cond):
            return self.target - x

    target = torch.tensor([[5.0]])
    x0 = torch.zeros(1, 1)

    gd = EquilibriumSolver(TowardPoint(target), step_size=0.1, momentum=0.0)
    nag = EquilibriumSolver(TowardPoint(target), step_size=0.1, momentum=0.9)

    out_gd = gd.sample(x0.clone(), n_steps=5, show_progress=False)
    out_nag = nag.sample(x0.clone(), n_steps=5, show_progress=False)
    assert not torch.allclose(out_gd, out_nag)


def test_gradient_descent_solver_is_equilibrium_solver_alias():
    assert GradientDescentSolver is EquilibriumSolver
