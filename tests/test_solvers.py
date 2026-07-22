import torch

from deltaflow.inverse import GaussianLikelihood, IdentityOperator, LinearTweedie, MaskOperator
from deltaflow.solvers import EulerSolver, HeunSolver, PosteriorSolver
from tests.conftest import DummyVelocityField


def test_euler_and_heun_shape_match():
    torch.manual_seed(0)
    model = DummyVelocityField(dim=4)
    x = torch.randn(3, 4)

    y_euler = EulerSolver(model).sample(x, n_steps=10, show_progress=False)
    y_heun = HeunSolver(model).sample(x, n_steps=10, show_progress=False)

    assert y_euler.shape == x.shape
    assert y_heun.shape == x.shape


def test_heun_more_accurate_than_euler_on_linear_field():
    """For a constant-velocity field v = c, both solvers are exact, but for a
    time-varying field Heun should reduce truncation error relative to Euler."""
    torch.manual_seed(0)

    class TimeQuadraticField:
        # v(x, t) = 2 * t * ones. True x(1) - x(0) = 1 for t in [0, 1]. Model receives
        # a (B,)-shaped time tensor and must broadcast.
        def __init__(self):
            self.time_scale = 1.0
            self.model = self  # BaseSolver.model attribute

        def __call__(self, x, t, **cond):
            t_ = t.view(-1, *([1] * (x.dim() - 1)))
            return 2.0 * t_ * torch.ones_like(x)

    field = TimeQuadraticField()
    x0 = torch.zeros(1, 1)

    y_euler = EulerSolver(field).sample(x0.clone(), n_steps=4, show_progress=False)
    y_heun = HeunSolver(field).sample(x0.clone(), n_steps=4, show_progress=False)

    # Exact solution: integral of 2t from 0 to 1 = 1.
    err_euler = (y_euler - 1.0).abs().item()
    err_heun = (y_heun - 1.0).abs().item()
    assert err_heun < err_euler


def test_posterior_solver_wraps_base_solver_shape():
    torch.manual_seed(0)
    model = DummyVelocityField(dim=4)
    base = EulerSolver(model)

    y = torch.randn(2, 4)
    lik = GaussianLikelihood(y=y, operator=IdentityOperator(), sigma=1.0)
    posterior = PosteriorSolver(base, likelihood=lik, tweedie=LinearTweedie(), guidance_scale=0.1)

    x = torch.randn(2, 4)
    out = posterior.sample(x, n_steps=5, show_progress=False)
    assert out.shape == x.shape


def test_posterior_solver_pulls_toward_measurement_on_inpainting():
    """With a mask operator, posterior sampling should move the masked region
    toward the measurement more than plain unconditional sampling does.
    """
    torch.manual_seed(0)

    # Use a small 2D toy where the velocity field just points toward zero, so
    # unconditional trajectories collapse to a fixed point independent of y.
    class DecayField:
        def __init__(self):
            self.model = self
            self.time_scale = 1.0

        def __call__(self, x, t, **cond):
            return -x  # dx/dt = -x, x(t) = x(0) e^{-t} unconditionally.

    y = torch.tensor([[5.0]])
    mask = torch.tensor([[1.0]])
    op = MaskOperator(mask)
    lik = GaussianLikelihood(y=y, operator=op, sigma=1.0)

    x_init = torch.tensor([[0.0]])
    base = EulerSolver(DecayField())
    uncond = base.sample(x_init.clone(), n_steps=20, show_progress=False)

    posterior = PosteriorSolver(base, likelihood=lik, guidance_scale=0.2)
    cond = posterior.sample(x_init.clone(), n_steps=20, show_progress=False)

    # Unconditionally we stay near zero; with a measurement of 5, cond should be pulled toward 5.
    assert cond.item() > uncond.item()
