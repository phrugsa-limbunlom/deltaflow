import torch

from deltaflow.interpolants import LinearInterpolant


def test_linear_interpolant_boundary_conditions():
    torch.manual_seed(0)
    interpolant = LinearInterpolant()
    x1 = torch.randn(8, 4)
    x0 = torch.randn(8, 4)

    x_t0, u = interpolant.interpolate(x1, torch.zeros(8), x0=x0)
    x_t1, _ = interpolant.interpolate(x1, torch.ones(8), x0=x0)

    assert torch.allclose(x_t0, x0, atol=1e-6)
    assert torch.allclose(x_t1, x1, atol=1e-6)
    assert torch.allclose(u, x1 - x0, atol=1e-6)


def test_linear_interpolant_default_noise_shape():
    interpolant = LinearInterpolant()
    x1 = torch.randn(4, 3, 8, 8)
    t = torch.rand(4)

    x_t, u = interpolant.interpolate(x1, t)

    assert x_t.shape == x1.shape
    assert u.shape == x1.shape
