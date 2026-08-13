import torch

from deltaflow.interpolants import (
    LinearInterpolant,
    OTInterpolant,
    SchrodingerBridgeInterpolant,
    VariancePreservingInterpolant,
)


def test_ot_interpolant_boundary_conditions():
    torch.manual_seed(0)
    interpolant = OTInterpolant()
    x1 = torch.randn(8, 3, 4, 4)
    x0 = torch.randn(8, 3, 4, 4)

    # At t=0 and t=1 the OT-permuted path must still hit x0_permuted and x1 exactly.
    x_t0, u0 = interpolant.interpolate(x1, torch.zeros(8), x0=x0.clone())
    x_t1, _ = interpolant.interpolate(x1, torch.ones(8), x0=x0.clone())

    # The permutation is deterministic in expectation - just verify shape / boundary structure.
    assert x_t0.shape == x1.shape
    assert torch.allclose(x_t1, x1, atol=1e-6)
    # At t=0, x_t equals the permuted x0; u = x1 - x0_permuted.
    assert torch.allclose(x_t0 + u0, x1, atol=1e-6)


def test_ot_permutation_moves_toward_nearest_neighbours():
    """With clearly clustered x0/x1, OT should pair members of the same cluster."""
    torch.manual_seed(0)
    # Two clusters, four samples: x0 targets should be paired with the nearest x1.
    x0 = torch.tensor([[10.0], [10.1], [-10.0], [-10.1]])
    x1 = torch.tensor([[-10.05], [10.02], [10.05], [-10.02]])
    interpolant = OTInterpolant()

    x_t, _ = interpolant.interpolate(x1, torch.full((4,), 0.0), x0=x0.clone())
    # After OT, x_t at t=0 should have magnitudes matching x1's cluster of each row.
    assert torch.allclose(x_t.sign(), x1.sign())


def test_variance_preserving_boundary_conditions():
    torch.manual_seed(0)
    interpolant = VariancePreservingInterpolant()
    x1 = torch.randn(6, 2)
    x0 = torch.randn(6, 2)

    x_t0, _ = interpolant.interpolate(x1, torch.zeros(6), x0=x0.clone())
    x_t1, _ = interpolant.interpolate(x1, torch.ones(6), x0=x0.clone())

    # alpha(0)=0, sigma(0)=1 => x_t(0) = x0.
    assert torch.allclose(x_t0, x0, atol=1e-6)
    # alpha(1)=1, sigma(1)=0 => x_t(1) = x1.
    assert torch.allclose(x_t1, x1, atol=1e-6)


def test_linear_interpolant_still_works():
    """Regression: the original LinearInterpolant behaviour must be unchanged."""
    torch.manual_seed(0)
    x1 = torch.randn(4, 3, 8, 8)
    interp = LinearInterpolant()
    x_t, u = interp.interpolate(x1, torch.rand(4))
    assert x_t.shape == x1.shape
    assert u.shape == x1.shape


def test_schrodinger_bridge_boundary_conditions():
    """At t=0/1 the bridge variance vanishes, so x_t must hit x0/x1 exactly."""
    torch.manual_seed(0)
    interpolant = SchrodingerBridgeInterpolant(sigma=1.5)
    x1 = torch.randn(6, 3, 4, 4)
    x0 = torch.randn(6, 3, 4, 4)

    x_t0, _ = interpolant.interpolate(x1, torch.zeros(6), x0=x0.clone())
    x_t1, _ = interpolant.interpolate(x1, torch.ones(6), x0=x0.clone())

    assert torch.allclose(x_t0, x0, atol=1e-5)
    assert torch.allclose(x_t1, x1, atol=1e-5)


def test_schrodinger_bridge_zero_sigma_matches_linear():
    """sigma=0 collapses the bridge onto the deterministic straight-line path."""
    torch.manual_seed(0)
    x1 = torch.randn(8, 4)
    x0 = torch.randn(8, 4)
    t = torch.rand(8)

    sb = SchrodingerBridgeInterpolant(sigma=0.0)
    linear = LinearInterpolant()

    x_t_sb, u_sb = sb.interpolate(x1, t, x0=x0.clone())
    x_t_lin, u_lin = linear.interpolate(x1, t, x0=x0.clone())

    assert torch.allclose(x_t_sb, x_t_lin, atol=1e-6)
    assert torch.allclose(u_sb, u_lin, atol=1e-6)


def test_schrodinger_bridge_shapes_and_finite():
    torch.manual_seed(0)
    interpolant = SchrodingerBridgeInterpolant(sigma=1.0)
    x1 = torch.randn(5, 2, 6, 6)
    t = torch.rand(5)

    x_t, u = interpolant.interpolate(x1, t)

    assert x_t.shape == x1.shape
    assert u.shape == x1.shape
    assert torch.isfinite(x_t).all()
    assert torch.isfinite(u).all()
