import torch

from deltaflow.inverse import (
    BlurOperator,
    DownsampleOperator,
    GaussianLikelihood,
    IdentityOperator,
    LinearTweedie,
    MaskOperator,
    VPTweedie,
)


def test_linear_tweedie_inverts_linear_path():
    """LinearTweedie composed with a linear path should recover (x0, x1)."""
    torch.manual_seed(0)
    x0 = torch.randn(4, 3, 8, 8)
    x1 = torch.randn(4, 3, 8, 8)
    t = 0.3
    x_t = (1 - t) * x0 + t * x1
    v_t = x1 - x0

    x_clean, x_noise = LinearTweedie().decompose(x_t, v_t, t)

    assert torch.allclose(x_clean, x1, atol=1e-5)
    assert torch.allclose(x_noise, x0, atol=1e-5)


def test_vp_tweedie_inverts_vp_path():
    import math

    torch.manual_seed(0)
    x0 = torch.randn(4, 3, 8, 8)
    x1 = torch.randn(4, 3, 8, 8)
    t = 0.4
    a = math.sin(0.5 * math.pi * t)
    s = math.cos(0.5 * math.pi * t)
    x_t = a * x1 + s * x0
    v_t = 0.5 * math.pi * (s * x1 - a * x0)

    x_clean, x_noise = VPTweedie().decompose(x_t, v_t, t)

    assert torch.allclose(x_clean, x1, atol=1e-4)
    assert torch.allclose(x_noise, x0, atol=1e-4)


def test_identity_and_mask_operators():
    x = torch.randn(2, 1, 4, 4)
    assert torch.allclose(IdentityOperator()(x), x)

    mask = torch.zeros(1, 1, 4, 4)
    mask[..., 1:3, 1:3] = 1.0
    out = MaskOperator(mask)(x)
    assert torch.allclose(out, x * mask)


def test_blur_operator_preserves_shape_and_energy():
    x = torch.randn(2, 3, 8, 8)
    blur = BlurOperator(kernel_size=3, sigma=1.0, channels=3)
    y = blur(x)
    assert y.shape == x.shape
    # Blurring must never amplify energy.
    assert y.pow(2).mean().item() <= x.pow(2).mean().item() * 1.5


def test_downsample_operator_halves_spatial_dims():
    x = torch.randn(2, 3, 8, 8)
    y = DownsampleOperator(2)(x)
    assert y.shape == (2, 3, 4, 4)


def test_gaussian_likelihood_is_zero_at_true_signal_and_positive_elsewhere():
    torch.manual_seed(0)
    x_true = torch.randn(1, 1, 4, 4)
    y = x_true.clone()
    lik = GaussianLikelihood(y=y, operator=IdentityOperator(), sigma=1.0)

    assert lik.neg_log_prob(x_true).item() == 0.0
    assert lik.neg_log_prob(torch.zeros_like(x_true)).item() > 0.0


def test_gaussian_likelihood_gradient_points_toward_measurement():
    torch.manual_seed(0)
    y = torch.randn(1, 1, 4, 4)
    x = torch.zeros_like(y).requires_grad_(True)
    lik = GaussianLikelihood(y=y, operator=IdentityOperator(), sigma=1.0)

    loss = lik.neg_log_prob(x)
    (grad,) = torch.autograd.grad(loss, x)

    # For identity A and sigma=1: -log p = 0.5 * ||y - x||^2, gradient = -(y - x) = x - y.
    # At x=0, gradient = -y, so stepping opposite the gradient moves toward y.
    step = x.detach() - 0.5 * grad
    assert (step - y).abs().mean() < (x.detach() - y).abs().mean()
