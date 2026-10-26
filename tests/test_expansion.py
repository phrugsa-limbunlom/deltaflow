"""Tests for Active Flow Expansion (deltaflow.expansion) and the signed replay loss."""

import math

import pytest
import torch
import torch.nn as nn

from deltaflow.core import BaseVelocityField
from deltaflow.expansion import (
    ActFlow,
    FlowRepresentation,
    KernelUncertainty,
    UncertaintyGuidance,
    generable_coverage,
)
from deltaflow.inverse import Likelihood
from deltaflow.losses import ConditionalFlowMatchingLoss, SignedReplayFlowMatchingLoss
from deltaflow.solvers import EulerSolver, PosteriorSolver


class MLPField(BaseVelocityField):
    def __init__(self, dim: int = 2, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 1, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x, t, **cond):
        return self.net(torch.cat([x, t.view(-1, 1)], dim=-1))


# --- KernelUncertainty --------------------------------------------------------


def _dense_variance(z_ref, z, reg, kernel_fn):
    k_ref = kernel_fn(z_ref, z_ref)
    k_star = kernel_fn(z_ref, z)
    inv = torch.linalg.inv(k_ref + reg * torch.eye(k_ref.shape[0], dtype=k_ref.dtype))
    return torch.diagonal(kernel_fn(z, z)) - (k_star * (inv @ k_star)).sum(0)


def test_kernel_uncertainty_prior_without_queries():
    z = torch.randn(5, 3)
    unc = KernelUncertainty(kernel="linear")
    unc.fit(torch.empty(0, 3))
    assert torch.allclose(unc.variance(z), z.pow(2).sum(-1), atol=1e-6)
    assert torch.allclose(KernelUncertainty(kernel="rbf").variance(z), torch.ones(5))


@pytest.mark.parametrize("n_ref", [4, 40])  # dual (t < D) and primal (t > D) paths
def test_kernel_uncertainty_matches_closed_form(n_ref):
    torch.manual_seed(0)
    z_ref = torch.randn(n_ref, 8, dtype=torch.float64)
    z = torch.randn(6, 8, dtype=torch.float64)
    unc = KernelUncertainty(kernel="linear", reg=0.5)
    unc.fit(z_ref)
    assert unc._primal == (n_ref > 8)
    expected = _dense_variance(z_ref, z, 0.5, lambda a, b: a @ b.T)
    assert torch.allclose(unc.variance(z), expected, atol=1e-8)


def test_kernel_uncertainty_rbf_matches_closed_form():
    torch.manual_seed(0)
    z_ref = torch.randn(10, 3, dtype=torch.float64)
    z = torch.randn(4, 3, dtype=torch.float64)
    unc = KernelUncertainty(kernel="rbf", reg=0.1, lengthscale=0.7)
    unc.fit(z_ref)
    rbf = lambda a, b: torch.exp(-0.5 * torch.cdist(a, b).pow(2) / 0.49)  # noqa: E731
    assert torch.allclose(unc.variance(z), _dense_variance(z_ref, z, 0.1, rbf), atol=1e-8)


def test_kernel_uncertainty_shrinks_where_queried_and_is_differentiable():
    torch.manual_seed(0)
    z_ref = torch.randn(20, 4)
    unc = KernelUncertainty(kernel="rbf", reg=1e-2, lengthscale=0.5)
    unc.fit(z_ref)
    far = z_ref[:1] + 10.0
    assert unc(z_ref[:1]).item() < unc(far).item()

    z = z_ref[:3].clone().requires_grad_(True)
    unc(z).sum().backward()
    assert torch.isfinite(z.grad).all()


def test_kernel_uncertainty_posterior_mean_fits_labels():
    torch.manual_seed(0)
    z = torch.randn(30, 2)
    y = (z[:, 0] > 0).float()
    unc = KernelUncertainty(kernel="rbf", reg=1e-3, lengthscale=0.5)
    unc.fit(z, y)
    assert ((unc.mean(z) > 0.5).float() == y).float().mean() > 0.95


def test_kernel_uncertainty_max_points():
    unc = KernelUncertainty(max_points=5)
    unc.fit(torch.randn(12, 3))
    assert unc.n_points == 5


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_unfitted_uncertainty_stays_in_autograd_graph(kernel):
    """The first ActFlow round guides with the prior, which must be differentiable."""
    z = torch.randn(4, 3, requires_grad=True)
    sigma = KernelUncertainty(kernel=kernel)(z)
    assert sigma.requires_grad
    sigma.sum().backward()
    assert torch.isfinite(z.grad).all()


# --- FlowRepresentation -------------------------------------------------------


def test_flow_representation_hidden_layer_shapes_and_hook_cleanup():
    torch.manual_seed(0)
    field = MLPField(hidden=16)
    rep = FlowRepresentation(field, s=0.9, layer="net.1")
    z = rep(torch.randn(7, 2))
    assert z.shape == (7, 16)
    assert torch.allclose(z.norm(dim=-1), torch.ones(7), atol=1e-5)
    assert len(field.net[1]._forward_hooks) == 0


def test_flow_representation_pooling():
    rep = FlowRepresentation(lambda x, t: x, s=1.0, normalize=False)
    assert rep(torch.randn(3, 5, 8)).shape == (3, 8)  # tokens, mean over N
    assert rep(torch.randn(3, 4, 6, 6)).shape == (3, 4)  # feature map, mean over space
    flat = FlowRepresentation(lambda x, t: x, s=1.0, pool="flatten", normalize=False)
    assert flat(torch.randn(3, 4, 6, 6)).shape == (3, 144)


def test_flow_representation_is_deterministic_given_noise_and_differentiable():
    field = MLPField()
    rep = FlowRepresentation(field, s=0.8, layer="net.1")
    x = torch.randn(4, 2, requires_grad=True)
    noise = torch.randn(4, 2)
    assert torch.allclose(rep(x, noise=noise), rep(x, noise=noise))
    rep(x, noise=noise).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# --- UncertaintyGuidance ------------------------------------------------------


def test_uncertainty_guidance_is_a_likelihood_and_infinite_beta_is_plain_sampling():
    torch.manual_seed(0)
    field = MLPField()
    unc = KernelUncertainty()
    unc.fit(torch.randn(10, 2))
    guidance = UncertaintyGuidance(FlowRepresentation(field), unc, beta=math.inf)
    assert isinstance(guidance, Likelihood)

    x0 = torch.randn(8, 2)
    plain = EulerSolver(field).sample(x0.clone(), n_steps=10, show_progress=False)
    tilted = PosteriorSolver(EulerSolver(field), guidance, guidance_scale=1.0).sample(
        x0.clone(), n_steps=10, show_progress=False
    )
    assert torch.allclose(plain, tilted, atol=1e-6)


def test_uncertainty_guidance_increases_uncertainty():
    torch.manual_seed(0)
    field = MLPField()
    rep = FlowRepresentation(field, s=1.0, normalize=False)
    unc = KernelUncertainty(kernel="rbf", reg=1e-2, lengthscale=0.5)
    x0 = torch.randn(64, 2)
    plain = EulerSolver(field).sample(x0.clone(), n_steps=10, show_progress=False)
    unc.fit(rep(plain).detach())  # everything plain sampling reaches is now "known"

    guidance = UncertaintyGuidance(rep, unc, beta=0.05)
    tilted = PosteriorSolver(EulerSolver(field), guidance, guidance_scale=0.05).sample(
        x0.clone(), n_steps=10, show_progress=False
    )
    with torch.no_grad():
        assert unc(rep(tilted)).mean() > unc(rep(plain)).mean()


# --- SignedReplayFlowMatchingLoss --------------------------------------------


def test_signed_replay_alpha_zero_is_flow_matching():
    field = MLPField()
    x, x_neg = torch.randn(16, 2), torch.randn(16, 2)
    torch.manual_seed(1)
    ref = ConditionalFlowMatchingLoss()(field, x)
    torch.manual_seed(1)
    got = SignedReplayFlowMatchingLoss(alpha=0.0)(field, x, x_neg)
    assert torch.allclose(ref, got)


def test_signed_replay_fixed_alpha_subtracts_rejected_term():
    field = MLPField()
    base = ConditionalFlowMatchingLoss()
    x_pos, x_neg = torch.randn(16, 2), torch.randn(16, 2)
    torch.manual_seed(2)
    expected = base(field, x_pos) - 0.3 * base(field, x_neg)
    torch.manual_seed(2)
    got = SignedReplayFlowMatchingLoss(base, alpha=0.3)(field, x_pos, x_neg)
    assert torch.allclose(expected, got)


def test_signed_replay_grad_norm_calibration():
    torch.manual_seed(0)
    field = MLPField()
    params = list(field.parameters())

    class Fixed:  # deterministic per-sample loss so gradients can be recomputed
        def __call__(self, model, x1, **cond):
            t = torch.full((x1.shape[0],), 0.5)
            return (model(x1, t) - x1).pow(2).mean()

    x_pos, x_neg = torch.randn(16, 2), torch.randn(16, 2) * 3
    loss_fn = SignedReplayFlowMatchingLoss(Fixed(), alpha=0.25, alpha_mode="grad_norm")
    loss = loss_fn(field, x_pos, x_neg)
    loss.backward()

    def norm(loss_val):
        grads = torch.autograd.grad(loss_val, params)
        return torch.sqrt(sum(g.pow(2).sum() for g in grads))

    g_pos, g_neg = norm(Fixed()(field, x_pos)), norm(Fixed()(field, x_neg))
    assert math.isclose(loss_fn.last_alpha * g_neg.item(), 0.25 * g_pos.item(), rel_tol=1e-4)


# --- ActFlow ------------------------------------------------------------------


def _half_plane_verifier(x):
    return x[:, 0] > 0


def test_actflow_round_updates_buffers_and_model():
    torch.manual_seed(0)
    field = MLPField()
    opt = torch.optim.Adam(field.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in field.parameters()]
    act = ActFlow(
        field,
        _half_plane_verifier,
        opt,
        sample_shape=(2,),
        representation=FlowRepresentation(field, layer="net.1"),
        loss=SignedReplayFlowMatchingLoss(alpha=0.1, alpha_mode="grad_norm"),
        beta=0.5,
        guidance_scale=0.1,
        n_steps=5,
        finetune_steps=3,
        batch_size=8,
    )
    r0 = act.step(32)
    r1 = act.step(32)
    assert act.queries.shape == (64, 2) and act.labels.shape == (64,)
    assert act.accepted.shape[0] + act.rejected.shape[0] == 64
    assert act.uncertainty.n_points == 32  # refit on the first round's queries
    assert r0.finetuned and r1.round == 1
    assert 0.0 <= r1.acceptance_rate <= 1.0
    assert all(torch.isfinite(r.samples).all() for r in (r0, r1))
    assert any(not torch.equal(a, b) for a, b in zip(before, field.parameters()))
    assert len(act.history) == 2


def test_actflow_warmup_and_self_training_baseline():
    torch.manual_seed(0)
    field = MLPField()
    opt = torch.optim.SGD(field.parameters(), lr=1e-2)
    act = ActFlow(
        field, _half_plane_verifier, opt, (2,), beta=math.inf, n_steps=4, min_accepted=10_000
    )
    rounds = act.run(2, 16)
    assert not any(r.finetuned for r in rounds)
    assert rounds[0].loss is None


def test_actflow_rejects_mismatched_verifier():
    field = MLPField()
    act = ActFlow(field, lambda x: torch.ones(3), torch.optim.SGD(field.parameters(), lr=0.1), (2,))
    with pytest.raises(ValueError):
        act.query(torch.randn(5, 2))


# --- metrics ------------------------------------------------------------------


def test_generable_coverage_full_and_partial():
    torch.manual_seed(0)
    uniform = torch.rand(20000, 2) * 2 - 1
    full = generable_coverage(uniform, lambda g: torch.ones(g.shape[0]), (-1, 1), bins=10)
    assert full["coverage"] == 1.0 and full["validity"] == 1.0

    left = uniform[uniform[:, 0] < 0]
    half = generable_coverage(left, lambda g: torch.ones(g.shape[0]), (-1, 1), bins=10)
    assert half["coverage"] == pytest.approx(0.5)
    assert half["n_valid_bins"] == 100
