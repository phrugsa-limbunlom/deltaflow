import torch

from deltaflow.interpolants import LinearInterpolant, OTInterpolant
from deltaflow.losses import ConditionalFlowMatchingLoss, FlowMatchingLoss
from deltaflow.trainer.coupling import IndependentCoupling, OTCoupling
from tests.conftest import DummyVelocityField


def test_flow_matching_loss_backwards_compat_alias():
    assert FlowMatchingLoss is ConditionalFlowMatchingLoss


def test_independent_coupling_returns_noise_and_data_shapes():
    x1 = torch.randn(6, 3, 4, 4)
    x0, x1_out = IndependentCoupling().sample_pair(x1)
    assert x0.shape == x1.shape
    assert torch.allclose(x1_out, x1)


def test_ot_coupling_permutation_reduces_transport_cost():
    torch.manual_seed(0)
    x1 = torch.tensor([[10.0], [-10.0], [10.5], [-10.5]])
    coupling = OTCoupling()
    x0, _ = coupling.sample_pair(x1)
    # After OT permutation the pairs (x0[i], x1[i]) should have signs matching.
    # We can only check that total cost <= random cost with high probability;
    # verify via total squared L2 vs. an unpermuted baseline.
    # Draw a fresh independent x0 as baseline (same seed dependence).
    torch.manual_seed(0)
    x0_indep = torch.randn_like(x1)

    cost_ot = ((x0 - x1) ** 2).sum().item()
    cost_indep = ((x0_indep - x1) ** 2).sum().item()
    assert cost_ot <= cost_indep + 1e-6


def test_conditional_flow_matching_loss_with_ot_coupling_runs():
    torch.manual_seed(0)
    model = DummyVelocityField(dim=4)
    loss_fn = ConditionalFlowMatchingLoss(
        interpolant=LinearInterpolant(),
        coupling=OTCoupling(),
    )
    x1 = torch.randn(6, 4)
    loss = loss_fn(model, x1)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_conditional_flow_matching_loss_with_ot_interpolant_runs():
    """OT can equivalently be plugged via the interpolant, not the coupling."""
    torch.manual_seed(0)
    model = DummyVelocityField(dim=4)
    loss_fn = ConditionalFlowMatchingLoss(interpolant=OTInterpolant())
    x1 = torch.randn(6, 4)
    loss = loss_fn(model, x1)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
