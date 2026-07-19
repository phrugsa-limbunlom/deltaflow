import torch

from deltaflow.samplers import FlowSampler
from tests.conftest import DummyVelocityField


def test_flow_sampler_output_shape():
    torch.manual_seed(0)
    model = DummyVelocityField(dim=4)
    sampler = FlowSampler(model)

    x = torch.randn(5, 4)
    out = sampler.sample(x, n_steps=10, show_progress=False)

    assert out.shape == x.shape


def test_flow_sampler_with_x_cond():
    torch.manual_seed(0)
    model = DummyVelocityField(dim=4)
    sampler = FlowSampler(model)

    x_cond = torch.randn(3, 4)
    out = sampler.sample(torch.zeros_like(x_cond), n_steps=5, x_cond=x_cond, t_start=0.5, show_progress=False)

    assert out.shape == x_cond.shape
