"""Tests for the DiT velocity field and its adaLN-Zero conditioning."""

import torch

from deltaflow.core.base import BaseVelocityField
from deltaflow.models import DiT, LabelEmbedding, TimestepEmbedding, modulate


def _tiny_dit(**overrides) -> DiT:
    cfg = dict(
        input_size=8,
        patch_size=2,
        in_channels=1,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=3,
        class_dropout_prob=0.1,
    )
    cfg.update(overrides)
    return DiT(**cfg)


def test_dit_is_velocity_field_and_preserves_shape():
    model = _tiny_dit()
    assert isinstance(model, BaseVelocityField)

    x = torch.randn(4, 1, 8, 8)
    t = torch.rand(4)
    y = torch.randint(0, 3, (4,))

    v = model(x, t, y=y)
    assert v.shape == x.shape


def test_dit_accepts_cond_y_keyword():
    """The label must flow through the opaque ``**cond`` channel as ``y``."""
    model = _tiny_dit()
    x = torch.randn(2, 1, 8, 8)
    t = torch.rand(2)
    cond = {"y": torch.tensor([0, 2])}

    v = model(x, t, **cond)
    assert v.shape == x.shape


def test_dit_runs_unconditionally_without_labels():
    model = _tiny_dit().eval()
    x = torch.randn(2, 1, 8, 8)
    v = model(x, torch.rand(2))
    assert v.shape == x.shape


def test_dit_scalar_time_broadcasts():
    model = _tiny_dit().eval()
    x = torch.randn(3, 1, 8, 8)
    v = model(x, 0.5, y=torch.zeros(3, dtype=torch.long))
    assert v.shape == x.shape


def test_zero_init_gives_zero_velocity():
    """adaLN-Zero + zero final layer => predicted velocity is exactly 0."""
    model = _tiny_dit().eval()
    x = torch.randn(4, 1, 8, 8)
    t = torch.rand(4)
    y = torch.randint(0, 3, (4,))

    v = model(x, t, y=y)
    assert torch.allclose(v, torch.zeros_like(v))


def test_block_gates_are_zero_initialized():
    model = _tiny_dit()
    for block in model.blocks:
        assert torch.all(block.adaLN_modulation[-1].weight == 0)
        assert torch.all(block.adaLN_modulation[-1].bias == 0)
    assert torch.all(model.final_layer.linear.weight == 0)
    assert torch.all(model.final_layer.linear.bias == 0)


def test_label_embedding_has_null_class():
    emb = LabelEmbedding(num_classes=5, hidden_size=8, dropout_prob=0.0)
    assert emb.embedding_table.num_embeddings == 6
    assert emb.null_index == 5

    labels = torch.tensor([0, 1, 2])
    out = emb(labels, train=False)
    assert out.shape == (3, 8)


def test_label_embedding_token_drop_forces_null():
    emb = LabelEmbedding(num_classes=4, hidden_size=8, dropout_prob=0.0)
    labels = torch.tensor([0, 1, 2, 3])
    force = torch.tensor([True, False, True, False])

    dropped = emb.token_drop(labels, force_drop_ids=force)
    assert dropped.tolist() == [4, 1, 4, 3]


def test_cfg_interpolates_between_cond_and_uncond():
    """cfg_scale=1 must equal the plain conditional velocity."""
    torch.manual_seed(0)
    model = _tiny_dit().eval()
    # Perturb the zero-initialized output so the field is non-trivial.
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.1)
        model.final_layer.linear.bias.normal_(std=0.1)

    x = torch.randn(3, 1, 8, 8)
    t = torch.rand(3)
    y = torch.tensor([0, 1, 2])

    v_cond = model(x, t, y=y)
    v_cfg1 = model.forward_with_cfg(x, t, y, cfg_scale=1.0)
    assert torch.allclose(v_cond, v_cfg1, atol=1e-5)


def test_timestep_embedding_shape():
    emb = TimestepEmbedding(hidden_size=16)
    t = torch.rand(5)
    assert emb(t).shape == (5, 16)


def test_modulate_identity_at_zero():
    x = torch.randn(2, 4, 8)
    shift = torch.zeros(2, 8)
    scale = torch.zeros(2, 8)
    assert torch.allclose(modulate(x, shift, scale), x)


def test_dit_backprop_updates_gates():
    """One optimizer step must move the gates off their zero initialization."""
    torch.manual_seed(0)
    model = _tiny_dit()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)

    x = torch.randn(4, 1, 8, 8)
    t = torch.rand(4)
    y = torch.randint(0, 3, (4,))
    target = torch.randn_like(x)

    loss = ((model(x, t, y=y) - target) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    assert not torch.all(model.final_layer.linear.weight == 0)
