"""Check that the docs trace follows real DiT layers, not a schematic."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from deltaflow.models import DiT


@pytest.fixture
def showcase():
    pytest.importorskip("matplotlib")
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "90-showcase"
        / "10-dit-tensor-viz"
        / "main.py"
    )
    spec = importlib.util.spec_from_file_location("dit_walkthrough", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def field(showcase):
    torch.manual_seed(7)
    model = DiT(
        input_size=showcase.IMG,
        patch_size=showcase.PATCH,
        in_channels=showcase.CHANNELS,
        hidden_size=showcase.HIDDEN,
        depth=showcase.DEPTH,
        num_heads=showcase.HEADS,
        num_classes=showcase.N_CLASSES,
    ).eval()
    # Exercise nonzero residual branches and output rather than the initial identity.
    with torch.no_grad():
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=0.02)
        model.final_layer.adaLN_modulation[-1].weight.normal_(std=0.02)
        model.final_layer.linear.weight.normal_(std=0.02)
    return model


def test_trace_intermediates_match_real_layers(showcase, field):
    x = torch.randn(2, showcase.CHANNELS, showcase.IMG, showcase.IMG)
    t, y = torch.tensor([0.0, 0.9]), torch.tensor([0, 1])
    traced = showcase.trace(field, x, t, y)
    observed = {}
    handles = []

    def capture(name):
        def hook(module, inputs, output):
            observed[name] = output

        return hook

    def capture_input(name):
        def hook(module, inputs):
            observed[name] = inputs[0]

        return hook

    for b, block in enumerate(field.blocks):
        handles.append(block.register_forward_hook(capture((b, "output"))))
        handles.append(block.attn.register_forward_pre_hook(capture_input((b, "attention_input"))))
        handles.append(block.norm2.register_forward_pre_hook(capture_input((b, "after_attention"))))
        handles.append(block.mlp.register_forward_pre_hook(capture_input((b, "mlp_input"))))
        handles.append(block.mlp[1].register_forward_hook(capture((b, "mlp_hidden"))))
    handles.append(field.final_layer.linear.register_forward_pre_hook(capture_input("final_input")))
    try:
        with torch.no_grad():
            velocity = field(x, t, y=y)
    finally:
        for handle in handles:
            handle.remove()

    for b, state in enumerate(traced["block_states"]):
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, observed[b, name])
    torch.testing.assert_close(traced["final_input"], observed["final_input"])
    torch.testing.assert_close(traced["v"], velocity)


def test_export_contains_layer_tensors_and_exact_pixel_layout(showcase, field, tmp_path):
    data = json.loads(showcase.export_trace(field, tmp_path / "trace.json").read_text())
    assert data["schema_version"] == 3
    assert len(data["traces"]) == 2 * len(showcase.EXPORT_T) * 3
    cfg = data["config"]
    assert cfg["channels"] == 3
    assert torch.tensor(data["x0"]).shape == (cfg["img"], cfg["img"], 3)
    for frames in data["sampling"].values():
        assert len(frames) == showcase.SAMPLE_STEPS + 1
        assert torch.tensor(frames).shape == (showcase.SAMPLE_STEPS + 1, cfg["img"], cfg["img"], 3)
    for tr in data["traces"].values():
        assert len(tr["block_states"]) == cfg["depth"]
        for state in tr["block_states"]:
            for name, values in state.items():
                width = 4 * cfg["D"] if name == "mlp_hidden" else cfg["D"]
                assert torch.tensor(values).shape == (cfg["N"], width)
                assert torch.isfinite(torch.tensor(values)).all()
        assert torch.tensor(tr["final_input"]).shape == (cfg["N"], cfg["D"])
        assert torch.tensor(tr["x_t"]).shape == (cfg["img"], cfg["img"], 3)
        assert torch.tensor(tr["final"]).shape == (cfg["N"], cfg["patch"] ** 2 * 3)
        patches = torch.tensor(tr["final"]).reshape(
            1, cfg["grid"], cfg["grid"], cfg["patch"], cfg["patch"], cfg["channels"]
        )
        pixels = (
            torch.einsum("bhwpqc->bchpwq", patches)
            .reshape(cfg["channels"], cfg["img"], cfg["img"])
            .permute(1, 2, 0)
        )
        torch.testing.assert_close(pixels, torch.tensor(tr["v"]), atol=0, rtol=0)


def test_rgb_channels_are_distinct_and_export_preserves_values(showcase):
    images = showcase.class_images(torch.tensor([0, 1]), noise=0)
    assert images.shape == (2, 3, showcase.IMG, showcase.IMG)
    assert not torch.equal(images[:, 0], images[:, 1])
    assert not torch.equal(images[:, 1], images[:, 2])
    torch.testing.assert_close(images[0], images[1].flip(1))
    raw = torch.arange(3 * showcase.IMG**2).reshape(3, showcase.IMG, showcase.IMG).float() - 10
    exported = torch.tensor(showcase._image(raw))
    torch.testing.assert_close(exported, raw.permute(1, 2, 0), atol=0, rtol=0)
    assert exported.min() < 0 and exported.max() > 1
