"""Smoke test for the training loop on a tiny image dataset."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("PIL", reason="Pillow required for ImageFolderStream tests")
from PIL import Image  # noqa: E402

from deltaflow.losses import ConditionalFlowMatchingLoss  # noqa: E402
from deltaflow.models import TinyVelocityField  # noqa: E402
from deltaflow.trainer import TrainConfig, build_loader, load_checkpoint, train  # noqa: E402
from deltaflow.trainer.data import ImageFolderStream  # noqa: E402


def _make_toy_image_dir(tmp_path: Path, n: int = 6, size: int = 16) -> Path:
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(n):
        arr = (torch.rand(size, size) * 255).byte().numpy()
        Image.fromarray(arr, mode="L").save(d / f"img_{i:03d}.png")
    return d


def test_image_folder_stream_returns_normalized_tensors(tmp_path):
    d = _make_toy_image_dir(tmp_path)
    ds = ImageFolderStream(d, image_size=8, mode="L")
    x = ds[0]
    assert x.shape == (1, 8, 8)
    assert x.min().item() >= -1.0 - 1e-6
    assert x.max().item() <= 1.0 + 1e-6


def test_train_loop_runs_and_checkpoints(tmp_path):
    torch.manual_seed(0)
    d = _make_toy_image_dir(tmp_path, n=4, size=16)
    ds = ImageFolderStream(d, image_size=16, mode="L")
    loader = build_loader(ds, batch_size=2, num_workers=0, drop_last=False)

    model = TinyVelocityField(channels=1, hidden=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = ConditionalFlowMatchingLoss()

    cfg = TrainConfig(
        max_steps=4,
        grad_accum_steps=2,
        mixed_precision=False,
        log_every=2,
        checkpoint_every=2,
        checkpoint_dir=tmp_path / "ckpts",
        ema_beta=0.9,
        device="cpu",
    )
    trained = train(model, opt, loss_fn, loader, cfg)
    assert trained is not None

    # 'last.pt' plus at least one intermediate checkpoint should exist.
    assert (tmp_path / "ckpts" / "last.pt").exists()

    # Verify we can resume: create a fresh model and restore the checkpoint.
    model2 = TinyVelocityField(channels=1, hidden=16)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    step = load_checkpoint(tmp_path / "ckpts" / "last.pt", model=model2, optimizer=opt2)
    assert step == 4
