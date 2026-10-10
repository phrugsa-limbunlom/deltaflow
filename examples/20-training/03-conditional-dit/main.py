"""
20-training/03-conditional-dit: class-conditional generation with a DiT
velocity field and adaLN-Zero conditioning (issue #11).

Two classes draw tiny 8x8 "images": class 0 is a bright top half, class 1 is a
bright bottom half. A small `DiT` is trained with conditional flow matching,
then sampled per class with classifier-free guidance. The per-class pixel means
confirm the label conditioning took effect.

Run: python examples/20-training/03-conditional-dit/main.py
"""

import torch

from deltaflow.losses import FlowMatchingLoss
from deltaflow.models import DiT
from deltaflow.solvers import EulerSolver

IMG = 8
N_CLASSES = 2


def sample_batch(n: int):
    """Return (images, labels): class 0 bright on top, class 1 bright on bottom."""
    y = torch.randint(0, N_CLASSES, (n,))
    x = torch.zeros(n, 1, IMG, IMG)
    half = IMG // 2
    top = y == 0
    x[top, :, :half, :] = 1.0
    x[~top, :, half:, :] = 1.0
    return x + 0.05 * torch.randn_like(x), y


def main():
    torch.manual_seed(0)
    field = DiT(
        input_size=IMG,
        patch_size=2,
        in_channels=1,
        hidden_size=64,
        depth=2,
        num_heads=4,
        num_classes=N_CLASSES,
        class_dropout_prob=0.1,
    )
    loss_fn = FlowMatchingLoss()
    opt = torch.optim.Adam(field.parameters(), lr=2e-3)

    field.train()
    for step in range(300):
        x1, y = sample_batch(64)
        loss = loss_fn(field, x1, y=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step={step:4d}  loss={loss.item():.4f}")

    # Classifier-free-guided sampling, conditioned on each class.
    field.eval()
    for cls in range(N_CLASSES):
        y = torch.full((64,), cls)
        solver = EulerSolver(lambda x, t, y=y: field.forward_with_cfg(x, t, y, cfg_scale=2.0))
        samples = solver.sample(torch.randn(64, 1, IMG, IMG), n_steps=50, show_progress=False)
        half = IMG // 2
        top_mean = samples[:, :, :half, :].mean().item()
        bot_mean = samples[:, :, half:, :].mean().item()
        print(f"class {cls}: top_mean={top_mean:+.3f}  bottom_mean={bot_mean:+.3f}")


if __name__ == "__main__":
    main()
