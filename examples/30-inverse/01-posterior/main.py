"""End-to-end DeltaFlow v1 example: train a velocity field, sample unconditionally
with Euler, then run posterior sampling against a mask (inpainting) - all in
pixel space on a tiny synthetic image dataset.

Run::

    python examples/30-inverse/01-posterior/main.py

The point of this example is *not* image quality (the dataset is
synthetic and the backbone is a toy TinyVelocityField); it is to
demonstrate that the following three pieces snap together as designed:

    1. Train under conditional flow matching, with the coupling strategy
       swapped in via ``trainer.coupling``.
    2. Sample unconditionally with ``EulerSolver``.
    3. Wrap that same Euler solver in ``PosteriorSolver`` and sample from
       ``p(x | y = mask * x_true)`` - the base solver is *reused*, not
       re-implemented.

The last step is repeated once with :class:`OTCoupling` at train time and
once with :class:`IndependentCoupling`, to mirror the ablation Flower
uses to justify OT as a default.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from deltaflow.inverse import GaussianLikelihood, LinearTweedie, MaskOperator
from deltaflow.losses import ConditionalFlowMatchingLoss
from deltaflow.models import TinyVelocityField
from deltaflow.solvers import EulerSolver, PosteriorSolver
from deltaflow.trainer import TrainConfig, IndependentCoupling, OTCoupling, train


def make_toy_dataset(n: int = 128, size: int = 16, seed: int = 0) -> TensorDataset:
    """Synthetic "images": axis-aligned Gaussian blobs at random centres,
    intensity-normalised to ~[-1, 1]. Enough structure for the velocity
    field to learn *something* and for the mask-inpainting task to be
    meaningful."""
    g = torch.Generator().manual_seed(seed)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    imgs = []
    for _ in range(n):
        cx = torch.randint(4, size - 4, (1,), generator=g).item()
        cy = torch.randint(4, size - 4, (1,), generator=g).item()
        sigma = 2.5
        blob = torch.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
        imgs.append((blob * 2.0 - 1.0).unsqueeze(0))
    return TensorDataset(torch.stack(imgs, dim=0))


def train_one(coupling, tag: str, ckpt_root: Path, steps: int = 300) -> torch.nn.Module:
    torch.manual_seed(0)
    ds = make_toy_dataset(n=256, size=16)
    # DataLoader over a TensorDataset returns a (tensor,) tuple; TrainConfig extracts x1.
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

    model = TinyVelocityField(channels=1, hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = ConditionalFlowMatchingLoss(coupling=coupling)

    cfg = TrainConfig(
        max_steps=steps,
        grad_accum_steps=1,
        mixed_precision=False,
        log_every=100,
        checkpoint_every=steps,  # single final checkpoint per variant
        checkpoint_dir=ckpt_root / tag,
        ema_beta=0.999,
        device="cpu",
    )
    return train(model, opt, loss_fn, loader, cfg)


@torch.no_grad()
def unconditional_sample(model, n: int = 4, size: int = 16) -> torch.Tensor:
    solver = EulerSolver(model)
    x = torch.randn(n, 1, size, size)
    return solver.sample(x, n_steps=50, show_progress=False)


def posterior_sample(model, y, mask, size: int = 16, n: int = 4) -> torch.Tensor:
    """Sample from p(x | y = mask * x_true) using PosteriorSolver."""
    base = EulerSolver(model)
    operator = MaskOperator(mask)
    # Larger sigma keeps the per-step likelihood gradient bounded on this toy
    # (untuned) network; grad_normalize rescales it to the base step's norm.
    likelihood = GaussianLikelihood(y=y, operator=operator, sigma=1.0)
    solver = PosteriorSolver(
        base_solver=base,
        likelihood=likelihood,
        tweedie=LinearTweedie(),
        guidance_scale=0.5,
        grad_normalize=True,
    )
    x = torch.randn(n, 1, size, size)
    return solver.sample(x, n_steps=50, show_progress=False)


def reconstruction_error(samples: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean squared error on the *observed* region of the target."""
    diff = (samples - target) * mask
    return diff.pow(2).mean().item()


def main() -> None:
    ckpt_root = Path("checkpoints") / "posterior_example"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    print("[1/3] Training a velocity field with each coupling strategy...")
    model_indep = train_one(IndependentCoupling(), "independent", ckpt_root, steps=300)
    model_ot = train_one(OTCoupling(), "ot", ckpt_root, steps=300)

    print("[2/3] Unconditional Euler sampling (sanity check)...")
    _ = unconditional_sample(model_indep)
    _ = unconditional_sample(model_ot)

    print("[3/3] Posterior sampling under a centre-mask inpainting measurement...")
    # Ground-truth image + partial mask (observe only the outer ring).
    torch.manual_seed(42)
    target = make_toy_dataset(n=1, size=16, seed=42)[0][0].unsqueeze(0)  # (1, 1, 16, 16)
    mask = torch.ones_like(target)
    mask[..., 4:12, 4:12] = 0.0  # centre is unobserved
    y = target * mask

    samples_indep = posterior_sample(model_indep, y, mask, n=8)
    samples_ot = posterior_sample(model_ot, y, mask, n=8)

    err_indep = reconstruction_error(samples_indep, target.expand_as(samples_indep), mask)
    err_ot = reconstruction_error(samples_ot, target.expand_as(samples_ot), mask)

    print()
    print(f"  observed-region MSE, independent-coupled model: {err_indep:.4f}")
    print(f"  observed-region MSE, OT-coupled model:          {err_ot:.4f}")
    print()
    print("Note: this toy setup is not tuned; it demonstrates the API wiring, not SOTA quality.")


if __name__ == "__main__":
    main()
