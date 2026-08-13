"""90-showcase/04-inverse-posterior-viz: visualise posterior sampling for an
inverse problem (centre-mask inpainting).

We train a tiny velocity field on synthetic Gaussian-blob "images" with
conditional flow matching, then reconstruct an image whose centre is masked
out. `PosteriorSolver` wraps the same Euler solver used for unconditional
sampling and injects a measurement-likelihood gradient at every step, so the
base integrator is reused, not re-implemented.

The figure shows the ground truth, the masked measurement the solver actually
sees, the posterior-mean reconstruction, and the absolute error.

Run: python examples/90-showcase/04-inverse-posterior-viz/main.py
"""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from deltaflow.inverse import GaussianLikelihood, LinearTweedie, MaskOperator
from deltaflow.losses import ConditionalFlowMatchingLoss
from deltaflow.models import TinyVelocityField
from deltaflow.solvers import EulerSolver, PosteriorSolver
from deltaflow.trainer import TrainConfig, OTCoupling, train

SIZE = 16
CMAP = "viridis"          # image colormap: purple-blue-green-yellow (theme vibe)
CMAP_ERR = "magma"        # error colormap


def make_toy_dataset(n: int, size: int = SIZE, seed: int = 0) -> TensorDataset:
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
        blob = torch.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * 2.5 ** 2))
        imgs.append((blob * 2.0 - 1.0).unsqueeze(0))
    return TensorDataset(torch.stack(imgs, dim=0))


def train_field(steps: int = 400) -> torch.nn.Module:
    torch.manual_seed(0)
    ds = make_toy_dataset(n=256)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    model = TinyVelocityField(channels=1, hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = ConditionalFlowMatchingLoss(coupling=OTCoupling())
    cfg = TrainConfig(
        max_steps=steps, grad_accum_steps=1, mixed_precision=False,
        log_every=100, checkpoint_every=steps,
        checkpoint_dir=Path("outputs") / "inverse_posterior_viz" / "ckpt",
        ema_beta=0.999, device="cpu",
    )
    return train(model, opt, loss_fn, loader, cfg)


def posterior_reconstruct(model, y, mask, n: int = 16) -> torch.Tensor:
    base = EulerSolver(model)
    likelihood = GaussianLikelihood(y=y, operator=MaskOperator(mask), sigma=1.0)
    solver = PosteriorSolver(
        base_solver=base, likelihood=likelihood, tweedie=LinearTweedie(),
        guidance_scale=0.5, grad_normalize=True,
    )
    x = torch.randn(n, 1, SIZE, SIZE)
    return solver.sample(x, n_steps=60, show_progress=False)


def posterior_reconstruct_traj(model, y, mask, n: int = 16, n_steps: int = 60):
    """Step the posterior solver manually so we can record the running mean."""
    base = EulerSolver(model)
    likelihood = GaussianLikelihood(y=y, operator=MaskOperator(mask), sigma=1.0)
    solver = PosteriorSolver(
        base_solver=base, likelihood=likelihood, tweedie=LinearTweedie(),
        guidance_scale=0.5, grad_normalize=True,
    )
    x = torch.randn(n, 1, SIZE, SIZE)
    dt = 1.0 / n_steps
    frames = [x.mean(dim=0, keepdim=True).detach().clone()]
    for i in range(n_steps):
        x = solver.step(x, i * dt, dt)
        frames.append(x.mean(dim=0, keepdim=True).detach().clone())
    return x.detach(), frames


def show(ax, img, title, cmap=CMAP, vmin=-1.0, vmax=1.0):
    ax.imshow(img.detach().squeeze().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    print("[1/2] Training a tiny velocity field (OT coupling)...")
    model = train_field(steps=400)

    print("[2/2] Posterior sampling under a centre-mask measurement...")
    torch.manual_seed(42)
    target = make_toy_dataset(n=1, seed=42)[0][0].unsqueeze(0)  # (1, 1, 16, 16)
    mask = torch.ones_like(target)
    mask[..., 4:12, 4:12] = 0.0                                  # centre unobserved
    y = target * mask

    samples, mean_frames = posterior_reconstruct_traj(model, y, mask, n=16)
    recon = samples.mean(dim=0, keepdim=True)                    # posterior mean
    error = (recon - target).abs()

    # Show the measurement with the masked centre greyed to mid-tone for clarity.
    y_show = y.clone()
    y_show[mask == 0] = 0.0

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    show(axes[0], target, "Ground truth")
    show(axes[1], y_show, "Measurement $y = M \\odot x$")
    show(axes[2], recon, "Posterior mean reconstruction")
    show(axes[3], error, "Absolute error", cmap=CMAP_ERR, vmin=0.0, vmax=1.0)
    fig.suptitle(
        "Posterior sampling reconstructs the masked centre without retraining",
        fontsize=13,
    )
    fig.tight_layout()

    out_dir = Path("outputs") / "inverse_posterior_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inverse_posterior.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[demo] wrote {out_path}")
    print(f"[demo] masked-region MAE = {error[mask == 0].mean().item():.4f}")

    gif_path = write_animation(target, y_show, mean_frames,
                               out_dir / "inverse_posterior.gif")
    if gif_path is not None:
        print(f"[demo] wrote {gif_path}")


def write_animation(target, y_show, mean_frames, out_path, fps=15):
    """Animate the posterior mean forming from noise into the reconstruction."""
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
    show(axes[0], target, "Ground truth")
    show(axes[1], y_show, "Measurement $y = M \\odot x$")
    im = axes[2].imshow(mean_frames[0].squeeze().numpy(), cmap=CMAP,
                        vmin=-1.0, vmax=1.0)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    n_frames = len(mean_frames)

    def update(k):
        im.set_data(mean_frames[k].squeeze().numpy())
        step = int(round(k / (n_frames - 1) * 60))
        axes[2].set_title(f"Posterior mean  |  step {step:>2}/60", fontsize=11)
        return (im,)

    fig.suptitle("Posterior sampling fills the masked centre step by step",
                 fontsize=13)
    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000 // fps, blit=False)
    try:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    except Exception as exc:
        plt.close(fig)
        print(f"[demo] skipping animation ({exc}); install pillow to enable it")
        return None
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
