"""Time-unconditioned flow matching visualisation (arXiv:2605.08344).

Trains two velocity fields on the 2D two-moons target, one time-conditioned
``v(x, t)`` (`ConditionalFlowMatchingLoss`) and one time-blind ``v(x)``
(`TimeUnconditionedFlowMatchingLoss`). In 2D there are no "extra" dimensions,
so time is *not* identifiable and the time-blind model is visibly worse: this
is the paper's caveat. The identifiability plot shows why it works in high
dimensions: when data lies near a low-dimensional subspace of a
high-dimensional space, the interpolation time is recoverable from a single
noisy observation.

    loss_curves.png        training loss of both models.
    samples.png            Euler samples of the conditioned and time-blind
                           models next to the data.
    samples.gif            the same samplers animated from noise to samples.
    time_identifiability.png
                           data on a k=2 subspace of R^d: the energy in the
                           orthogonal directions of ``x_t = (1-t) x0 + t x1``
                           gives ``t_hat = 1 - sqrt(E)``; error shrinks as d
                           grows (rate ~ 1/sqrt(d-k)).
    geometry_3d.png        two-moons embedded on a plane in R^3: the noisy
                           interpolant ``x_t`` is a thick cloud at small t and
                           collapses onto the data plane as t -> 1. Its
                           height above the plane is ``(1-t) * noise``.
    geometry_3d.gif        the same collapse animated over t.
    time_posterior.png     distribution of the off-plane energy for several t,
                           at d=3 (overlapping, t ambiguous) and d=512
                           (separated, t identifiable), and the posterior
                           p(t | x_t) sharpening as d grows.
    velocity_averaging.png the learned 2D fields: v(x, t=0.2), v(x, t=0.8)
                           and the time-blind v(x), which is forced to be a
                           compromise between them.

Run:

    python examples/90-showcase/12-time-unconditioned-fm-viz/main.py

Outputs are written to ``outputs/time_unconditioned_fm_viz/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - install hint only
    raise SystemExit(
        "This example needs matplotlib. Install it with:\n    pip install matplotlib"
    ) from exc

from deltaflow.losses import ConditionalFlowMatchingLoss, TimeUnconditionedFlowMatchingLoss
from deltaflow.models import TimeBlindVelocityField

C_TARGET = "#d6dbe6"
C_COND = "#0f9bab"
C_BLIND = "#3f9e73"
C_ACCENT = "#155e63"
_LIM = (-3.5, 3.5)


def mlp(in_dim: int, hidden: int = 128, out_dim: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class TimeConditionedMLP(nn.Module):
    def __init__(self, dim: int = 2):
        super().__init__()
        self.net = mlp(dim + 1, out_dim=dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **cond) -> torch.Tensor:
        return self.net(torch.cat([x, t.view(-1, 1)], dim=-1))


def two_moons(n: int, noise: float = 0.05) -> torch.Tensor:
    n_a = n // 2
    n_b = n - n_a
    theta_a = torch.rand(n_a) * np.pi
    theta_b = torch.rand(n_b) * np.pi
    xa = torch.stack([torch.cos(theta_a), torch.sin(theta_a)], dim=-1)
    xb = torch.stack([1.0 - torch.cos(theta_b), 0.5 - torch.sin(theta_b)], dim=-1)
    x = torch.cat([xa, xb], dim=0)
    x = (x - x.mean(dim=0)) * 2.0
    return x + noise * torch.randn_like(x)


def train(model, loss_fn, steps: int, batch: int = 256) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(steps):
        loss = loss_fn(model, two_moons(batch))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def euler_sample(model, n: int, steps: int = 50, return_traj: bool = False):
    x = torch.randn(n, 2)
    traj = [x.clone()]
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt)
        x = x + dt * model(x, t)
        traj.append(x.clone())
    return torch.stack(traj) if return_traj else x


def write_gif(target, cond_traj, blind_traj, out: Path, fps: int = 15) -> Path:
    """Animate both samplers integrating from noise next to the data."""
    from matplotlib import animation

    n_frames = cond_traj.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), dpi=100)
    axes[0].scatter(target[:, 0], target[:, 1], s=5, c=C_TARGET, alpha=0.8, edgecolors="none")
    titles = ["data", "time-conditioned", "time-blind"]
    scats = []
    for ax, title in zip(axes, titles):
        ax.set_xlim(_LIM)
        ax.set_ylim(_LIM)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax, color, traj in ((axes[1], C_COND, cond_traj), (axes[2], C_BLIND, blind_traj)):
        scats.append(ax.scatter(traj[0, :, 0], traj[0, :, 1], s=5, c=color, alpha=0.8, edgecolors="none"))
    suptitle = fig.suptitle("", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    def update(i: int):
        scats[0].set_offsets(cond_traj[i].numpy())
        scats[1].set_offsets(blind_traj[i].numpy())
        suptitle.set_text(f"Euler sampling  |  t = {i / (n_frames - 1):.2f}")
        return scats

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out


def plot_losses(cond: list[float], blind: list[float], out: Path) -> Path:
    def smooth(v, k=50):
        return np.convolve(v, np.ones(k) / k, mode="valid")

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=130)
    ax.plot(smooth(cond), color=C_COND, label="time-conditioned  v(x, t)")
    ax.plot(smooth(blind), color=C_BLIND, label="time-blind  v(x)")
    ax.set_xlabel("training step")
    ax.set_ylabel("flow matching loss (smoothed)")
    ax.set_title("Time-blind loss is higher in 2D (time not identifiable)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_samples(target, cond_s, blind_s, out: Path) -> Path:
    panels = [
        ("data", target, C_TARGET),
        ("time-conditioned", cond_s, C_COND),
        ("time-blind", blind_s, C_BLIND),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), dpi=130)
    for ax, (title, pts, color) in zip(axes, panels):
        ax.scatter(pts[:, 0], pts[:, 1], s=5, c=color, alpha=0.8, edgecolors="none")
        ax.set_xlim(_LIM)
        ax.set_ylim(_LIM)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def estimate_time(d: int, k: int = 2, n: int = 2000, seed: int = 0):
    """Recover ``t`` from orthogonal-subspace energy of ``x_t = (1-t) x0 + t x1``.

    Data live in the first ``k`` coordinates, so the remaining ``d-k``
    coordinates are pure noise scaled by ``(1-t)``: their mean square is
    ``(1-t)^2``.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g)
    x0 = torch.randn(n, d, generator=g)
    x1 = torch.zeros(n, d)
    x1[:, :k] = torch.randn(n, k, generator=g)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    energy = xt[:, k:].pow(2).mean(dim=-1)
    return t.numpy(), (1 - energy.sqrt()).clamp(0, 1).numpy()


def plot_identifiability(out: Path) -> Path:
    dims = [4, 16, 64, 256, 1024]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=130)

    t, t_hat = estimate_time(d=256)
    ax1.scatter(t, t_hat, s=4, c=C_COND, alpha=0.5, edgecolors="none")
    ax1.plot([0, 1], [0, 1], color=C_ACCENT, linewidth=1.0)
    ax1.set_xlabel("true t")
    ax1.set_ylabel("estimated t from a single x_t")
    ax1.set_title("Time is readable from x_t  (d=256, k=2)")

    errs = []
    for d in dims:
        t, t_hat = estimate_time(d=d)
        errs.append(np.sqrt(np.mean((t - t_hat) ** 2)))
    ax2.loglog(dims, errs, "o-", color=C_BLIND, label="RMSE of t_hat")
    ref = errs[0] * np.sqrt((dims[0] - 2) / (np.array(dims) - 2))
    ax2.loglog(dims, ref, "--", color=C_ACCENT, label=r"$\propto 1/\sqrt{d-k}$")
    ax2.set_xlabel("ambient dimension d")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Identifiability improves with dimension")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


C_T = plt.get_cmap("viridis")


def _moons_on_plane(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x0, x1)`` in R^3 with data on the ``z = 0`` plane."""
    torch.manual_seed(seed)
    x1 = torch.zeros(n, 3)
    x1[:, :2] = two_moons(n) * 0.6
    return torch.randn(n, 3), x1


def _style_3d(ax, title: str) -> None:
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.5, 2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_zlabel("off-data direction", fontsize=8, labelpad=-10)
    ax.view_init(elev=14, azim=-60)
    ax.set_title(title, fontsize=10)


def _draw_plane(ax) -> None:
    g = np.linspace(-2.5, 2.5, 2)
    gx, gy = np.meshgrid(g, g)
    ax.plot_surface(gx, gy, np.zeros_like(gx), color=C_TARGET, alpha=0.25, linewidth=0)


def _draw_xt(ax, x0, x1, t: float):
    xt = (1 - t) * x0 + t * x1
    h = xt[:, 2].abs().numpy()
    ax.scatter(x1[:, 0], x1[:, 1], 0, s=2, c=C_ACCENT, alpha=0.25, depthshade=False)
    return ax.scatter(
        xt[:, 0], xt[:, 1], xt[:, 2], s=4, c=h, cmap="viridis_r", vmin=0, vmax=2,
        alpha=0.8, depthshade=False,
    )


def plot_geometry_3d(out: Path, ts=(0.1, 0.5, 0.9)) -> Path:
    x0, x1 = _moons_on_plane(1500)
    fig = plt.figure(figsize=(12.0, 4.2), dpi=130)
    for i, t in enumerate(ts):
        ax = fig.add_subplot(1, len(ts), i + 1, projection="3d")
        _draw_plane(ax)
        _draw_xt(ax, x0, x1, t)
        _style_3d(ax, f"t = {t:.1f}   |   off-plane spread = {1 - t:.1f}")
    fig.suptitle(
        "Data lives on a plane. The noisy interpolant x_t collapses onto it as t -> 1,\n"
        "so the height above the plane encodes t.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def write_geometry_gif(out: Path, n_frames: int = 40, fps: int = 12) -> Path:
    from matplotlib import animation

    x0, x1 = _moons_on_plane(1500)
    fig = plt.figure(figsize=(5.2, 4.8), dpi=100)
    ax = fig.add_subplot(projection="3d")

    def update(i: int):
        t = i / (n_frames - 1) * 0.98
        ax.cla()
        _draw_plane(ax)
        _draw_xt(ax, x0, x1, t)
        _style_3d(ax, f"x_t at t = {t:.2f}   (off-plane spread = {1 - t:.2f})")
        return ()

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out


def _time_posterior(s: float, m: int, grid: np.ndarray) -> np.ndarray:
    """Posterior p(t | S) under ``S = (1-t)^2 chi^2_m`` with a uniform prior."""
    a = (1 - grid) ** 2
    log_lik = -m * np.log(1 - grid) - s / (2 * a)
    p = np.exp(log_lik - log_lik.max())
    return p / (p.sum() * (grid[1] - grid[0]))


def plot_time_posterior(out: Path, k: int = 2) -> Path:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), dpi=130)
    ts = (0.2, 0.4, 0.6, 0.8)
    for ax, d in zip(axes[:2], (3, 512)):
        m = d - k
        for t in ts:
            e = (1 - t) ** 2 * rng.chisquare(m, size=4000) / m
            ax.hist(np.sqrt(e), bins=60, range=(0, 2.5), density=True, alpha=0.6,
                    color=C_T(t), label=f"t = {t}")
        ax.set_xlabel(r"off-data RMS  $\|x_t^\perp\| / \sqrt{d-k}$")
        ax.set_yticks([])
        ax.set_title(f"d = {d}: " + ("histograms overlap, t is ambiguous" if d < 10
                                    else "histograms separate, t is identifiable"),
                     fontsize=10)
        ax.legend(fontsize=8)

    grid = np.linspace(0.0, 0.99, 600)
    t_true = 0.5
    for i, d in enumerate((3, 16, 128, 1024)):
        m = d - k
        s = (1 - t_true) ** 2 * rng.chisquare(m)
        axes[2].plot(grid, _time_posterior(s, m, grid), color=C_T(i / 3), label=f"d = {d}")
    axes[2].axvline(t_true, color="k", linestyle="--", linewidth=0.8, label="true t")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel(r"$p(t \mid x_t)$")
    axes[2].set_title("Posterior over t from one x_t sharpens with d", fontsize=10)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


@torch.no_grad()
def plot_velocity_averaging(cond_model, blind_model, target, out: Path, n: int = 22) -> Path:
    g = np.linspace(_LIM[0], _LIM[1], n)
    gx, gy = np.meshgrid(g, g)
    pts = torch.from_numpy(np.stack([gx.ravel(), gy.ravel()], -1)).float()
    fields = [
        ("time-conditioned  v(x, t=0.2)", cond_model(pts, torch.full((len(pts),), 0.2)), C_COND),
        ("time-conditioned  v(x, t=0.8)", cond_model(pts, torch.full((len(pts),), 0.8)), C_COND),
        ("time-blind  v(x)  (one field for all t)", blind_model(pts), C_BLIND),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), dpi=130)
    for ax, (title, v, color) in zip(axes, fields):
        v = v.numpy()
        ax.scatter(target[:, 0], target[:, 1], s=3, c=C_TARGET, edgecolors="none")
        ax.quiver(gx, gy, v[:, 0].reshape(n, n), v[:, 1].reshape(n, n), color=color,
                  pivot="mid", width=0.004)
        ax.set_xlim(_LIM)
        ax.set_ylim(_LIM)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
    fig.suptitle(
        "In 2D the same x is visited at many t, so the time-blind field averages "
        "the velocities it should switch between",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    steps = 2500

    cond_model = TimeConditionedMLP()
    blind_model = TimeBlindVelocityField(mlp(2))

    print(f"[demo] training both models for {steps} steps ...")
    cond_losses = train(cond_model, ConditionalFlowMatchingLoss(), steps)
    blind_losses = train(blind_model, TimeUnconditionedFlowMatchingLoss(), steps)
    print(
        f"  final loss: conditioned={np.mean(cond_losses[-100:]):.4f}  "
        f"blind={np.mean(blind_losses[-100:]):.4f}"
    )

    cond_traj = euler_sample(cond_model, 2000, return_traj=True)
    blind_traj = euler_sample(blind_model, 2000, return_traj=True)
    cond_s, blind_s = cond_traj[-1], blind_traj[-1]
    target = two_moons(2000)

    out_dir = Path("outputs") / "time_unconditioned_fm_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[demo] wrote {write_gif(target, cond_traj, blind_traj, out_dir / 'samples.gif')}")
    print(f"[demo] wrote {plot_losses(cond_losses, blind_losses, out_dir / 'loss_curves.png')}")
    print(f"[demo] wrote {plot_samples(target, cond_s, blind_s, out_dir / 'samples.png')}")
    print(f"[demo] wrote {plot_identifiability(out_dir / 'time_identifiability.png')}")
    print(f"[demo] wrote {plot_geometry_3d(out_dir / 'geometry_3d.png')}")
    print(f"[demo] wrote {write_geometry_gif(out_dir / 'geometry_3d.gif')}")
    print(f"[demo] wrote {plot_time_posterior(out_dir / 'time_posterior.png')}")
    vel_path = plot_velocity_averaging(
        cond_model, blind_model, target, out_dir / "velocity_averaging.png"
    )
    print(f"[demo] wrote {vel_path}")


if __name__ == "__main__":
    main()
