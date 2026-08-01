"""Sampling-flow visualisation demo.

Trains a tiny MLP velocity field with flow matching on a 2D "two-moons"
target, then runs :class:`~deltaflow.solvers.euler.EulerSolver` step-by-step
while recording every intermediate state. From that trajectory tensor of
shape ``(n_steps + 1, N, 2)`` we produce four plots that together explain
what the sampler is *doing*:

    snapshots.png      - point clouds at t = 0.0, 0.25, 0.5, 0.75, 1.0
                         (source -> target evolution).
    trajectories.png   - individual particle paths from noise to data,
                         with start/end markers.
    velocity_field.png - quiver plot of v(x, t) at three times, overlaid
                         on the point cloud at that time.
    flow.gif           - animation of the full sampling process (optional;
                         requires ``pillow``).

Run::

    python examples/90-showcase/02-sampling-flow-viz/main.py

Outputs are written to ``outputs/sampling_flow_viz/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
except ImportError as exc:  # pragma: no cover - install hint only
    raise SystemExit(
        "This example needs matplotlib. Install it with:\n"
        "    pip install matplotlib"
    ) from exc

from deltaflow.core.base import BaseVelocityField
from deltaflow.losses import FlowMatchingLoss
from deltaflow.solvers import EulerSolver


# --------------------------------------------------------------------------- #
# Model + target
# --------------------------------------------------------------------------- #

class MLPVelocityField(BaseVelocityField):
    """Small MLP that predicts a 2D velocity from ``(x, t)``."""

    def __init__(self, dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **cond) -> torch.Tensor:
        t_ = t.view(-1, 1)
        return self.net(torch.cat([x, t_], dim=-1))


def two_moons(n: int, noise: float = 0.05) -> torch.Tensor:
    """Classic two-moons dataset, scaled to sit roughly in ``[-2, 2]^2``."""
    n_a = n // 2
    n_b = n - n_a
    theta_a = torch.rand(n_a) * np.pi
    theta_b = torch.rand(n_b) * np.pi
    xa = torch.stack([torch.cos(theta_a), torch.sin(theta_a)], dim=-1)
    xb = torch.stack([1.0 - torch.cos(theta_b), 0.5 - torch.sin(theta_b)], dim=-1)
    x = torch.cat([xa, xb], dim=0)
    x = (x - x.mean(dim=0)) * 2.0
    x = x + noise * torch.randn_like(x)
    return x


# --------------------------------------------------------------------------- #
# Trajectory-recording sampler
# --------------------------------------------------------------------------- #

@torch.no_grad()
def sample_with_trajectory(
    solver: EulerSolver,
    x0: torch.Tensor,
    n_steps: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``solver.step`` in a loop and record every intermediate state.

    Returns:
        traj:  ``(n_steps + 1, N, D)`` tensor of states, ``traj[0] == x0``.
        times: ``(n_steps + 1,)`` tensor of the times each state sits at.
    """
    dt = (t_end - t_start) / n_steps
    traj = [x0.clone()]
    times = [t_start]
    x = x0
    for i in range(n_steps):
        t_val = t_start + i * dt
        x = solver.step(x, t_val, dt)
        traj.append(x.clone())
        times.append(t_val + dt)
    return torch.stack(traj, dim=0), torch.tensor(times)


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #

_XLIM = (-3.5, 3.5)
_YLIM = (-3.0, 3.0)


def _plot_snapshots(
    traj: torch.Tensor,
    times: torch.Tensor,
    target: torch.Tensor,
    out_path: Path,
    fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Path:
    """Point-cloud panels at selected time fractions along the trajectory."""
    n_states = traj.shape[0]
    idxs = [min(n_states - 1, int(round(f * (n_states - 1)))) for f in fractions]

    fig, axes = plt.subplots(1, len(idxs), figsize=(3.0 * len(idxs), 3.2), dpi=130)
    for ax, i in zip(axes, idxs):
        # Faint target reference (helps the eye see where we are heading).
        ax.scatter(
            target[:, 0], target[:, 1],
            s=4, c="lightgray", alpha=0.6, edgecolors="none", label="target",
        )
        ax.scatter(
            traj[i, :, 0], traj[i, :, 1],
            s=6, c="crimson", alpha=0.75, edgecolors="none",
        )
        ax.set_xlim(_XLIM)
        ax.set_ylim(_YLIM)
        ax.set_aspect("equal")
        ax.set_title(f"t = {times[i].item():.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Sampling flow: source (t=0) -> target (t=1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_trajectories(
    traj: torch.Tensor,
    target: torch.Tensor,
    out_path: Path,
    n_lines: int = 200,
    seed: int = 0,
) -> Path:
    """Line plot of a random subset of individual particle paths."""
    rng = np.random.default_rng(seed)
    n_particles = traj.shape[1]
    idx = rng.choice(n_particles, size=min(n_lines, n_particles), replace=False)

    traj_np = traj[:, idx, :].cpu().numpy()  # (T, n_lines, 2)

    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=130)

    ax.scatter(
        target[:, 0], target[:, 1],
        s=4, c="lightgray", alpha=0.6, edgecolors="none", label="target",
    )
    # Paths.
    for k in range(traj_np.shape[1]):
        ax.plot(traj_np[:, k, 0], traj_np[:, k, 1],
                color="steelblue", alpha=0.35, linewidth=0.6)
    # Start and end markers.
    ax.scatter(traj_np[0, :, 0], traj_np[0, :, 1],
               s=10, c="black", alpha=0.7, edgecolors="none", label="start (t=0)")
    ax.scatter(traj_np[-1, :, 0], traj_np[-1, :, 1],
               s=12, c="crimson", alpha=0.9, edgecolors="none", label="end (t=1)")

    ax.set_xlim(_XLIM)
    ax.set_ylim(_YLIM)
    ax.set_aspect("equal")
    ax.set_title(f"Particle trajectories under the learned velocity field  ({traj_np.shape[1]} paths)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_velocity_field(
    field: BaseVelocityField,
    traj: torch.Tensor,
    times: torch.Tensor,
    out_path: Path,
    grid: int = 22,
    fractions: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> Path:
    """Quiver plot of ``v(x, t)`` on a grid at a few time fractions."""
    xs = np.linspace(_XLIM[0], _XLIM[1], grid)
    ys = np.linspace(_YLIM[0], _YLIM[1], grid)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = torch.from_numpy(
        np.stack([xx.ravel(), yy.ravel()], axis=-1)
    ).float()

    n_states = traj.shape[0]
    idxs = [min(n_states - 1, int(round(f * (n_states - 1)))) for f in fractions]

    fig, axes = plt.subplots(1, len(idxs), figsize=(3.6 * len(idxs), 3.6), dpi=130)
    for ax, i in zip(axes, idxs):
        t_val = float(times[i].item())
        t_tensor = torch.full((grid_pts.shape[0],), t_val)
        with torch.no_grad():
            v = field(grid_pts, t_tensor).cpu().numpy()
        u = v[:, 0].reshape(grid, grid)
        w = v[:, 1].reshape(grid, grid)
        speed = np.hypot(u, w)

        ax.quiver(
            xx, yy, u, w, speed,
            cmap="viridis", pivot="mid",
            scale=None, width=0.004, alpha=0.9,
        )
        ax.scatter(
            traj[i, :, 0], traj[i, :, 1],
            s=6, c="crimson", alpha=0.7, edgecolors="none",
        )
        ax.set_xlim(_XLIM)
        ax.set_ylim(_YLIM)
        ax.set_aspect("equal")
        ax.set_title(f"v(x, t)  at t = {t_val:.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Learned velocity field at different times", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_animation(
    traj: torch.Tensor,
    times: torch.Tensor,
    target: torch.Tensor,
    out_path: Path,
    fps: int = 20,
) -> Optional[Path]:
    """Save a GIF of the point cloud evolving from t=0 to t=1.

    Returns ``None`` if the pillow writer isn't available.
    """
    traj_np = traj.cpu().numpy()
    target_np = target.cpu().numpy()

    fig, ax = plt.subplots(figsize=(5.0, 4.8), dpi=110)
    ax.scatter(target_np[:, 0], target_np[:, 1],
               s=4, c="lightgray", alpha=0.6, edgecolors="none")
    scat = ax.scatter([], [], s=8, c="crimson", alpha=0.85, edgecolors="none")
    title = ax.set_title("")
    ax.set_xlim(_XLIM)
    ax.set_ylim(_YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame_idx: int):
        scat.set_offsets(traj_np[frame_idx])
        title.set_text(f"Sampling flow  |  t = {float(times[frame_idx].item()):.2f}")
        return scat, title

    anim = animation.FuncAnimation(
        fig, update, frames=traj_np.shape[0], interval=1000 // fps, blit=False,
    )
    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
    except Exception as exc:  # pillow not installed, or codec missing
        plt.close(fig)
        print(f"[demo] skipping animation ({exc}); install pillow to enable it")
        return None
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- Train a small velocity field on two-moons -----------------------
    field = MLPVelocityField(dim=2, hidden=128)
    loss_fn = FlowMatchingLoss()
    opt = torch.optim.Adam(field.parameters(), lr=2e-3)

    n_train_steps = 2000
    batch_size = 512

    print(f"[demo] training velocity field for {n_train_steps} steps ...")
    for step in range(n_train_steps):
        x1 = two_moons(batch_size)
        loss = loss_fn(field, x1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == n_train_steps - 1:
            print(f"  step={step:4d}  loss={loss.item():.4f}")

    # ---- Sample and record every intermediate state ----------------------
    field.eval()
    solver = EulerSolver(field)
    n_particles = 1000
    n_steps = 50
    x0 = torch.randn(n_particles, 2)
    traj, times = sample_with_trajectory(solver, x0, n_steps=n_steps)
    print(f"[demo] recorded trajectory of shape {tuple(traj.shape)}")

    # A fresh, large sample of the target for the "where are we going" reference.
    target = two_moons(2000)

    # ---- Plots -----------------------------------------------------------
    out_dir = Path("outputs") / "sampling_flow_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots_path = _plot_snapshots(traj, times, target, out_dir / "snapshots.png")
    print(f"[demo] wrote {snapshots_path}")

    traj_path = _plot_trajectories(traj, target, out_dir / "trajectories.png")
    print(f"[demo] wrote {traj_path}")

    vf_path = _plot_velocity_field(field, traj, times, out_dir / "velocity_field.png")
    print(f"[demo] wrote {vf_path}")

    anim_path = _write_animation(traj, times, target, out_dir / "flow.gif")
    if anim_path is not None:
        print(f"[demo] wrote {anim_path}")


if __name__ == "__main__":
    main()
