"""Equilibrium Matching training and visualisation demo.

Trains a tiny noise-unconditional field with Equilibrium Matching (EqM) on the
same 2D *two-moons* target as the flow-matching showcase
(``examples/90-showcase/02-sampling-flow-viz``), then samples it by gradient
descent on the learned landscape (no ODE integration). Using the same dataset
makes the contrast with flow matching direct. EqM discards time-conditional
dynamics, so the learned field ``f(x)`` is time-invariant and a single picture
of it describes the whole sampler. That is the story these plots tell.

    energy_landscape.png   the time-invariant field ``f(x)`` as a quiver, over
                           a heatmap of the gradient norm ``|f(x)|``. The norm
                           collapses toward zero on the two moons, so those
                           points are the equilibria (minima) of the implicit
                           energy the field descends.
    gd_snapshots.png       point clouds at several gradient-descent iterations
                           (pure noise settling onto the moons).
    gd_trajectories.png    individual particle paths under gradient descent,
                           with start and end markers.
    eqm_sampling.gif       animation of the optimisation-based sampler
                           (optional, needs pillow).

Because the field is time-invariant, contrast this with the flow-matching
showcase, whose velocity field needs one quiver panel per time. Here one
landscape is enough.

Run:

    python examples/90-showcase/09-equilibrium-matching-viz/main.py

Outputs are written to ``outputs/equilibrium_matching_viz/``.
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
        "This example needs matplotlib. Install it with:\n" "    pip install matplotlib"
    ) from exc

from deltaflow.core import BaseEquilibriumField
from deltaflow.interpolants import EquilibriumInterpolant
from deltaflow.losses import EquilibriumMatchingLoss

# Figure palette: deliberately NOT the docs theme primary (blue).
# Reuses the logo's non-primary teal/green companions so figures stay on-brand.
C_TARGET = "#d6dbe6"  # faint target reference (cool neutral)
C_PARTICLE = "#0f9bab"  # particles / samples: teal (non-primary)
C_TRAJ = "#3f9e73"  # trajectories: green
C_START = "#155e63"  # start markers: deep teal
CMAP = "viridis"  # gradient-norm heatmap (purple-to-yellow, non-primary)


# --------------------------------------------------------------------------- #
# Model + target
# --------------------------------------------------------------------------- #


class EquilibriumField(BaseEquilibriumField):
    """Noise-unconditional gradient field ``f(x)`` (no time, gamma is implicit)."""

    def __init__(self, dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, **cond) -> torch.Tensor:
        # The field is time-invariant, it takes no time or gamma argument.
        return self.net(x)


def two_moons(n: int, noise: float = 0.05) -> torch.Tensor:
    """Classic two-moons dataset, scaled to sit roughly in ``[-2, 2]^2``.

    Identical to the flow-matching showcase so the two demos can be compared
    directly.
    """
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
# Trajectory-recording gradient-descent sampler
# --------------------------------------------------------------------------- #


@torch.no_grad()
def sample_gd_with_trajectory(
    field: BaseEquilibriumField,
    x0: torch.Tensor,
    n_steps: int,
    step_size: float,
    momentum: float = 0.0,
) -> torch.Tensor:
    """Gradient descent on the landscape, recording every intermediate state.

    Mirrors `EquilibriumSolver` (vanilla GD, or NAG-GD when
    ``momentum > 0``) but stores each iterate so we can plot the path.

    Returns:
        traj: ``(n_steps + 1, N, D)`` tensor of states, ``traj[0] == x0``.
    """

    def grad(x: torch.Tensor) -> torch.Tensor:
        return field(x)

    x = x0.clone()
    m = torch.zeros_like(x)
    traj = [x.clone()]
    for _ in range(n_steps):
        if momentum != 0.0:
            m = grad(x + step_size * momentum * m)
            g = m
        else:
            g = grad(x)
        x = x + step_size * g
        traj.append(x.clone())
    return torch.stack(traj, dim=0)


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #

_LIM = (-3.5, 3.5)


def _plot_energy_landscape(
    field: BaseEquilibriumField,
    target: torch.Tensor,
    out_path: Path,
    grid: int = 60,
    quiver_grid: int = 26,
) -> Path:
    """Quiver of the time-invariant field over a heatmap of ``|f(x)|``."""
    # Dense grid for the gradient-norm heatmap.
    xs = np.linspace(_LIM[0], _LIM[1], grid)
    ys = np.linspace(_LIM[0], _LIM[1], grid)
    xx, yy = np.meshgrid(xs, ys)
    pts = torch.from_numpy(np.stack([xx.ravel(), yy.ravel()], axis=-1)).float()
    with torch.no_grad():
        v = field(pts)
    norm = v.norm(dim=-1).cpu().numpy().reshape(grid, grid)

    # Coarser grid for the arrows.
    qs = np.linspace(_LIM[0], _LIM[1], quiver_grid)
    qxx, qyy = np.meshgrid(qs, qs)
    qpts = torch.from_numpy(np.stack([qxx.ravel(), qyy.ravel()], axis=-1)).float()
    with torch.no_grad():
        qv = field(qpts).cpu().numpy()
    u = qv[:, 0].reshape(quiver_grid, quiver_grid)
    w = qv[:, 1].reshape(quiver_grid, quiver_grid)

    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=130)
    mesh = ax.pcolormesh(xx, yy, norm, cmap=CMAP, shading="auto", alpha=0.9)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="|f(x)|  (0 at equilibria)")
    ax.quiver(qxx, qyy, u, w, color="white", pivot="mid", width=0.003, alpha=0.7)
    ax.scatter(
        target[:, 0], target[:, 1], s=6, c=C_TARGET, alpha=0.7, edgecolors="none", label="data"
    )
    ax.set_xlim(_LIM)
    ax.set_ylim(_LIM)
    ax.set_aspect("equal")
    ax.set_title("Time-invariant EqM landscape, arrows point downhill to the two moons")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_snapshots(
    traj: torch.Tensor,
    target: torch.Tensor,
    out_path: Path,
    fractions: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0),
) -> Path:
    """Point-cloud panels at selected gradient-descent iterations."""
    n_states = traj.shape[0]
    idxs = [min(n_states - 1, int(round(f * (n_states - 1)))) for f in fractions]

    fig, axes = plt.subplots(1, len(idxs), figsize=(3.0 * len(idxs), 3.2), dpi=130)
    for ax, i in zip(axes, idxs):
        ax.scatter(target[:, 0], target[:, 1], s=4, c=C_TARGET, alpha=0.6, edgecolors="none")
        ax.scatter(
            traj[i, :, 0], traj[i, :, 1], s=6, c=C_PARTICLE, alpha=0.75, edgecolors="none"
        )
        ax.set_xlim(_LIM)
        ax.set_ylim(_LIM)
        ax.set_aspect("equal")
        ax.set_title(f"step {i}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Gradient-descent sampling, noise settles onto the moons", fontsize=11)
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
    """Line plot of a random subset of gradient-descent particle paths."""
    rng = np.random.default_rng(seed)
    n_particles = traj.shape[1]
    idx = rng.choice(n_particles, size=min(n_lines, n_particles), replace=False)
    traj_np = traj[:, idx, :].cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=130)
    ax.scatter(
        target[:, 0], target[:, 1], s=4, c=C_TARGET, alpha=0.6, edgecolors="none", label="data"
    )
    for k in range(traj_np.shape[1]):
        ax.plot(traj_np[:, k, 0], traj_np[:, k, 1], color=C_TRAJ, alpha=0.35, linewidth=0.6)
    ax.scatter(
        traj_np[0, :, 0],
        traj_np[0, :, 1],
        s=10,
        c=C_START,
        alpha=0.7,
        edgecolors="none",
        label="start (noise)",
    )
    ax.scatter(
        traj_np[-1, :, 0],
        traj_np[-1, :, 1],
        s=12,
        c=C_PARTICLE,
        alpha=0.9,
        edgecolors="none",
        label="end (sample)",
    )
    ax.set_xlim(_LIM)
    ax.set_ylim(_LIM)
    ax.set_aspect("equal")
    ax.set_title(f"Optimisation-based sampling paths  ({traj_np.shape[1]} particles)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_animation(
    traj: torch.Tensor,
    target: torch.Tensor,
    out_path: Path,
    fps: int = 20,
) -> Optional[Path]:
    """Save a GIF of the point cloud descending onto the modes."""
    traj_np = traj.cpu().numpy()
    target_np = target.cpu().numpy()

    fig, ax = plt.subplots(figsize=(5.0, 4.8), dpi=110)
    ax.scatter(target_np[:, 0], target_np[:, 1], s=4, c=C_TARGET, alpha=0.6, edgecolors="none")
    scat = ax.scatter([], [], s=8, c=C_PARTICLE, alpha=0.85, edgecolors="none")
    title = ax.set_title("")
    ax.set_xlim(_LIM)
    ax.set_ylim(_LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame_idx: int):
        scat.set_offsets(traj_np[frame_idx])
        title.set_text(f"EqM gradient-descent sampling  |  step {frame_idx}")
        return scat, title

    anim = animation.FuncAnimation(
        fig, update, frames=traj_np.shape[0], interval=1000 // fps, blit=False
    )
    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
    except Exception as exc:  # pillow not installed, or codec missing
        plt.close(fig)
        print(f"[demo] skipping animation ({exc}), install pillow to enable it")
        return None
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- Train an EqM field on the two-moons target ----------------------
    field = EquilibriumField(dim=2, hidden=128)
    loss_fn = EquilibriumMatchingLoss(interpolant=EquilibriumInterpolant())
    opt = torch.optim.Adam(field.parameters(), lr=2e-3)

    n_train_steps = 2500
    batch_size = 256

    print(f"[demo] training EqM field for {n_train_steps} steps ...")
    for step in range(n_train_steps):
        x1 = two_moons(batch_size)
        loss = loss_fn(field, x1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == n_train_steps - 1:
            print(f"  step={step:4d}  loss={loss.item():.4f}")

    # ---- Optimisation-based sampling with recorded trajectory ------------
    field.eval()
    x0 = torch.randn(1000, 2)
    traj = sample_gd_with_trajectory(field, x0, n_steps=16, step_size=0.05, momentum=0.0)
    print(f"[demo] recorded gradient-descent trajectory of shape {tuple(traj.shape)}")

    # A fresh, large sample of the target for reference in the plots.
    target = two_moons(2000)

    # Sanity print, sample statistics should track the two-moons target.
    final = traj[-1]
    print(f"[demo] sample mean={final.mean(dim=0).tolist()} std={final.std(dim=0).tolist()}")

    # ---- Plots -----------------------------------------------------------
    out_dir = Path("outputs") / "equilibrium_matching_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo] wrote {_plot_energy_landscape(field, target, out_dir / 'energy_landscape.png')}")
    print(f"[demo] wrote {_plot_snapshots(traj, target, out_dir / 'gd_snapshots.png')}")
    print(f"[demo] wrote {_plot_trajectories(traj, target, out_dir / 'gd_trajectories.png')}")

    anim_path = _write_animation(traj, target, out_dir / "eqm_sampling.gif")
    if anim_path is not None:
        print(f"[demo] wrote {anim_path}")


if __name__ == "__main__":
    main()
