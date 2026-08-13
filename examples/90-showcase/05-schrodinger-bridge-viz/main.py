"""90-showcase/05-schrodinger-bridge-viz: visualise the Schrödinger-bridge
path itself, then train a velocity field against it and inspect the learned
sampler.

Two things are shown, because they are easy to conflate:

1. **The training-time bridge is stochastic.** For a fixed ``(x0, x1)`` pair,
   :class:`~deltaflow.interpolants.SchrodingerBridgeInterpolant` samples a
   different, noisy path ``x_t`` every time it is called - a Brownian bridge
   wiggling around the straight line, with amplitude set by ``sigma``.
   ``bridge_paths.png`` draws many such samples at a few ``sigma`` values,
   including ``sigma=0`` (the deterministic straight line);
   ``bridge_paths.gif`` animates the same realisations unrolling over ``t``.
2. **The learned sampler is deterministic.** Training regresses a velocity
   field onto the *conditional* target velocity of that stochastic bridge;
   once trained, generation integrates the resulting (deterministic)
   probability-flow ODE with :class:`~deltaflow.solvers.euler.EulerSolver`,
   exactly as for the other flow-matching demos. ``snapshots.png``,
   ``trajectories.png``, and ``flow.gif`` show that post-training behaviour
   on a two-moons target, with the mini-batch OT coupling recommended for
   Schrödinger-bridge training (see the interpolant's docstring).

Run::

    python examples/90-showcase/05-schrodinger-bridge-viz/main.py

Outputs are written to ``outputs/schrodinger_bridge_viz/``.
"""

from __future__ import annotations

from pathlib import Path

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
from deltaflow.interpolants import SchrodingerBridgeInterpolant
from deltaflow.losses import ConditionalFlowMatchingLoss
from deltaflow.solvers import EulerSolver
from deltaflow.trainer import OTCoupling

# Figure palette: deliberately NOT the docs theme primary (purple).
C_TARGET = "#d6dbe6"     # faint target reference (cool neutral)
C_PARTICLE = "#0f9bab"   # particles / samples: teal
C_TRAJ = "#3f9e73"       # trajectories: green
C_START = "#155e63"      # start markers: deep teal
C_BRIDGE = "#b0563f"     # stochastic bridge paths: muted rust

_XLIM = (-3.5, 3.5)
_YLIM = (-3.0, 3.0)


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
# 1. The bridge path itself (no training involved)
# --------------------------------------------------------------------------- #

def _plot_bridge_paths(
    out_path: Path,
    sigmas: tuple[float, ...] = (0.0, 0.5, 1.5),
    n_pairs: int = 6,
    n_draws: int = 12,
    n_t: int = 60,
    seed: int = 0,
) -> Path:
    """For a handful of fixed ``(x0, x1)`` pairs, draw several stochastic
    bridge realisations at each ``sigma`` and overlay them, so the effect of
    ``sigma`` on path "wiggle" is directly visible."""
    torch.manual_seed(seed)
    x0 = torch.randn(n_pairs, 1) * 1.5 - 2.0
    x1 = torch.randn(n_pairs, 1) * 1.5 + 2.0
    ts = torch.linspace(0.0, 1.0, n_t)

    fig, axes = plt.subplots(1, len(sigmas), figsize=(4.2 * len(sigmas), 3.6), dpi=130, sharey=True)
    if len(sigmas) == 1:
        axes = [axes]

    for ax, sigma in zip(axes, sigmas):
        interpolant = SchrodingerBridgeInterpolant(sigma=sigma)
        for p in range(n_pairs):
            for _ in range(n_draws):
                x0_p = x0[p : p + 1].expand(n_t, -1)
                x1_p = x1[p : p + 1].expand(n_t, -1)
                x_t, _ = interpolant.interpolate(x1_p, ts, x0=x0_p)
                ax.plot(ts.numpy(), x_t[:, 0].numpy(), color=C_BRIDGE, alpha=0.35, linewidth=0.9)
            ax.scatter([0.0], x0[p].numpy(), s=18, c=C_START, zorder=3)
            ax.scatter([1.0], x1[p].numpy(), s=18, c=C_PARTICLE, zorder=3)
        ax.set_title(f"sigma = {sigma:g}", fontsize=11)
        ax.set_xlabel("t")
    axes[0].set_ylabel("x")
    fig.suptitle("Schrödinger-bridge conditional paths at increasing diffusivity", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_bridge_paths_animation(
    out_path: Path,
    sigmas: tuple[float, ...] = (0.0, 0.5, 1.5),
    n_pairs: int = 6,
    n_draws: int = 12,
    n_t: int = 60,
    seed: int = 0,
    fps: int = 20,
):
    """Animate the same bridge realisations as :func:`_plot_bridge_paths`,
    drawing each path progressively from ``t=0`` to ``t=1`` across the
    ``sigma`` panels, so the "wiggle vs. sigma" effect plays out over time."""
    torch.manual_seed(seed)
    x0 = torch.randn(n_pairs, 1) * 1.5 - 2.0
    x1 = torch.randn(n_pairs, 1) * 1.5 + 2.0
    ts = torch.linspace(0.0, 1.0, n_t)

    # Precompute every realisation once per sigma: shape (n_pairs, n_draws, n_t).
    all_paths = []
    for sigma in sigmas:
        interpolant = SchrodingerBridgeInterpolant(sigma=sigma)
        paths = torch.empty(n_pairs, n_draws, n_t)
        for p in range(n_pairs):
            for d in range(n_draws):
                x0_p = x0[p : p + 1].expand(n_t, -1)
                x1_p = x1[p : p + 1].expand(n_t, -1)
                x_t, _ = interpolant.interpolate(x1_p, ts, x0=x0_p)
                paths[p, d] = x_t[:, 0]
        all_paths.append(paths.numpy())

    fig, axes = plt.subplots(1, len(sigmas), figsize=(4.2 * len(sigmas), 3.6), dpi=110, sharey=True)
    if len(sigmas) == 1:
        axes = [axes]

    lines = []
    for ax, sigma, paths in zip(axes, sigmas, all_paths):
        ax.set_title(f"sigma = {sigma:g}", fontsize=11)
        ax.set_xlabel("t")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(paths.min() - 0.5, paths.max() + 0.5)
        panel_lines = [
            ax.plot([], [], color=C_BRIDGE, alpha=0.35, linewidth=0.9)[0]
            for _ in range(n_pairs * n_draws)
        ]
        lines.append(panel_lines)
        for p in range(n_pairs):
            ax.scatter([0.0], x0[p].numpy(), s=18, c=C_START, zorder=3)
            ax.scatter([1.0], x1[p].numpy(), s=18, c=C_PARTICLE, zorder=3)
    axes[0].set_ylabel("x")
    suptitle = fig.suptitle("", fontsize=12)
    fig.tight_layout()

    def update(frame_idx: int):
        k = frame_idx + 1
        for panel_lines, paths in zip(lines, all_paths):
            idx = 0
            for p in range(n_pairs):
                for d in range(n_draws):
                    panel_lines[idx].set_data(ts.numpy()[:k], paths[p, d, :k])
                    idx += 1
        suptitle.set_text(
            f"Schrödinger-bridge conditional paths at increasing diffusivity  |  t = {ts[frame_idx].item():.2f}"
        )
        return tuple(l for panel_lines in lines for l in panel_lines) + (suptitle,)

    anim = animation.FuncAnimation(fig, update, frames=n_t, interval=1000 // fps, blit=False)
    try:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    except Exception as exc:
        plt.close(fig)
        print(f"[demo] skipping animation ({exc}); install pillow to enable it")
        return None
    plt.close(fig)
    return out_path

@torch.no_grad()
def sample_with_trajectory(
    solver: EulerSolver, x0: torch.Tensor, n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt = 1.0 / n_steps
    traj = [x0.clone()]
    times = [0.0]
    x = x0
    for i in range(n_steps):
        t_val = i * dt
        x = solver.step(x, t_val, dt)
        traj.append(x.clone())
        times.append(t_val + dt)
    return torch.stack(traj, dim=0), torch.tensor(times)


def _plot_snapshots(
    traj: torch.Tensor, times: torch.Tensor, target: torch.Tensor, out_path: Path,
    fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Path:
    n_states = traj.shape[0]
    idxs = [min(n_states - 1, int(round(f * (n_states - 1)))) for f in fractions]

    fig, axes = plt.subplots(1, len(idxs), figsize=(3.0 * len(idxs), 3.2), dpi=130)
    for ax, i in zip(axes, idxs):
        ax.scatter(target[:, 0], target[:, 1], s=4, c=C_TARGET, alpha=0.6, edgecolors="none", label="target")
        ax.scatter(traj[i, :, 0], traj[i, :, 1], s=6, c=C_PARTICLE, alpha=0.75, edgecolors="none")
        ax.set_xlim(_XLIM); ax.set_ylim(_YLIM)
        ax.set_aspect("equal")
        ax.set_title(f"t = {times[i].item():.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Schrödinger-bridge trained sampler: source (t=0) -> target (t=1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_trajectories(
    traj: torch.Tensor, target: torch.Tensor, out_path: Path, n_lines: int = 200, seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    n_particles = traj.shape[1]
    idx = rng.choice(n_particles, size=min(n_lines, n_particles), replace=False)
    traj_np = traj[:, idx, :].cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=130)
    ax.scatter(target[:, 0], target[:, 1], s=4, c=C_TARGET, alpha=0.6, edgecolors="none", label="target")
    for k in range(traj_np.shape[1]):
        ax.plot(traj_np[:, k, 0], traj_np[:, k, 1], color=C_TRAJ, alpha=0.35, linewidth=0.6)
    ax.scatter(traj_np[0, :, 0], traj_np[0, :, 1], s=10, c=C_START, alpha=0.7, edgecolors="none", label="start (t=0)")
    ax.scatter(traj_np[-1, :, 0], traj_np[-1, :, 1], s=12, c=C_PARTICLE, alpha=0.9, edgecolors="none", label="end (t=1)")
    ax.set_xlim(_XLIM); ax.set_ylim(_YLIM)
    ax.set_aspect("equal")
    ax.set_title(f"Learned Schrödinger-bridge sampler trajectories ({traj_np.shape[1]} paths)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_animation(traj, times, target, out_path, fps: int = 20):
    traj_np = traj.cpu().numpy()
    target_np = target.cpu().numpy()

    fig, ax = plt.subplots(figsize=(5.0, 4.8), dpi=110)
    ax.scatter(target_np[:, 0], target_np[:, 1], s=4, c=C_TARGET, alpha=0.6, edgecolors="none")
    scat = ax.scatter([], [], s=8, c=C_PARTICLE, alpha=0.85, edgecolors="none")
    title = ax.set_title("")
    ax.set_xlim(_XLIM); ax.set_ylim(_YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    def update(frame_idx: int):
        scat.set_offsets(traj_np[frame_idx])
        title.set_text(f"Schrödinger-bridge sampling  |  t = {float(times[frame_idx].item()):.2f}")
        return scat, title

    anim = animation.FuncAnimation(fig, update, frames=traj_np.shape[0], interval=1000 // fps, blit=False)
    try:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    except Exception as exc:
        plt.close(fig)
        print(f"[demo] skipping animation ({exc}); install pillow to enable it")
        return None
    plt.close(fig)
    return out_path


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    out_dir = Path("outputs") / "schrodinger_bridge_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Visualise the bridge path itself, no training required -------
    bridge_path = _plot_bridge_paths(out_dir / "bridge_paths.png")
    print(f"[demo] wrote {bridge_path}")

    bridge_anim_path = _write_bridge_paths_animation(out_dir / "bridge_paths.gif")
    if bridge_anim_path is not None:
        print(f"[demo] wrote {bridge_anim_path}")

    # ---- 2. Train a velocity field against the Schrodinger-bridge path,
    #         with mini-batch OT coupling on (x0, x1) as recommended ------
    field = MLPVelocityField(dim=2, hidden=128)
    loss_fn = ConditionalFlowMatchingLoss(
        interpolant=SchrodingerBridgeInterpolant(sigma=0.5),
        coupling=OTCoupling(),
    )
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

    # ---- 3. Sample: integrate the learned (deterministic) probability-flow
    #         ODE and record every intermediate state -----------------------
    field.eval()
    solver = EulerSolver(field)
    n_particles = 1000
    n_steps = 50
    x0 = torch.randn(n_particles, 2)
    traj, times = sample_with_trajectory(solver, x0, n_steps=n_steps)
    print(f"[demo] recorded trajectory of shape {tuple(traj.shape)}")

    target = two_moons(2000)

    snapshots_path = _plot_snapshots(traj, times, target, out_dir / "snapshots.png")
    print(f"[demo] wrote {snapshots_path}")

    traj_path = _plot_trajectories(traj, target, out_dir / "trajectories.png")
    print(f"[demo] wrote {traj_path}")

    anim_path = _write_animation(traj, times, target, out_dir / "flow.gif")
    if anim_path is not None:
        print(f"[demo] wrote {anim_path}")


if __name__ == "__main__":
    main()
