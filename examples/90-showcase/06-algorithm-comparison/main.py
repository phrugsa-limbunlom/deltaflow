"""90-showcase/06-algorithm-comparison: train all four DeltaFlow paths on the
same target and compare them side by side.

Every algorithm below is the *same* training loop (same MLP, same optimizer,
same number of steps, same two-moons target); only the interpolant/coupling
combination passed to :class:`~deltaflow.losses.ConditionalFlowMatchingLoss`
changes:

    Linear (independent)    - LinearInterpolant(),                    no coupling
    OT coupling             - LinearInterpolant(),                    OTCoupling()
    Variance-preserving     - VariancePreservingInterpolant(),        no coupling
    Schrödinger bridge      - SchrodingerBridgeInterpolant(sigma=.5), OTCoupling()

That is the point of keeping "path" (`interpolant=`) and "coupling"
(`coupling=`) as independent, swappable arguments (see
`deltaflow.trainer.coupling`): every algorithm in the library is a
configuration of the same training loop, not a separate code path.

Run::

    python examples/90-showcase/06-algorithm-comparison/main.py

Outputs are written to ``outputs/algorithm_comparison/``, including an
animated ``comparison.gif`` showing all four samplers running side by side.
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
        "This example needs matplotlib. Install it with:\n"
        "    pip install matplotlib"
    ) from exc

from matplotlib import animation

from deltaflow.core.base import BaseVelocityField
from deltaflow.interpolants import (
    LinearInterpolant,
    SchrodingerBridgeInterpolant,
    VariancePreservingInterpolant,
)
from deltaflow.losses import ConditionalFlowMatchingLoss
from deltaflow.solvers import EulerSolver
from deltaflow.trainer import OTCoupling

C_TARGET = "#d6dbe6"     # faint target reference (cool neutral)
C_PARTICLE = "#0f9bab"   # trained samples: teal
C_TRAJ = "#3f9e73"       # trajectories: green
C_START = "#155e63"      # start markers: deep teal

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


@torch.no_grad()
def sample_with_trajectory(
    solver: EulerSolver, x0: torch.Tensor, n_steps: int,
) -> torch.Tensor:
    dt = 1.0 / n_steps
    traj = [x0.clone()]
    x = x0
    for i in range(n_steps):
        x = solver.step(x, i * dt, dt)
        traj.append(x.clone())
    return torch.stack(traj, dim=0)  # (n_steps + 1, N, 2)


def train_one(loss_fn, n_train_steps: int, batch_size: int, seed: int) -> tuple[MLPVelocityField, float]:
    torch.manual_seed(seed)
    field = MLPVelocityField(dim=2, hidden=128)
    opt = torch.optim.Adam(field.parameters(), lr=2e-3)
    loss_val = float("nan")
    for step in range(n_train_steps):
        x1 = two_moons(batch_size)
        loss = loss_fn(field, x1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_val = loss.item()
    return field, loss_val


ALGORITHMS = {
    "Linear (independent)": lambda: ConditionalFlowMatchingLoss(
        interpolant=LinearInterpolant(),
    ),
    "OT coupling": lambda: ConditionalFlowMatchingLoss(
        interpolant=LinearInterpolant(),
        coupling=OTCoupling(),
    ),
    "Variance-preserving": lambda: ConditionalFlowMatchingLoss(
        interpolant=VariancePreservingInterpolant(),
    ),
    "Schrodinger bridge": lambda: ConditionalFlowMatchingLoss(
        interpolant=SchrodingerBridgeInterpolant(sigma=0.5),
        coupling=OTCoupling(),
    ),
}


def main() -> None:
    np.random.seed(0)
    out_dir = Path("outputs") / "algorithm_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train_steps = 1500
    batch_size = 512
    n_particles = 1000
    n_steps = 50

    target = two_moons(2000)

    results = {}
    for name, make_loss in ALGORITHMS.items():
        print(f"[demo] training {name!r} for {n_train_steps} steps ...")
        loss_fn = make_loss()
        field, final_loss = train_one(loss_fn, n_train_steps, batch_size, seed=0)
        field.eval()
        solver = EulerSolver(field)
        torch.manual_seed(1)  # same x0 for every algorithm, for a fair comparison
        x0 = torch.randn(n_particles, 2)
        traj = sample_with_trajectory(solver, x0, n_steps=n_steps)
        results[name] = (traj, final_loss)
        print(f"  final loss={final_loss:.4f}")

    # ---- Final-sample comparison -----------------------------------------
    fig, axes = plt.subplots(1, len(results), figsize=(3.4 * len(results), 3.6), dpi=130)
    for ax, (name, (traj, final_loss)) in zip(axes, results.items()):
        ax.scatter(target[:, 0], target[:, 1], s=4, c=C_TARGET, alpha=0.6, edgecolors="none")
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=6, c=C_PARTICLE, alpha=0.75, edgecolors="none")
        ax.set_xlim(_XLIM); ax.set_ylim(_YLIM)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\nfinal loss = {final_loss:.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Same training loop, four interpolant/coupling configurations", fontsize=12)
    fig.tight_layout()
    comparison_path = out_dir / "comparison.png"
    fig.savefig(comparison_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo] wrote {comparison_path}")

    # ---- Trajectory comparison ---------------------------------------------
    rng = np.random.default_rng(0)
    idx = rng.choice(n_particles, size=min(150, n_particles), replace=False)

    fig, axes = plt.subplots(1, len(results), figsize=(3.4 * len(results), 3.6), dpi=130)
    for ax, (name, (traj, _)) in zip(axes, results.items()):
        traj_np = traj[:, idx, :].cpu().numpy()
        ax.scatter(target[:, 0], target[:, 1], s=4, c=C_TARGET, alpha=0.6, edgecolors="none")
        for k in range(traj_np.shape[1]):
            ax.plot(traj_np[:, k, 0], traj_np[:, k, 1], color=C_TRAJ, alpha=0.35, linewidth=0.6)
        ax.scatter(traj_np[0, :, 0], traj_np[0, :, 1], s=8, c=C_START, alpha=0.7, edgecolors="none")
        ax.scatter(traj_np[-1, :, 0], traj_np[-1, :, 1], s=10, c=C_PARTICLE, alpha=0.9, edgecolors="none")
        ax.set_xlim(_XLIM); ax.set_ylim(_YLIM)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Sampled trajectories under each configuration", fontsize=12)
    fig.tight_layout()
    traj_comparison_path = out_dir / "trajectories_comparison.png"
    fig.savefig(traj_comparison_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo] wrote {traj_comparison_path}")

    # ---- Animated comparison: sampling side by side ------------------------
    anim_path = _write_comparison_animation(results, target, out_dir / "comparison.gif")
    if anim_path is not None:
        print(f"[demo] wrote {anim_path}")


def _write_comparison_animation(results, target, out_path, fps: int = 20):
    """Animate every algorithm's sampling process side by side, so the noise
    -> two-moons transport can be compared frame by frame."""
    target_np = target.cpu().numpy()
    names = list(results.keys())
    n_frames = next(iter(results.values()))[0].shape[0]

    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.2), dpi=80)
    scats = []
    for ax, name in zip(axes, names):
        ax.scatter(target_np[:, 0], target_np[:, 1], s=4, c=C_TARGET, alpha=0.6, edgecolors="none")
        scat = ax.scatter([], [], s=6, c=C_PARTICLE, alpha=0.8, edgecolors="none")
        scats.append(scat)
        ax.set_xlim(_XLIM); ax.set_ylim(_YLIM)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    suptitle = fig.suptitle("", fontsize=12)
    fig.tight_layout()

    def update(frame_idx: int):
        t_val = frame_idx / (n_frames - 1)
        for scat, name in zip(scats, names):
            traj = results[name][0]
            scat.set_offsets(traj[frame_idx].cpu().numpy())
        suptitle.set_text(f"Sampling under each configuration  |  t = {t_val:.2f}")
        return (*scats, suptitle)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
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
