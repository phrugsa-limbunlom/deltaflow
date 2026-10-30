"""90-showcase/11-active-flow-expansion-viz: watch ActFlow grow a flow's generable set.

The setting is a 2D toy in the spirit of the illustrative experiment in De
Santi et al., "Active Flow Expansion for Out-of-Distribution Discovery: from
Theory to Molecules" (2026). The valid design space is an annulus, judged by
a black-box verifier. The pre-training data covers only a 60 degree arc of
it, so the pre-trained flow can only generate that arc.

Two continued pre-training runs start from the same weights with the same
budget.

- **Self-training** (``beta=math.inf``) samples the model as is, keeps what
  the verifier accepts, and fine-tunes. It reinforces the arc it already
  generates.
- **ActFlow** (``beta=0.1``) tilts each batch toward high verifier
  uncertainty in the flow's own noised representation, so queries land on the
  frontier of the generable set and the flow spreads around the ring.

Outputs (in ``outputs/active_flow_expansion_viz/``).

- ``expansion_rounds.png``, model samples and each round's verifier-labelled
  queries, self-training (top) against ActFlow (bottom).
- ``uncertainty_rounds.png``, the uncertainty ``sigma_t(phi_s(x))`` over the
  plane at several rounds, with every query so far. Low uncertainty follows
  the queries, the bright frontier is where ActFlow samples next.
- ``coverage.png``, generable-set coverage and validity per round.
- ``expansion.gif``, both runs animated round by round.

Run::

    python examples/90-showcase/11-active-flow-expansion-viz/main.py
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Wedge
except ImportError as exc:  # pragma: no cover - install hint only
    raise SystemExit(
        "This example needs matplotlib. Install it with:\n    pip install matplotlib"
    ) from exc

from deltaflow.core import BaseVelocityField
from deltaflow.expansion import ActFlow, FlowRepresentation, KernelUncertainty, generable_coverage
from deltaflow.losses import ConditionalFlowMatchingLoss, SignedReplayFlowMatchingLoss
from deltaflow.solvers import EulerSolver

# Figure palette, deliberately NOT the docs theme primary (blue).
C_VALID = "#d6dbe6"  # valid region: cool neutral
C_DATA = "#155e63"  # pre-training arc outline: deep teal
C_PARTICLE = "#0f9bab"  # model samples: teal
C_ACCEPT = "#3f9e73"  # accepted queries / ActFlow: green
C_REJECT = "#e0b13c"  # rejected queries: yellow
C_BASELINE = "#7a5cc0"  # self-training curve: purple

R_IN, R_OUT = 1.5, 2.5
HALF_ANGLE = math.pi / 6
BOUNDS = (-3.5, 3.5)
ROUNDS = 20
SNAP_ROUNDS = (0, 2, 5, 20)
SIGMA_ROUNDS = (1, 3, 8, 20)
N_QUERIES = 128


def verifier(x: torch.Tensor) -> torch.Tensor:
    """Black-box validity oracle, the annulus R_IN <= |x| <= R_OUT."""
    r = x.norm(dim=-1)
    return (r >= R_IN) & (r <= R_OUT)


def arc_data(n: int) -> torch.Tensor:
    theta = (torch.rand(n) * 2 - 1) * HALF_ANGLE
    r = R_IN + (R_OUT - R_IN) * torch.rand(n)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)


class MLPVelocityField(BaseVelocityField):
    def __init__(self, dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **cond) -> torch.Tensor:
        return self.net(torch.cat([x, t.view(-1, 1)], dim=-1))


def pretrain(steps: int = 1500) -> MLPVelocityField:
    field = MLPVelocityField()
    opt = torch.optim.Adam(field.parameters(), lr=2e-3)
    loss_fn = ConditionalFlowMatchingLoss()
    data = arc_data(4000)
    for _ in range(steps):
        loss = loss_fn(field, data[torch.randint(len(data), (256,))])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return field


def make_actflow(field: MLPVelocityField, beta: float) -> ActFlow:
    return ActFlow(
        field,
        verifier,
        torch.optim.Adam(field.parameters(), lr=1e-3),
        sample_shape=(2,),
        representation=FlowRepresentation(field, s=0.9, layer="net.3"),
        uncertainty=KernelUncertainty(kernel="linear", reg=1e-2),
        loss=SignedReplayFlowMatchingLoss(alpha=0.05, alpha_mode="grad_norm"),
        beta=beta,
        guidance_scale=1.0,
        n_steps=30,
        finetune_steps=60,
        batch_size=128,
    )


@torch.no_grad()
def model_samples(field: MLPVelocityField, x0: torch.Tensor) -> torch.Tensor:
    return EulerSolver(field).sample(x0.clone(), n_steps=40, show_progress=False)


@torch.no_grad()
def sigma_grid(act: ActFlow, n: int = 90, n_noise: int = 4) -> torch.Tensor:
    """Mean of sigma_t(phi_s(x)) over a few noise draws, on an n x n grid."""
    g = torch.linspace(*BOUNDS, n)
    xx, yy = torch.meshgrid(g, g, indexing="xy")
    pts = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    gen = torch.Generator().manual_seed(0)
    sig = torch.zeros(pts.shape[0])
    for _ in range(n_noise):
        noise = torch.randn(pts.shape, generator=gen)
        sig += act.uncertainty(act.representation(pts, noise=noise)).float()
    return (sig / n_noise).reshape(n, n)


def run(pretrained: MLPVelocityField, beta: float, x_eval: torch.Tensor, track_sigma: bool):
    field = copy.deepcopy(pretrained)
    act = make_actflow(field, beta)
    empty = (torch.empty(0, 2), torch.empty(0, dtype=torch.bool))
    out = {
        "samples": [model_samples(field, x_eval)],
        "queries": [empty],
        "sigma": {},
        "history": {},
    }
    out["metrics"] = [generable_coverage(out["samples"][0], verifier, BOUNDS, bins=50)]

    for r in range(1, ROUNDS + 1):
        res = act.step(N_QUERIES)
        x = model_samples(field, x_eval)
        out["samples"].append(x)
        out["queries"].append((res.samples.detach(), res.labels))
        out["metrics"].append(generable_coverage(x, verifier, BOUNDS, bins=50))
        if track_sigma and r in SIGMA_ROUNDS:
            act.fit_uncertainty()  # sigma going into the next round
            out["sigma"][r] = sigma_grid(act)
            out["history"][r] = (act.queries.detach().clone(), act.labels.clone())
        print(
            f"  beta={beta:<4g} round {r:>2}  accept {res.acceptance_rate:4.2f}  "
            f"coverage {out['metrics'][-1]['coverage']:4.2f}"
        )
    return out


def _draw_domain(ax, outline_only: bool = False):
    if outline_only:
        for r in (R_IN, R_OUT):
            ax.add_patch(plt.Circle((0, 0), r, fill=False, ec="white", lw=0.8, alpha=0.8))
    else:
        ax.add_patch(Wedge((0, 0), R_OUT, 0, 360, width=R_OUT - R_IN, color=C_VALID))
    deg = math.degrees(HALF_ANGLE)
    ax.add_patch(
        Wedge((0, 0), R_OUT, -deg, deg, width=R_OUT - R_IN, fill=False, ec=C_DATA, lw=1.2, ls="--")
    )
    ax.set_xlim(BOUNDS)
    ax.set_ylim(BOUNDS)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_round(ax, samples, queries, metrics, title):
    _draw_domain(ax)
    ax.scatter(samples[:, 0], samples[:, 1], s=3, c=C_PARTICLE, alpha=0.5, edgecolors="none")
    xq, yq = queries
    if xq.shape[0]:
        ax.scatter(*xq[yq].T, s=9, c=C_ACCEPT, edgecolors="none", label="accepted query")
        ax.scatter(*xq[~yq].T, s=12, c=C_REJECT, marker="x", linewidths=0.8, label="rejected query")
    ax.set_title(f"{title}\ncoverage {metrics['coverage']:.2f}", fontsize=9)


def plot_rounds(base, act, out_path: Path) -> Path:
    fig, axes = plt.subplots(2, len(SNAP_ROUNDS), figsize=(3.0 * len(SNAP_ROUNDS), 6.6), dpi=130)
    for row, (name, res) in enumerate((("self-training", base), ("ActFlow", act))):
        for col, r in enumerate(SNAP_ROUNDS):
            label = "pre-trained" if r == 0 else f"round {r}"
            _draw_round(
                axes[row, col], res["samples"][r], res["queries"][r], res["metrics"][r], label
            )
        axes[row, 0].set_ylabel(name, fontsize=11)
    handles, labels = axes[1, -1].get_legend_handles_labels()
    handles.insert(0, plt.Line2D([], [], ls="", marker="o", ms=4, color=C_PARTICLE))
    labels.insert(0, "model samples")
    handles.append(plt.Line2D([], [], color=C_DATA, ls="--"))
    labels.append("pre-training arc")
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Generable-set expansion on an annulus (valid region in grey)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_uncertainty(act, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, len(SIGMA_ROUNDS), figsize=(3.1 * len(SIGMA_ROUNDS), 3.7), dpi=130)
    for ax, r in zip(axes, SIGMA_ROUNDS):
        ax.imshow(
            act["sigma"][r].numpy(), origin="lower", extent=(*BOUNDS, *BOUNDS), cmap="viridis"
        )
        _draw_domain(ax, outline_only=True)
        xq, _ = act["history"][r]
        if xq.shape[0]:
            ax.scatter(*xq.T, s=2, c="white", alpha=0.45, edgecolors="none")
        ax.set_title(f"after round {r}\n{xq.shape[0]} queries", fontsize=9)
    fig.suptitle(
        r"ActFlow verifier uncertainty $\sigma_t(\phi_s(x))$ (per-panel colour scale)", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_coverage(base, act, out_path: Path) -> Path:
    rounds = list(range(ROUNDS + 1))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=130)
    for ax, key in zip(axes, ("coverage", "validity")):
        ax.plot(rounds, [m[key] for m in act["metrics"]], color=C_ACCEPT, lw=2, label="ActFlow")
        ax.plot(
            rounds,
            [m[key] for m in base["metrics"]],
            color=C_BASELINE,
            lw=2,
            ls="--",
            label="self-training",
        )
        ax.set_xlabel("round")
        ax.set_xticks(range(0, ROUNDS + 1, 5))
        ax.set_title(key, fontsize=11)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_animation(base, act, out_path: Path, fps: int = 3):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0), dpi=100)
    frames = list(range(ROUNDS + 1)) + [ROUNDS] * 4  # hold the final round

    def update(i):
        r = frames[i]
        for ax, (name, res) in zip(axes, (("self-training", base), ("ActFlow", act))):
            ax.clear()
            _draw_round(ax, res["samples"][r], res["queries"][r], res["metrics"][r], name)
            ax.set_xlabel("pre-trained" if r == 0 else f"round {r}", fontsize=9)
        return ()

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps)
    try:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    except Exception as exc:
        plt.close(fig)
        print(f"[demo] skipping animation ({exc}), install pillow to enable it")
        return None
    plt.close(fig)
    return out_path


def main() -> None:
    out_dir = Path("outputs") / "active_flow_expansion_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    print("[demo] pre-training on the arc ...")
    pretrained = pretrain()
    x_eval = torch.randn(1500, 2)

    print("[demo] self-training baseline (beta = inf)")
    torch.manual_seed(1)
    base = run(pretrained, math.inf, x_eval, track_sigma=False)
    print("[demo] ActFlow (beta = 0.1)")
    torch.manual_seed(1)
    act = run(pretrained, 0.1, x_eval, track_sigma=True)

    for path in (
        plot_rounds(base, act, out_dir / "expansion_rounds.png"),
        plot_uncertainty(act, out_dir / "uncertainty_rounds.png"),
        plot_coverage(base, act, out_dir / "coverage.png"),
        write_animation(base, act, out_dir / "expansion.gif"),
    ):
        if path is not None:
            print(f"[demo] wrote {path}")


if __name__ == "__main__":
    main()
