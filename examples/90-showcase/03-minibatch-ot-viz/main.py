"""90-showcase/03-minibatch-ot-viz: visualise why mini-batch optimal-transport
coupling produces straighter flows than independent coupling.

Both panels pair the *same* source cloud (x0) with the *same* target cloud
(x1). The left panel pairs them independently (index i to index i); the right
panel permutes x0 by the mini-batch OT assignment so the total squared-L2
transport cost is minimised. OT coupling untangles the crossings, so the
displacements Delta = x1 - x0 (and the flow that integrates them) are shorter
and straighter, which is what lets OT-trained models sample in fewer steps.

Run: python examples/90-showcase/03-minibatch-ot-viz/main.py
"""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch

from deltaflow.interpolants.ot import _batch_ot_permutation

# Non-primary palette (theme primary is blue; these deliberately avoid it).
C_SOURCE = "#0f9bab"     # source samples: teal
C_TARGET = "#7a5cc0"     # target samples: purple
C_OT = "#3f9e73"         # OT coupling lines: green
C_INDEP = "#b0563f"      # independent coupling lines: muted rust (for contrast)


def transport_cost(x0: torch.Tensor, x1: torch.Tensor) -> float:
    return (x0 - x1).pow(2).sum(dim=-1).mean().item()


def draw(ax, x0, x1, line_color, title):
    x0n, x1n = x0.numpy(), x1.numpy()
    for a, b in zip(x0n, x1n):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=line_color,
                alpha=0.45, linewidth=1.0, zorder=1)
    ax.scatter(x0n[:, 0], x0n[:, 1], s=32, c=C_SOURCE, edgecolors="none",
               zorder=3, label="source $x_0$")
    ax.scatter(x1n[:, 0], x1n[:, 1], s=32, c=C_TARGET, edgecolors="none",
               zorder=3, label="target $x_1$")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def main():
    torch.manual_seed(3)
    n = 24

    # Source and target clouds that overlap in space, so an arbitrary (index i
    # to index i) pairing produces long, crossing displacements while the OT
    # assignment untangles them into short ones.
    x0 = torch.randn(n, 2) * 1.3 + torch.tensor([-1.1, 0.0])
    x1 = torch.randn(n, 2) * 1.3 + torch.tensor([1.1, 0.0])

    # Independent coupling pairs index i to index i (as drawn from the loaders).
    cost_indep = transport_cost(x0, x1)

    # OT coupling permutes x0 so total squared-L2 cost is minimised.
    perm = _batch_ot_permutation(x0, x1)
    x0_ot = x0[perm]
    cost_ot = transport_cost(x0_ot, x1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    draw(axes[0], x0, x1, C_INDEP,
         f"Independent coupling\nmean transport cost = {cost_indep:.2f}")
    draw(axes[1], x0_ot, x1, C_OT,
         f"Mini-batch OT coupling\nmean transport cost = {cost_ot:.2f}")
    fig.suptitle(
        "Mini-batch optimal transport straightens the source-to-target pairing",
        fontsize=13,
    )
    fig.tight_layout()

    out_dir = Path("outputs") / "minibatch_ot_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "minibatch_ot.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[demo] wrote {out_path}")
    print(f"[demo] independent cost={cost_indep:.3f}  ot cost={cost_ot:.3f}")

    gif_path = write_animation(x0, x1, x0_ot, out_dir / "minibatch_ot.gif")
    if gif_path is not None:
        print(f"[demo] wrote {gif_path}")


def write_animation(x0, x1, x0_ot, out_path, frames=60, fps=20):
    """Animate both couplings transporting the source cloud onto the target."""
    x0n, x1n, x0otn = x0.numpy(), x1.numpy(), x0_ot.numpy()

    lo = min(x0n.min(), x1n.min()) - 0.6
    hi = max(x0n.max(), x1n.max()) + 0.6

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, title, col in (
        (axes[0], "Independent coupling", C_INDEP),
        (axes[1], "Mini-batch OT coupling", C_OT),
    ):
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.scatter(x1n[:, 0], x1n[:, 1], s=32, c=C_TARGET, edgecolors="none",
                   zorder=2, label="target $x_1$")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    moving_indep = axes[0].scatter(x0n[:, 0], x0n[:, 1], s=32, c=C_SOURCE,
                                   edgecolors="none", zorder=3)
    moving_ot = axes[1].scatter(x0otn[:, 0], x0otn[:, 1], s=32, c=C_SOURCE,
                                edgecolors="none", zorder=3)
    fig.suptitle("Transporting the source cloud onto the target", fontsize=13)
    fig.tight_layout()

    def update(k):
        s = k / (frames - 1)
        pos_indep = (1 - s) * x0n + s * x1n
        pos_ot = (1 - s) * x0otn + s * x1n
        moving_indep.set_offsets(pos_indep)
        moving_ot.set_offsets(pos_ot)
        return moving_indep, moving_ot

    anim = animation.FuncAnimation(fig, update, frames=frames,
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
