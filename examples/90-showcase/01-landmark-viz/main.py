"""Landmark-heatmap visualisation demo.

DeltaFlow's job is *pretraining* a velocity-field / feature backbone; the
actual landmark head lives downstream (e.g. in CDPM-Align). This example is
purely about *what a landmark model's output looks like* so you can develop
plotting utilities against it before the real head is wired up.

Run::

    python examples/90-showcase/01-landmark-viz/main.py

Produces two files under ``outputs/landmark_viz/``:

    overlay.png        - synthetic "X-ray" with predicted dots + labels.
    heatmaps_grid.png  - the K per-landmark heatmap channels laid out on a grid.

The core utility ``visualise_landmarks(image, heatmaps, out_dir)`` is
reusable: pass it a real image ``(1, H, W)`` and a real network output
``(K, H, W)`` and it produces the same two plots. The demo below just
*fabricates* plausible heatmaps by drawing Gaussians directly at chosen
coordinates - the exact shape a trained heatmap-regression network learns
to output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except ImportError as exc:  # pragma: no cover - install hint only
    raise SystemExit(
        "This example needs matplotlib. Install it with:\n"
        "    pip install matplotlib"
    ) from exc


# --------------------------------------------------------------------------- #
# Reusable visualisation utility
# --------------------------------------------------------------------------- #

def visualise_landmarks(
    image: torch.Tensor,
    heatmaps: torch.Tensor,
    out_dir: Path,
    names: Optional[Sequence[str]] = None,
    grid_shape: Optional[tuple[int, int]] = None,
) -> tuple[Path, Path]:
    """Save an overlay plot and a per-landmark heatmap grid.

    Args:
        image: ``(1, H, W)`` grayscale image tensor, values in any range
            (auto-normalised for display).
        heatmaps: ``(K, H, W)`` per-landmark heatmap tensor. Values are
            expected in roughly ``[0, 1]`` but not required.
        out_dir: directory to save the two PNGs into (created if missing).
        names: optional list of ``K`` short labels; falls back to
            ``["LM0", "LM1", ...]``.
        grid_shape: optional ``(rows, cols)`` for the heatmap grid; auto if None.

    Returns:
        ``(overlay_path, grid_path)``.
    """
    assert image.dim() == 3 and image.shape[0] == 1, "image must be (1, H, W)"
    assert heatmaps.dim() == 3, "heatmaps must be (K, H, W)"
    assert heatmaps.shape[-2:] == image.shape[-2:], "spatial sizes must match"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    K = heatmaps.shape[0]
    if names is None:
        names = [f"LM{i}" for i in range(K)]

    img_np = image[0].detach().cpu().numpy()
    hm_np = heatmaps.detach().cpu().numpy()

    # Argmax decode: for each heatmap channel, find the brightest pixel.
    coords = []  # list of (x, y)
    for k in range(K):
        idx = np.argmax(hm_np[k])
        y, x = np.unravel_index(idx, hm_np[k].shape)
        coords.append((int(x), int(y)))

    overlay_path = _plot_overlay(img_np, coords, names, out_dir / "overlay.png")
    grid_path = _plot_heatmap_grid(hm_np, names, grid_shape, out_dir / "heatmaps_grid.png")
    return overlay_path, grid_path


def _plot_overlay(
    img_np: np.ndarray,
    coords: list[tuple[int, int]],
    names: Sequence[str],
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.imshow(img_np, cmap="gray", origin="upper")
    for (x, y), name in zip(coords, names):
        ax.add_patch(Circle((x, y), radius=4, edgecolor="red", facecolor="none", linewidth=1.5))
        ax.text(
            x + 6, y - 6, name,
            color="red", fontsize=7, weight="bold",
            path_effects=None,
        )
    ax.set_title(f"Predicted landmarks (K = {len(coords)})")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_heatmap_grid(
    hm_np: np.ndarray,
    names: Sequence[str],
    grid_shape: Optional[tuple[int, int]],
    out_path: Path,
) -> Path:
    K = hm_np.shape[0]
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(K)))
        rows = int(np.ceil(K / cols))
    else:
        rows, cols = grid_shape

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0), dpi=120)
    axes = np.atleast_2d(axes)

    for k in range(rows * cols):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        ax.set_axis_off()
        if k < K:
            ax.imshow(hm_np[k], cmap="magma", origin="upper", vmin=0.0, vmax=max(1e-6, hm_np[k].max()))
            ax.set_title(names[k], fontsize=7)
    fig.suptitle(f"Per-landmark heatmaps ({K} channels)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Demo: fabricate a synthetic X-ray + plausible heatmap outputs
# --------------------------------------------------------------------------- #

def _make_synthetic_xray(size: int = 256, seed: int = 0) -> torch.Tensor:
    """A grayscale image with a few soft blobs so the overlay looks vaguely
    anatomical instead of pure noise. Purely for illustration."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size), dtype=np.float32)

    # A large "head-shaped" ellipsoidal blob + a few smaller ones.
    for cx, cy, sx, sy, amp in [
        (size * 0.50, size * 0.55, size * 0.30, size * 0.42, 0.7),  # skull
        (size * 0.35, size * 0.45, size * 0.06, size * 0.06, 0.4),  # orbit L
        (size * 0.65, size * 0.45, size * 0.06, size * 0.06, 0.4),  # orbit R
        (size * 0.50, size * 0.72, size * 0.10, size * 0.05, 0.5),  # jaw
    ]:
        img += amp * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2.0)

    img += 0.03 * rng.standard_normal(img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return torch.from_numpy(img).unsqueeze(0)


def _gaussian_heatmap(size: int, cx: float, cy: float, sigma: float = 4.0) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))


def _fabricate_heatmaps(
    size: int,
    landmarks: Sequence[tuple[float, float]],
    sigma: float = 4.0,
) -> torch.Tensor:
    """Stack one Gaussian heatmap per landmark into a ``(K, H, W)`` tensor.

    This is exactly the shape a trained heatmap-regression network would
    emit; here we synthesise it directly for demo purposes.
    """
    return torch.stack(
        [_gaussian_heatmap(size, cx, cy, sigma=sigma) for (cx, cy) in landmarks],
        dim=0,
    )


def main() -> None:
    size = 256

    # 19 plausible "cephalometric-style" landmark coordinates, chosen by eye
    # against the synthetic X-ray above. Names roughly follow the ISBI 2015
    # cephalometric challenge but are only illustrative here.
    landmarks: list[tuple[float, float, str]] = [
        (128, 60,  "S"),      # sella
        (100, 80,  "N"),      # nasion
        (95,  115, "Or"),     # orbitale
        (161, 115, "Or_r"),   # orbitale (right)
        (75,  120, "Po"),     # porion
        (181, 120, "Po_r"),   # porion (right)
        (110, 155, "ANS"),    # anterior nasal spine
        (140, 155, "PNS"),    # posterior nasal spine
        (120, 175, "A"),      # A point
        (128, 195, "B"),      # B point
        (128, 220, "Pog"),    # pogonion
        (128, 235, "Gn"),     # gnathion
        (128, 245, "Me"),     # menton
        (85,  195, "Go"),     # gonion (left)
        (171, 195, "Go_r"),   # gonion (right)
        (128, 100, "Ba"),     # basion
        (105, 175, "U1"),     # upper incisor tip
        (150, 195, "L1"),     # lower incisor tip
        (128, 130, "PtV"),    # pterygoid vertical
    ]

    image = _make_synthetic_xray(size=size, seed=0)
    coords = [(x, y) for (x, y, _) in landmarks]
    names = [n for (_, _, n) in landmarks]

    # In a real pipeline this line would be:
    #     heatmaps = model(image.unsqueeze(0))[0]
    heatmaps = _fabricate_heatmaps(size, coords, sigma=4.0)

    out_dir = Path("outputs") / "landmark_viz"
    overlay_path, grid_path = visualise_landmarks(
        image=image,
        heatmaps=heatmaps,
        out_dir=out_dir,
        names=names,
        grid_shape=(4, 5),  # 19 channels -> 4x5 grid with one empty cell
    )

    print(f"[demo] wrote {overlay_path}")
    print(f"[demo] wrote {grid_path}")
    print()

    # In a real pipeline, replace `_fabricate_heatmaps(...)` with a forward pass 
    # through trained landmark head:
    # heatmaps = model(image.unsqueeze(0))[0]  # (K, H, W)")


if __name__ == "__main__":
    main()