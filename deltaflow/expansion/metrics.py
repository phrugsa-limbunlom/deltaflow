"""Generable-set coverage, the evaluation metric for set expansion.

ActFlow is judged by how much of the valid design space the model can
*reach*, not by how well it matches the training density. For 2D problems
the paper estimates the generable set on a histogram, a bin is generable when
its estimated density is at least ``tau``, and reports which fraction of the
valid bins is generable.
"""

from typing import Callable, Dict, Sequence, Tuple, Union

import torch

Bounds = Union[Tuple[float, float], Sequence[Tuple[float, float]]]


def _axis_bounds(bounds: Bounds) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if len(bounds) == 2 and not isinstance(bounds[0], (tuple, list)):
        lo, hi = float(bounds[0]), float(bounds[1])
        return (lo, hi), (lo, hi)
    (x_lo, x_hi), (y_lo, y_hi) = bounds
    return (float(x_lo), float(x_hi)), (float(y_lo), float(y_hi))


@torch.no_grad()
def generable_coverage(
    samples: torch.Tensor,
    valid_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: Bounds = (-3.5, 3.5),
    bins: int = 100,
    tau: float = 0.01,
) -> Dict[str, float]:
    r"""Histogram estimate of generable-set coverage for 2D samples.

    Samples are binned on a ``bins x bins`` grid over ``bounds``. A bin
    \(b\) is *generable* when its density estimate
    \(\hat{p}(b) = n_b / (N\,|b|)\) is at least \(\tau\), giving the
    plug-in generable set \(\hat{\Omega}^\tau\). A bin is *valid* when
    ``valid_fn`` accepts its centre, giving \(\hat{\Omega}^*\). Then

    \[
    \text{coverage} = \frac{|\hat{\Omega}^\tau \cap \hat{\Omega}^*|}{|\hat{\Omega}^*|},
    \qquad
    \text{validity} = \frac{|\hat{\Omega}^\tau \cap \hat{\Omega}^*|}{|\hat{\Omega}^\tau|}.
    \]

    Args:
        samples: ``[N, 2]`` model samples.
        valid_fn: verifier on ``[M, 2]`` points returning bools or ``{0, 1}``.
        bounds: ``(lo, hi)`` for both axes, or ``((x_lo, x_hi), (y_lo, y_hi))``.
        bins: bins per axis.
        tau: density threshold \(\tau\).

    Returns:
        A dict with ``coverage``, ``validity``, ``n_generable_bins`` and
        ``n_valid_bins``.
    """
    if samples.dim() != 2 or samples.shape[1] != 2:
        raise ValueError(f"expected samples of shape [N, 2], got {tuple(samples.shape)}")
    (x_lo, x_hi), (y_lo, y_hi) = _axis_bounds(bounds)
    pts = samples.detach().float().cpu()
    n = pts.shape[0]
    wx, wy = (x_hi - x_lo) / bins, (y_hi - y_lo) / bins

    ix = torch.floor((pts[:, 0] - x_lo) / wx).long()
    iy = torch.floor((pts[:, 1] - y_lo) / wy).long()
    inside = (ix >= 0) & (ix < bins) & (iy >= 0) & (iy < bins)
    counts = torch.bincount(ix[inside] * bins + iy[inside], minlength=bins * bins).float()
    density = counts / max(n, 1) / (wx * wy)
    generable = density >= tau

    cx = x_lo + (torch.arange(bins, dtype=torch.float32) + 0.5) * wx
    cy = y_lo + (torch.arange(bins, dtype=torch.float32) + 0.5) * wy
    grid = torch.stack(torch.meshgrid(cx, cy, indexing="ij"), dim=-1).reshape(-1, 2)
    valid = torch.as_tensor(valid_fn(grid)).flatten().cpu()
    valid = valid if valid.dtype == torch.bool else valid.float() > 0.5

    hit = (generable & valid).sum().item()
    n_valid = int(valid.sum())
    n_gen = int(generable.sum())
    return {
        "coverage": hit / n_valid if n_valid else 0.0,
        "validity": hit / n_gen if n_gen else 0.0,
        "n_generable_bins": n_gen,
        "n_valid_bins": n_valid,
    }


__all__ = ["generable_coverage"]
