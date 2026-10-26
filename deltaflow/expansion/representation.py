"""Noised flow representation, the feature map ``phi_s`` used by ActFlow.

ActFlow measures verifier uncertainty in a *learned* space rather than in raw
design space. The space is ``Z_s``, the hidden activations of the velocity
network evaluated on a design ``x`` after it has been pushed back along the
probability path to an intermediate noise level ``s``. In DeltaFlow's time
convention (``t = 0`` noise, ``t = 1`` data) this is

    x_s = interpolant(x1 = x, t = s, x0 = noise)
    phi_s(x) = pool( h(x_s, s) )

where ``h`` is the output of a chosen sub-module of the velocity field. The
paper's ablation reports that ``s`` close to the data end (but not at it)
works best, so the default is ``s = 0.9``.

The map is differentiable in ``x``, which is what lets the uncertainty tilt of
Eq. (9) be applied as a sampling-time gradient.
"""

from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from ..core.base_interpolant import BaseInterpolant
from ..interpolants.linear import LinearInterpolant
from ..utils.numerical import safe_normalize


class FlowRepresentation:
    r"""Feature map \(\phi_s : \mathcal{X} \to \mathcal{Z}_s\) read from a velocity field.

    A design \(x\) is first noised to level \(s\) on the training path,

    \[
    x_s = (1 - s)\,x_0 + s\,x, \qquad x_0 \sim \mathcal{N}(0, I)
    \]

    (for the default linear path), then passed through the velocity field
    \(v_\theta(x_s, s)\). The activation of ``layer`` is captured by a
    temporary forward hook and pooled to one vector per sample. With
    ``layer=None`` the velocity output itself is used as the representation.

    Pooling rule for the captured activation \(h\), shaped ``[B, ...]``.

    - ``[B, D]`` is returned unchanged.
    - ``[B, N, D]`` (token sequences, e.g. a DiT block) is averaged over the
      ``N`` tokens.
    - ``[B, C, *spatial]`` (feature maps) is averaged over the spatial axes.

    Use ``pool="flatten"`` to keep every coordinate instead.

    Because \(x_0\) is random, \(\phi_s\) is stochastic for \(s < 1\). Pass
    ``noise`` explicitly for a deterministic evaluation.

    References:
        De Santi et al., "Active Flow Expansion for Out-of-Distribution
        Discovery: from Theory to Molecules" (2026),
        https://arxiv.org/abs/2606.08802.

    Args:
        model: velocity field ``model(x, t, **cond)``.
        s: noise level \(s \in (0, 1]\) at which features are read.
        layer: sub-module (or its dotted name, resolved with
            ``model.get_submodule``) whose output is the representation.
            ``None`` uses the model output.
        interpolant: path used to noise \(x\) to level \(s\). Defaults to
            `LinearInterpolant`, matching the usual training path.
        pool: ``"mean"`` (default) or ``"flatten"``.
        normalize: L2-normalise each feature vector. This keeps the linear
            kernel bounded, \(k(x, x) \le 1\), as the confidence-sequence
            theory assumes.
        time_scale: multiplies \(s\) before it reaches the model, matching
            `BaseSolver.time_scale`.
    """

    def __init__(
        self,
        model: Callable,
        s: float = 0.9,
        layer: Optional[Union[str, nn.Module]] = None,
        interpolant: Optional[BaseInterpolant] = None,
        pool: str = "mean",
        normalize: bool = True,
        time_scale: float = 1.0,
    ):
        if not 0.0 < s <= 1.0:
            raise ValueError(f"s must lie in (0, 1], got {s}")
        if pool not in ("mean", "flatten"):
            raise ValueError(f"pool must be 'mean' or 'flatten', got {pool!r}")
        self.model = model
        self.s = float(s)
        self.layer = layer
        self.interpolant = interpolant or LinearInterpolant()
        self.pool = pool
        self.normalize = normalize
        self.time_scale = time_scale

    def _resolve_layer(self) -> Optional[nn.Module]:
        if self.layer is None or isinstance(self.layer, nn.Module):
            return self.layer
        if not isinstance(self.model, nn.Module):
            raise TypeError("a layer name requires the model to be an nn.Module")
        return self.model.get_submodule(self.layer)

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            return h.unsqueeze(-1)
        if self.pool == "flatten" or h.dim() == 2:
            return h.flatten(1)
        if h.dim() == 3:
            return h.mean(dim=1)
        return h.flatten(2).mean(dim=-1)

    def __call__(
        self, x: torch.Tensor, noise: Optional[torch.Tensor] = None, **cond
    ) -> torch.Tensor:
        """Return ``phi_s(x)`` with shape ``[B, D]`` (differentiable in ``x``)."""
        t = torch.full((x.shape[0],), self.s, device=x.device, dtype=x.dtype)
        x_s, _ = self.interpolant.interpolate(x, t, x0=noise)

        module = self._resolve_layer()
        captured = {}
        handle = None
        if module is not None:

            def _hook(_mod, _inp, out):
                captured["h"] = out[0] if isinstance(out, (tuple, list)) else out

            handle = module.register_forward_hook(_hook)
        try:
            out = self.model(x_s, t * self.time_scale, **cond)
        finally:
            if handle is not None:
                handle.remove()

        h = captured["h"] if module is not None else out
        z = self._pool(h)
        if self.normalize:
            z = safe_normalize(z, dim=-1)
        return z


__all__ = ["FlowRepresentation"]
