"""Backbone wrapper: turn any ``nn.Module`` into a DeltaFlow velocity field.

The design constraint from the build prompt is to *wrap* an existing image
backbone (UNet, DiT, ...) rather than ship one from scratch. This module
provides a thin adapter that

- adopts the `BaseVelocityField` interface,
- normalises the time signature (float / 0-d tensor / (B,) tensor) and
  applies an optional ``time_scale``,
- forwards extra keyword conditioning unchanged.

Example - wrap a HuggingFace ``diffusers`` UNet::

    from diffusers import UNet2DModel
    from deltaflow.models import WrappedBackbone

    unet = UNet2DModel(...)
    v_theta = WrappedBackbone(
        unet,
        forward_fn=lambda net, x, t, **c: net(x, t, return_dict=False)[0],
        time_scale=1000.0,   # UNet2DModel expects diffusion timesteps
    )

For quick tests without pulling in a full UNet dependency, use
`TinyVelocityField` at the bottom of this module.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn

from ..core.base_velocity_field import BaseVelocityField


class WrappedBackbone(BaseVelocityField):
    """Adapter that turns any ``nn.Module`` backbone into a velocity field.

    Args:
        backbone: the underlying network (UNet, DiT, ...).
        forward_fn: callable ``(backbone, x, t, **cond) -> velocity``. If
            ``None``, the default is ``backbone(x, t, **cond)``.
        time_scale: multiplier applied to ``t`` before it enters the
            backbone (useful when the backbone was trained on
            diffusion-style integer timesteps).
    """

    def __init__(
        self,
        backbone: nn.Module,
        forward_fn: Optional[Callable] = None,
        time_scale: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.forward_fn = forward_fn
        self.time_scale = time_scale

    def _prepare_time(self, x: torch.Tensor, t) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return t.to(device=x.device, dtype=x.dtype) * self.time_scale

    def forward(self, x: torch.Tensor, t, **cond) -> torch.Tensor:
        t_ = self._prepare_time(x, t)
        if self.forward_fn is not None:
            return self.forward_fn(self.backbone, x, t_, **cond)
        return self.backbone(x, t_, **cond)


class TinyVelocityField(BaseVelocityField):
    """A tiny convolutional velocity field for tests and toy runs.

    Not a real backbone - just enough capacity to keep unit tests
    self-contained. For real training use `WrappedBackbone` around a
    proper UNet / DiT implementation.
    """

    def __init__(self, channels: int = 1, hidden: int = 32):
        super().__init__()
        self.channels = channels
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.net = nn.Sequential(
            nn.Conv2d(channels + hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **cond) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        t_emb = self.time_embed(t.view(-1, 1).to(x.dtype))  # (B, hidden)
        t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1).expand(
            -1, -1, x.shape[2], x.shape[3]
        )
        h = torch.cat([x, t_emb], dim=1)
        return self.net(h)


__all__ = ["TinyVelocityField", "WrappedBackbone"]
