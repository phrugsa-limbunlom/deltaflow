"""Linear measurement operators ``A: x -> y`` for inverse problems.

All operators are plain `torch.nn.Module` (or callable) objects with
differentiable ``forward``. They can be composed with a VAE decoder for
the latent-space case - see `deltaflow.inverse.likelihood`.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityOperator(nn.Module):
    """``A(x) = x``. Useful as a no-op default and for denoising tasks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MaskOperator(nn.Module):
    """Elementwise masking (inpainting).

    Args:
        mask: broadcastable to ``x``. Values of 1 keep, values of 0 drop.
            Passed either as a `torch.Tensor` or, at call time, via
            the ``mask`` keyword argument to `forward`.
    """

    def __init__(self, mask: Optional[torch.Tensor] = None):
        super().__init__()
        if mask is not None:
            self.register_buffer("mask", mask.to(torch.float32))
        else:
            self.mask = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        m = mask if mask is not None else self.mask
        if m is None:
            raise ValueError("MaskOperator requires a mask, either at construction or per call")
        return x * m.to(dtype=x.dtype, device=x.device)


class DownsampleOperator(nn.Module):
    """Average-pool downsampling by an integer factor."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=self.factor, stride=self.factor)


class BlurOperator(nn.Module):
    """Gaussian blur with a fixed kernel.

    Args:
        kernel_size: odd integer, side length of the Gaussian kernel.
        sigma: standard deviation of the Gaussian.
        channels: number of channels the operator will see. Required so the
            depthwise convolution weight is registered up front.
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, channels: int = 1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.register_buffer("kernel", self._make_kernel(kernel_size, sigma, channels))

    @staticmethod
    def _make_kernel(k: int, sigma: float, c: int) -> torch.Tensor:
        ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss = gauss / gauss.sum()
        kernel_2d = gauss[:, None] * gauss[None, :]
        return kernel_2d.view(1, 1, k, k).expand(c, 1, k, k).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        return F.conv2d(
            x,
            self.kernel.to(dtype=x.dtype, device=x.device),
            padding=pad,
            groups=self.channels,
        )


__all__ = ["BlurOperator", "DownsampleOperator", "IdentityOperator", "MaskOperator"]
