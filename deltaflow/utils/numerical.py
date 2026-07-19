"""
Numerical stability utilities for mixed-precision training.

General-purpose helpers for preventing NaN/Inf values in deep learning
computations, particularly under FP16/BF16 autocast.
"""

from typing import Optional

import torch


class StabilityConfig:
    """Default numerical stability bounds."""

    EPS_FP32 = 1e-8
    EPS_FP16 = 1e-6
    NOISE_BOUND = 5.0
    PREDICTION_BOUND = 10.0
    LOSS_BOUND = 50.0
    COSINE_SIM_BOUND = 1.0


def get_eps(dtype: torch.dtype) -> float:
    """Return an appropriate epsilon for the given dtype."""
    if dtype in (torch.float16, torch.bfloat16):
        return StabilityConfig.EPS_FP16
    return StabilityConfig.EPS_FP32


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: Optional[float] = None) -> torch.Tensor:
    """L2-normalize with numerical stability (avoids division by ~zero norms)."""
    if eps is None:
        eps = get_eps(x.dtype)
    norm = x.norm(p=2, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return x / norm


def safe_sqrt(x: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    """Compute a square root, clamping the input to be non-negative first."""
    min_val = eps if eps is not None else 0.0
    return torch.sqrt(torch.clamp(x, min=min_val))


def clamp_noise(noise: torch.Tensor, bound: Optional[float] = None) -> torch.Tensor:
    """Clamp a noise tensor to prevent extreme values under FP16."""
    bound = bound if bound is not None else StabilityConfig.NOISE_BOUND
    return torch.clamp(noise, -bound, bound)


def clamp_prediction(pred: torch.Tensor, bound: Optional[float] = None) -> torch.Tensor:
    """Clamp model predictions to prevent extreme values."""
    bound = bound if bound is not None else StabilityConfig.PREDICTION_BOUND
    return torch.clamp(pred, -bound, bound)


def clamp_loss(loss: torch.Tensor, max_val: Optional[float] = None) -> torch.Tensor:
    """Clamp a scalar loss to prevent gradient explosion."""
    max_val = max_val if max_val is not None else StabilityConfig.LOSS_BOUND
    return torch.clamp(loss, min=0.0, max=max_val)


def clamp_cosine_similarity(sim: torch.Tensor) -> torch.Tensor:
    """Clamp cosine similarity to the valid range [-1, 1]."""
    return torch.clamp(sim, -StabilityConfig.COSINE_SIM_BOUND, StabilityConfig.COSINE_SIM_BOUND)
