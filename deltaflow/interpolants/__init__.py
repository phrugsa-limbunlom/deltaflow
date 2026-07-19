"""Probability paths connecting noise x0 to data x1."""

from .base import BaseInterpolant
from .linear import LinearInterpolant

__all__ = ["BaseInterpolant", "LinearInterpolant"]
