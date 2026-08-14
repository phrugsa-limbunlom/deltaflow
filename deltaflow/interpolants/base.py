"""Backward-compatibility shim. Prefer importing `BaseInterpolant`
from `deltaflow.core`.
"""

from ..core.base_interpolant import BaseInterpolant

__all__ = ["BaseInterpolant"]
