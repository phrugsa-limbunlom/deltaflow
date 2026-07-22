"""Backward-compatibility shim. Prefer importing :class:`BaseInterpolant`
from :mod:`deltaflow.core`.
"""

from ..core.base_interpolant import BaseInterpolant

__all__ = ["BaseInterpolant"]
