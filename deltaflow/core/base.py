"""Backward-compatibility shim. Prefer importing from the split modules
(:mod:`deltaflow.core.base_velocity_field`, etc.) or from :mod:`deltaflow.core`
directly.
"""

from .base_velocity_field import BaseVelocityField

__all__ = ["BaseVelocityField"]
