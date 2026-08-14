"""Backward-compatibility shim. Prefer importing from the split modules
(`deltaflow.core.base_velocity_field`, etc.) or from `deltaflow.core`
directly.
"""

from .base_velocity_field import BaseVelocityField

__all__ = ["BaseVelocityField"]
