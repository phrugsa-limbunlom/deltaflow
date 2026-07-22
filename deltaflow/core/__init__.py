"""Base abstractions shared across DeltaFlow's models, losses, and solvers.

Every user-facing component (velocity field, interpolant, solver, loss)
subclasses one of these bases, so new variants are drop-in and not
rewrites of the surrounding machinery.
"""

from .base_interpolant import BaseInterpolant
from .base_loss import BaseLoss
from .base_solver import BaseSolver
from .base_velocity_field import BaseVelocityField

__all__ = [
    "BaseInterpolant",
    "BaseLoss",
    "BaseSolver",
    "BaseVelocityField",
]
