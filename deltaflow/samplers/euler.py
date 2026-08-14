"""Backward-compatibility shim: `FlowSampler` now lives at
`deltaflow.solvers.euler` as `EulerSolver` (with the
``FlowSampler`` alias preserved).
"""

from ..solvers.euler import EulerSolver, FlowSampler

__all__ = ["EulerSolver", "FlowSampler"]
