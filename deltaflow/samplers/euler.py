"""Backward-compatibility shim: :class:`FlowSampler` now lives at
:mod:`deltaflow.solvers.euler` as :class:`EulerSolver` (with the
``FlowSampler`` alias preserved).
"""

from ..solvers.euler import EulerSolver, FlowSampler

__all__ = ["EulerSolver", "FlowSampler"]
