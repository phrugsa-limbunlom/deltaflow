"""Backward-compatibility shim. The loss lives in
:mod:`deltaflow.losses.conditional_flow_matching` now.
"""

from .conditional_flow_matching import ConditionalFlowMatchingLoss, FlowMatchingLoss

__all__ = ["ConditionalFlowMatchingLoss", "FlowMatchingLoss"]
