"""
DeltaFlow: flow matching and anatomy-invariant guidance alignment for
radiograph generative pretraining.

DeltaFlow provides composable primitives for:
    - Conditional flow matching (probability paths, ODE samplers)
    - Delta alignment: a multi-scale, anatomy-cancelling loss that isolates
      the guidance signal Δh = h_cond - h_uncond and aligns it across
      augmented views, independent of anatomy-specific content.

The name reflects both pieces: "Delta" is the guidance-difference feature
(Δh) at the core of the alignment mechanism, "Flow" is the flow-matching
generative engine that consumes it.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deltaflow")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
