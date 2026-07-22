"""
DeltaFlow: a PyTorch library for flow matching, mini-batch optimal-transport
coupling, and posterior sampling for inverse problems.

Module map:

- :mod:`deltaflow.core`         -- abstract base classes every component
  subclasses (:class:`BaseVelocityField`, :class:`BaseInterpolant`,
  :class:`BaseSolver`, :class:`BaseLoss`).
- :mod:`deltaflow.interpolants` -- probability paths (linear, mini-batch OT,
  variance-preserving).
- :mod:`deltaflow.losses`       -- conditional flow matching, plus the
  optional delta-alignment loss for guidance-representation pretraining.
- :mod:`deltaflow.solvers`      -- Euler, Heun, and the
  :class:`~deltaflow.solvers.PosteriorSolver` which *wraps* a base solver
  and injects the measurement-likelihood gradient per step (FlowDPS /
  Flower style).
- :mod:`deltaflow.trainer`      -- streaming image dataset, mixed-precision
  training loop with grad accumulation and checkpoint/resume, and
  train-time coupling strategies (independent vs OT).
- :mod:`deltaflow.inverse`      -- measurement operators, Tweedie
  decomposition, and Gaussian likelihood. v1 targets pixel space with an
  optional decoder pullback for the latent-space case (see
  ``deltaflow.inverse.__init__`` docstring).
- :mod:`deltaflow.models`       -- backbone wrappers, EMA, and projector
  heads used by the delta-alignment loss.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deltaflow")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
