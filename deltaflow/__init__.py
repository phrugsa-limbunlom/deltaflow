"""
DeltaFlow: a PyTorch library for flow matching, mini-batch optimal-transport
coupling, and posterior sampling for inverse problems.

Module map:

- `deltaflow.core`, abstract base classes every component
  subclasses (`BaseVelocityField`, `BaseInterpolant`,
  `BaseSolver`, `BaseLoss`).
- `deltaflow.interpolants`, probability paths (linear, mini-batch OT,
  variance-preserving, Schrödinger bridge, and the Equilibrium Matching
  energy-compatible target).
- `deltaflow.losses`, conditional flow matching, plus the
  optional delta-alignment loss for guidance-representation pretraining.
- `deltaflow.solvers`, Euler, Heun, the `EquilibriumSolver` which samples an
  Equilibrium Matching field by gradient descent on its implicit energy
  landscape, and the
  `PosteriorSolver` which *wraps* a base solver
  and injects the measurement-likelihood gradient per step (FlowDPS /
  Flower style).
- `deltaflow.trainer`, streaming image dataset, mixed-precision
  training loop with grad accumulation and checkpoint/resume, and
  train-time coupling strategies (independent vs OT).
- `deltaflow.inverse`, measurement operators, Tweedie
  decomposition, and Gaussian likelihood. v1 targets pixel space with an
  optional decoder pullback for the latent-space case (see
  ``deltaflow.inverse.__init__`` docstring).
- `deltaflow.models`, backbone wrappers, EMA, and projector
  heads used by the delta-alignment loss.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # The distribution name on PyPI is ``torchdeltaflow`` (the plain
    # ``deltaflow`` name is claimed by an unrelated 2020 package). The
    # *import* name is still ``deltaflow``, which is why we can't just
    # call ``version(__name__)``.
    __version__ = version("torchdeltaflow")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
