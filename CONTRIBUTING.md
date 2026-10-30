# Contributing

Contributions are welcome. This project is small and research-oriented, so
please open an issue to discuss non-trivial changes before sending a PR.

## Setup

```bash
git clone https://github.com/phrugsa-limbunlom/deltaflow
cd deltaflow
pip install -e ".[dev]"
```

## Project layout

Each subpackage of `deltaflow/` has one job:

| Package | Responsibility |
| --- | --- |
| `core` | Abstract base classes (`BaseVelocityField`, `BaseInterpolant`, `BaseSolver`, `BaseLoss`, `BaseCoupling`, ...). No dependencies on other subpackages. |
| `utils` | Stateless helpers: numerical stability, `broadcast_time`, OT assignment, image I/O. No dependencies on other subpackages. |
| `interpolants` | Probability paths `x_t` between noise `x0` (`t=0`) and data `x1` (`t=1`) and their target velocity. |
| `models` | Velocity-field networks and wrappers (`DiT`, `WrappedBackbone`, `EMA`, projectors). |
| `losses` | Training objectives built from an interpolant and a velocity field. |
| `inverse` | Measurement operators, likelihoods, and Tweedie decompositions for inverse problems. |
| `solvers` | Samplers: `EulerSolver`, `HeunSolver`, `PosteriorSolver`, `EquilibriumSolver`. |
| `trainer` | Training loop, checkpointing, train-time couplings, image streaming. |
| `datasets` | Ready-made datasets (e.g. `RadiographDataset`). |
| `expansion` | Active Flow Expansion, which composes solvers, losses, and inverse. |

### Dependency rule

Imports only point *down* this stack, never up or sideways:

```text
expansion, trainer, datasets
        |
losses, solvers
        |
interpolants, models, inverse
        |
core, utils
```

For example, a loss may import from `core` and `interpolants` but never from
`trainer`. If two modules need the same helper, move it into `utils` (or an
abstraction into `core`) rather than importing across.

### Coding conventions

- Import third-party dependencies at the top of the module. Required
  dependencies (`torch`, `numpy`, `scipy`, `tqdm`, `einops`) are never wrapped
  in `try/except ImportError`. Optional extras (e.g. Pillow) go through one
  adapter that raises a clear install hint (see `deltaflow/utils/image_io.py`).
- Import bases from `deltaflow.core`, not from sibling feature packages.
- Time convention: `t=0` is noise and `t=1` is data, everywhere.
- Use `broadcast_time(t, x)` to reshape a per-sample `(B,)` time for
  broadcasting, and the `deltaflow.utils` numerical helpers in loss math.
- `BaseSolver.sample` runs under `torch.no_grad()`. A solver that needs
  autograd inside a step enables it locally (see `PosteriorSolver`).

### Adding a component

- **Interpolant:** subclass `core.BaseInterpolant`, implement
  `interpolate(x1, t, x0=None) -> (x_t, target_velocity)`, export it from
  `interpolants/__init__.py`, and add a test in `tests/test_interpolants.py`.
- **Solver:** subclass `core.BaseSolver` and implement
  `step(x, t, dt, **cond)`. Call `self.velocity(x, t, **cond)` to evaluate the
  model. `sample` is inherited.
- **Loss:** subclass `core.BaseLoss`. Accept an `interpolant=` argument
  rather than hard-coding a path, and forward `**cond` to the model unchanged.

Then add the new name to the matching `docs/api/*.md` page and to
`CHANGELOG.md` under *Unreleased*.

## Before opening a PR

```bash
black deltaflow tests examples
isort deltaflow tests examples
pytest
```

## Adding an example

Examples live under `examples/<tier>/<NN-name>/main.py`, where `<tier>` is
one of `00-foundations`, `10-sampling`, `20-training`, `90-showcase`. Each
example should run standalone in well under a minute on CPU.
