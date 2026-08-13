# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- API-reference docstrings now include the underlying mathematics in
  LaTeX (rendered via MathJax/arithmatex): probability paths and target
  velocities for the interpolants (`Linear`, `VariancePreserving`, `OT`,
  `SchrodingerBridge`), the Euler/Heun update rules, the DPS posterior-solver
  step, the conditional-flow-matching and delta-alignment objectives, the
  `LinearTweedie`/`VPTweedie` inversions, and the Gaussian likelihood.

- `deltaflow.interpolants.SchrodingerBridgeInterpolant`: an entropic
  Schrödinger-bridge probability path (a Brownian bridge with tunable
  diffusivity `sigma` around the straight-line mean). `sigma=0` recovers
  `LinearInterpolant` exactly. Intended to be paired with
  `deltaflow.trainer.OTCoupling` for a simulation-free approximation of the
  dynamic Schrödinger bridge.
- Showcase example `05-schrodinger-bridge-viz`: visualises the bridge path
  at increasing `sigma` (`bridge_paths.png` static, `bridge_paths.gif`
  animated), then trains a velocity field against it (with `OTCoupling`)
  and inspects the resulting sampler on a two-moons target.
- Showcase example `06-algorithm-comparison`: trains the same model on the
  same target under all four interpolant/coupling configurations (linear
  independent, OT coupling, variance-preserving, Schrödinger bridge) and
  compares final samples, sampled trajectories, and an animated side-by-side
  sampling GIF. A README section ("Comparing the paths") links these
  figures, alongside the Schrödinger-bridge sampling animation.

## [0.2.1]

### Fixed

- README images used relative paths (`docs/assets/...`), which don't resolve
  on PyPI's standalone README renderer. Switched to absolute
  `raw.githubusercontent.com` URLs so images load correctly on PyPI.

## [0.2.0]

### Added

- `deltaflow.interpolants.OTInterpolant`: the linear path applied after a
  mini-batch optimal-transport re-ordering of `x0`.
- `deltaflow.trainer` couplings: `IndependentCoupling` and `OTCoupling`
  (Hungarian assignment, greedy fallback) plus `ConditionalFlowMatchingLoss`.
- `deltaflow.solvers`: `EulerSolver`, `HeunSolver` (second-order integration),
  and `PosteriorSolver`, which wraps a base solver with a per-step
  measurement-likelihood gradient.
- `deltaflow.inverse`: measurement operators (`MaskOperator`, `BlurOperator`,
  `DownsampleOperator`, `IdentityOperator`), Tweedie estimators
  (`LinearTweedie`, `VPTweedie`), and `GaussianLikelihood` with an optional
  decoder pullback for the latent-space case.
- `deltaflow.interpolants.VariancePreservingInterpolant` (trigonometric
  diffusion path).
- Showcase examples: `01-landmark-viz`, `02-sampling-flow-viz`,
  `03-minibatch-ot-viz`, `04-inverse-posterior-viz`, reproducing the README
  figures and animations.
- MkDocs Material documentation site (`docs/`), published via
  `.github/workflows/docs.yml`.
- Animated logo, favicon, and algorithm figures under `docs/assets/`.

### Changed

- Rewrote `README.md` in a component-table style with paper references,
  visuals, and per-method usage snippets (flow matching, OT couplings,
  variance-preserving path, delta alignment, posterior sampling).
- Fixed paper attributions and introduced a `Likelihood` protocol in
  `deltaflow.inverse`.

## [0.1.0]

### Added

- Initial scaffold: `core`, `interpolants` (`LinearInterpolant`), `samplers`
  (`FlowSampler`), `losses` (`FlowMatchingLoss`, `DeltaAlignmentLoss`),
  `models` (`MultiScaleProjector`, `EMA`), `datasets` (generic radiograph
  wrappers).
- Tiered examples (`00-foundations`, `10-sampling`, `20-training`,
  `90-showcase`).
- Unit tests for interpolants, samplers, losses, and models.
