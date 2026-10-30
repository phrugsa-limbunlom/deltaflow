# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `deltaflow.losses.TimeUnconditionedFlowMatchingLoss` and
  `deltaflow.models.TimeBlindVelocityField`: time-blind flow matching
  (arXiv:2605.08344), where the model never receives the interpolation time.
  New **Time-Unconditioned Flow Matching** guide explains why time is
  recoverable from `x_t` in high dimensions, with the
  `examples/90-showcase/12-time-unconditioned-fm-viz` figures.

- `deltaflow.expansion`: Active Flow Expansion (ActFlow, De Santi et al.,
  2026, arXiv:2606.08802), continued pre-training that grows a flow's generable
  set toward the valid design space of a black-box verifier. Ships
  `FlowRepresentation` (hidden features `phi_s` of the velocity field at noise
  level `s`), `KernelUncertainty` (the closed-form Eq. 10 posterior standard
  deviation, linear or RBF kernel, differentiable), `UncertaintyGuidance` (the
  Eq. 9 tilt `p * exp(sigma / beta)` as a `Likelihood`, sampled by the existing
  `PosteriorSolver`), the `ActFlow` round loop (Algorithm 1), and
  `generable_coverage`. See the new
  `examples/20-training/04-active-flow-expansion` walkthrough and the
  **Active Flow Expansion** guide.
- `deltaflow.losses.SignedReplayFlowMatchingLoss`: accept-attract,
  reject-repel flow matching `L+ - alpha_t * L-`, with a fixed weight or the
  paper's gradient-norm calibration of `alpha_t`.
- `examples/90-showcase/11-active-flow-expansion-viz`: round-by-round
  self-training vs ActFlow on the annulus toy, with labelled queries,
  uncertainty maps, coverage curves, and an animated GIF.

- `deltaflow.models.DiT`: a class-conditional Diffusion Transformer velocity
  field with **adaLN-Zero** conditioning (Peebles & Xie, 2023). Ships a reusable
  `TimestepEmbedding`, a `LabelEmbedding` with a learned null token for
  classifier-free guidance, zero-initialized modulation gates (identity at init,
  zero initial velocity), `cond["y"]` support through the existing
  `BaseVelocityField` API, and a `forward_with_cfg` guided-sampling helper. See
  the new `examples/20-training/03-conditional-dit` walkthrough.
- `examples/90-showcase/10-dit-tensor-viz` and the **DiT Walkthrough** docs page
  (Concepts), an interactive image-to-velocity tour with named layer navigation,
  patch-linked tensors, attention and MLP intermediates in every transformer
  block, and final projection and unpatchify views. Tensors come from a trained
  three-channel RGB model and are checked against `DiT.forward`. Includes
  individual-channel views, CFG, sampling, and static reference figures.
  The interactive tour is also embedded on Home.

### Changed

- `scipy` is now a core dependency (previously gated behind the `ot`
  extra), so `OTInterpolant`/`OTCoupling` use the exact Hungarian-algorithm
  mini-batch OT assignment by default. The `ot` extra is kept as a
  backwards-compatible no-op. The greedy nearest-neighbour fallback has been
  removed, since `scipy` is always installed.
- Internal cleanup: hard dependencies (`tqdm`, `scipy`, `numpy`) are imported
  at module level instead of inside guarded `try/except ImportError` blocks.
  Optional Pillow handling lives in a single adapter,
  `deltaflow.utils.image_io` (`require_pillow`, `open_image`, `list_images`,
  `IMAGE_EXTENSIONS`), shared by `ImageFolderStream` and `RadiographDataset`
  (which now also picks up `.webp` files). Losses depend on
  `deltaflow.core.BaseCoupling` rather than `deltaflow.trainer`, and the
  `l1`/`l2`/`huber` reduction is shared instead of duplicated.

### Removed (breaking)

- Legacy aliases and shim modules. Use the canonical names instead:
  - `FlowSampler` -> `EulerSolver`
  - `FlowMatchingLoss` -> `ConditionalFlowMatchingLoss`
  - `GradientDescentSolver` -> `EquilibriumSolver`
  - `deltaflow.samplers` -> `deltaflow.solvers`
  - `deltaflow.core.base` and `deltaflow.interpolants.base` -> `deltaflow.core`
  - `deltaflow.losses.flow_matching` -> `deltaflow.losses`
  - `deltaflow.interpolants.ot._batch_ot_permutation` ->
    `deltaflow.utils.batch_ot_permutation`
- `deltaflow.solvers.gradient_descent` renamed to
  `deltaflow.solvers.equilibrium` (matches `EquilibriumSolver`).
- Unused `deltaflow.inverse.tweedie.TweedieEstimates`.

### Changed (internal / readability)

- `BaseSolver._eval_velocity` is now the public `BaseSolver.velocity(x, t)`.
- `BaseSolver.sample` always runs under `torch.no_grad()`. `PosteriorSolver`
  enables autograd locally for its likelihood gradient. Redundant `sample`
  overrides were removed from `HeunSolver` and `PosteriorSolver`.
- New `deltaflow.utils.broadcast_time` replaces the repeated
  `t.view(-1, *[1] * ...)` reshapes in interpolants, Tweedie, and solvers.
- `trainer.train` routes through a pass-through `GradScaler` instead of
  repeating `float16` checks, and saves checkpoints through one helper.
- `ActFlow` uses a `_model_mode` context manager for train/eval switching.
- `delta_alignment_loss` reuses `MultiScaleProjector.project_delta_h`.
- `CONTRIBUTING.md` documents the package layout, dependency rule, and how to
  add an interpolant, solver, or loss.

## [0.2.2]

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
