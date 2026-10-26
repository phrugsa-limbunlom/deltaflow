# API Reference

Auto-generated documentation for DeltaFlow's public API. Every page renders the
docstrings, signatures, and source of the classes and functions in that
subpackage.

| Module | Contents |
|---|---|
| [core](core.md) | `BaseVelocityField`, `BaseInterpolant`, `BaseSolver`, `BaseLoss` |
| [interpolants](interpolants.md) | `LinearInterpolant`, `OTInterpolant`, `VariancePreservingInterpolant` |
| [losses](losses.md) | `FlowMatchingLoss`, `ConditionalFlowMatchingLoss`, `DeltaAlignmentLoss`, `SignedReplayFlowMatchingLoss` |
| [solvers](solvers.md) | `EulerSolver`, `HeunSolver`, `PosteriorSolver` |
| [samplers](samplers.md) | `FlowSampler` and re-exported solvers |
| [models](models.md) | `TinyVelocityField`, `WrappedBackbone`, `EMA`, `MultiScaleProjector`, `ProjectorHead` |
| [inverse](inverse.md) | Operators, Tweedie decompositions, `GaussianLikelihood` |
| [expansion](expansion.md) | `ActFlow`, `FlowRepresentation`, `KernelUncertainty`, `UncertaintyGuidance`, `generable_coverage` |
| [trainer](trainer.md) | `TrainConfig`, `train`, coupling strategies, checkpointing |
| [datasets](datasets.md) | Radiograph dataset wrappers |
| [utils](utils.md) | Numerical stability helpers |
