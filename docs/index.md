# DeltaFlow

Flow matching and anatomy-invariant guidance alignment for radiograph
generative pretraining.

- **Delta** — the guidance-difference feature `Δh = h_cond - h_uncond` that
  the alignment loss operates on, chosen to cancel out anatomy-specific
  content and isolate the conditioning signal.
- **Flow** — the flow-matching generative engine (`deltaflow.interpolants`,
  `deltaflow.samplers`, `deltaflow.losses.FlowMatchingLoss`) that produces
  and consumes that guidance.

See [Getting Started](getting-started.md) for installation and a first
example, or browse the `examples/` tree in the repository for a full,
runnable curriculum.
