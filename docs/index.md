# DeltaFlow

DeltaFlow provides composable PyTorch primitives for generative modelling in
**data-scarce domains** such as medical imaging, with a focus on
*representation learning* and *studying the underlying data distribution* (not
just sample synthesis).

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting Started](getting-started/installation.md)**

    Install DeltaFlow and fit your first flow-matching model in a few lines.

- :material-book-open-variant: **[Guides](guides/flow-matching.md)**

    Task-oriented walkthroughs: flow matching, delta alignment, training, and
    inverse problems.

- :material-sitemap: **[Architecture](concepts/architecture.md)**

    The four base abstractions that make every component drop-in.

- :material-api: **[API Reference](api/index.md)**

    Auto-generated documentation for every public class and function.

</div>

## What is DeltaFlow?

DeltaFlow is built on a single idea, a **delta** (the change carried from one
distribution to another) and the **flow** that realises it.

- **Δ, the change between two things.** The learning signal is a *displacement*
  \(\Delta = x_1 - x_0\), the straight-line velocity carrying a source sample
  \(x_0\) (noise) to a data sample \(x_1\). Regressing a velocity field onto
  \(\Delta\) is exactly the flow-matching objective (Lipman et al., 2023).
- **Flow, integrating the change.** A learned field \(v_\theta(x, t)\) is
  integrated as an ODE \(dx/dt = v_\theta(x, t)\), transporting the *whole*
  source distribution onto the data distribution.
- **Straighter Δ via optimal transport.** **Mini-batch optimal-transport
  coupling** solves the discrete Kantorovich assignment on each batch, the
  source to target pairing that minimises squared transport cost, so the
  displacements, and the flow that follows them, are as straight as possible and
  sample in fewer steps. This hard assignment is the zero-entropy limit of the
  static Schrödinger bridge (Tong et al., 2023).

The *same difference principle* also drives an optional representation-learning
loss, **delta alignment**, which aligns the guidance-difference feature
\(\Delta h = h_\text{cond} - h_\text{uncond}\) for generative pretraining in
data-scarce domains.

The library targets 2D radiography (chest X-ray, cephalometric, and hand
radiographs), but the core primitives (`interpolants`, `samplers`, `losses`,
`models.projector`) are domain-agnostic.

## Visualizing the algorithm

Flow matching learns a time-conditioned velocity field \(v(x, t)\) that
transports a simple source density (Gaussian noise, \(t=0\)) onto the data
distribution (\(t=1\)). The panels below come from the
[sampling-flow example](examples.md).

![Sampling snapshots from noise to data](assets/sampling-flow/snapshots.png)

![Animated sampling flow](assets/sampling-flow/flow.gif)

## What's inside

| Module | Role |
|---|---|
| `deltaflow.core` | Abstract base classes every component subclasses |
| `deltaflow.interpolants` | Probability paths (linear, mini-batch OT, variance-preserving) |
| `deltaflow.samplers` / `deltaflow.solvers` | Euler & Heun integration, posterior sampling |
| `deltaflow.losses` | Flow matching + delta alignment |
| `deltaflow.models` | Backbones, EMA, multi-scale projector heads |
| `deltaflow.inverse` | Measurement operators, Tweedie decomposition, likelihoods |
| `deltaflow.trainer` | Streaming data, mixed-precision loop, OT coupling |
| `deltaflow.datasets` | Radiograph dataset wrappers (chest / cephalo / hand) |

## Citation

If you use DeltaFlow in your research, please cite it (see
[`CITATION.cff`](https://github.com/phrugsa-limbunlom/deltaflow/blob/main/CITATION.cff)).
