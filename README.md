# DeltaFlow

**Flow matching and anatomy-invariant guidance alignment for radiograph generative pretraining.**

`DeltaFlow` provides composable PyTorch primitives for two things that are
usually bundled together in ad-hoc research code:

- **Flow matching** — a simulation-free generative objective that regresses
  a velocity field onto the conditional velocity of a probability path
  (Lipman et al., 2023), sampled with a simple Euler ODE solver.
- **Delta alignment** — a multi-scale, anatomy-cancelling loss that aligns
  the *guidance-difference* feature `Δh = h_cond - h_uncond` across
  augmented views, instead of aligning raw (anatomy-entangled) features.
  This is the mechanism behind the name: **Delta** is the guidance-difference
  feature, **Flow** is the flow-matching engine that consumes it.

The library targets 2D radiography (chest X-ray, cephalometric, and hand
radiographs), but the core primitives (`interpolants`, `samplers`, `losses`,
`models.projector`) are domain-agnostic.

## What's inside

| Module | Role |
|---|---|
| `deltaflow.core` | `BaseVelocityField` — the interface every backbone implements |
| `deltaflow.interpolants` | Probability paths, e.g. `LinearInterpolant` (rectified flow) |
| `deltaflow.samplers` | `FlowSampler` — Euler ODE integration |
| `deltaflow.losses` | `FlowMatchingLoss`, `DeltaAlignmentLoss` |
| `deltaflow.models` | `MultiScaleProjector` (per-level projection heads), `EMA` |
| `deltaflow.datasets` | Generic radiograph dataset wrappers (chest / cephalo / hand) |

## Install

```bash
pip install -e .
# or, for docs/dev tooling:
pip install -e ".[dev]"
```

## Usage

### Flow matching

```python
import torch
from deltaflow.interpolants import LinearInterpolant
from deltaflow.losses import FlowMatchingLoss
from deltaflow.samplers import FlowSampler

loss_fn = FlowMatchingLoss(interpolant=LinearInterpolant())
loss = loss_fn(model, x1)               # model(x_t, t) -> predicted velocity
loss.backward()

samples = FlowSampler(model).sample(torch.randn(1000, 2), n_steps=50)
```

### Delta alignment

```python
from deltaflow.losses.delta_alignment import DeltaAlignmentLoss
from deltaflow.models import MultiScaleProjector

projector = MultiScaleProjector(feature_dims={"enc_1_4": 256, "bottleneck": 1024})
loss_fn = DeltaAlignmentLoss(projector, lambda_flow=1.0, lambda_align=5.0)

total, loss_dict = loss_fn(
    v_c1, v_u1, target_v1,
    v_c2, v_u2, target_v2,
    feats_u1, feats_c1, feats_u2, feats_c2,
)
```

See [`examples/`](examples/) for runnable, tiered walkthroughs
(`00-foundations/`, `10-sampling/`, `20-training/`, `90-showcase/`).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use DeltaFlow in your research, please cite it (see [CITATION.cff](CITATION.cff)).
