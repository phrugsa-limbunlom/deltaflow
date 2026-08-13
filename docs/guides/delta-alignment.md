# Delta Alignment

Delta alignment is the **"Delta"** in DeltaFlow, a multi-scale,
anatomy-cancelling loss for guidance-representation pretraining.

## The idea

For a conditionally-generated backbone, run both a conditional and an
unconditional forward pass and take the **guidance-difference feature** at each
hierarchy level:

\[
\Delta h = h_\text{cond} - h_\text{uncond}.
\]

\(\Delta h\) isolates *what the conditioning signal changed*, largely
cancelling the anatomy-specific content that both passes share. Aligning
\(\Delta h\) across two augmented views of the same input (rather than
aligning the raw, anatomy-entangled features) encourages a guidance
representation that is consistent regardless of the anatomy it is applied to.
This is especially valuable in **data-scarce** medical-imaging settings, where
the shared anatomy would otherwise dominate a naive feature-alignment signal.

## Usage

```python
from deltaflow.losses import DeltaAlignmentLoss
from deltaflow.models import MultiScaleProjector

projector = MultiScaleProjector(feature_dims={"enc_1_4": 256, "bottleneck": 1024})
loss_fn = DeltaAlignmentLoss(projector, lambda_flow=1.0, lambda_align=5.0)

total, loss_dict = loss_fn(
    v_c1, v_u1, target_v1,      # view 1: cond / uncond velocities + FM target
    v_c2, v_u2, target_v2,      # view 2
    feats_u1, feats_c1,         # view 1: per-level uncond / cond feature dicts
    feats_u2, feats_c2,         # view 2
)
```

- `lambda_flow` weights the flow-matching term (keeps the backbone generative).
- `lambda_align` weights the \(\Delta h\) alignment term.
- The returned `loss_dict` breaks out each component for logging.

## Multi-scale projection

The [`MultiScaleProjector`](../api/models.md) holds one projection head per
feature level (keyed by name, e.g. `"enc_1_4"`, `"bottleneck"`). Each level's
\(\Delta h\) is projected and L2-normalized before the cosine-based alignment,
so levels with different channel counts contribute comparably.

!!! note "Numerical stability"
    The loss uses the helpers in [`deltaflow.utils`](../api/utils.md)
    (`safe_normalize`, `clamp_*`) so it stays finite under FP16/BF16 autocast.
