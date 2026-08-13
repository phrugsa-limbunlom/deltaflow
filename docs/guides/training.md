# Training at Image Scale

`deltaflow.trainer` provides a **framework-light** training loop (no
Trainer/Fabric/Accelerate) that still handles image-scale workloads and drops
into any script.

## Features

- Mixed precision (`torch.amp.autocast` + `GradScaler`).
- Gradient accumulation.
- Checkpoint + resume (model, optimizer, scaler, EMA, step counter).
- Optional EMA of the velocity-field weights.
- Pluggable train-time coupling via `BaseCoupling`.

## Minimal loop

```python
from deltaflow.trainer import TrainConfig, train, build_loader
from deltaflow.trainer.coupling import OTCoupling

loader = build_loader(...)                 # your DataLoader
config = TrainConfig(
    max_steps=10_000,
    grad_accum_steps=4,
    use_ema=True,
    coupling=OTCoupling(),                 # or IndependentCoupling()
)
train(model, loss_fn, loader, optimizer, config)
```

## Coupling strategies

At train time each minibatch pairs data samples \(x_1\) with noise \(x_0\).
The pairing is pluggable:

| Coupling | Behaviour |
|---|---|
| `IndependentCoupling` | Pairs noise and data independently (standard flow matching) |
| `OTCoupling` | Solves a mini-batch OT assignment so paired \((x_0, x_1)\) are close, yielding straighter flows |

`OTCoupling` uses the Hungarian algorithm when `scipy` is available (install
the [`ot` extra](../getting-started/installation.md)) and a greedy
nearest-neighbour matcher otherwise.

## Checkpointing

```python
from deltaflow.trainer import save_checkpoint, load_checkpoint

save_checkpoint(path, model, optimizer, scaler, ema, step)
state = load_checkpoint(path, model, optimizer, scaler, ema)
```

See the [`trainer` API reference](../api/trainer.md) for the full signatures.
