"""Image-scale training loop for flow matching.

Features required by the design constraints:

- Mixed precision (``torch.amp.autocast`` + ``GradScaler``).
- Gradient accumulation.
- Checkpoint + resume (model, optimizer, scaler, EMA, step counter).
- Optional EMA of the velocity field weights.
- Pluggable train-time coupling via
  `BaseCoupling`.

The loop is intentionally framework-light - no Trainer/Fabric/etc. - so it
can be dropped into any script and still handle image-scale workloads.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..core.base_loss import BaseLoss
from ..models.ema import EMA


@dataclass
class TrainConfig:
    """Configuration for `train`.

    Attributes:
        max_steps: total optimizer steps to run (after ``resume_from``).
        grad_accum_steps: accumulate gradients over this many minibatches
            before each optimizer step.
        mixed_precision: if ``True``, run forward+backward under
            ``torch.amp.autocast`` and scale the loss with
            `torch.amp.GradScaler`. On CUDA uses ``bfloat16`` if
            supported, else ``float16``. On CPU it is a no-op.
        amp_dtype: override the autocast dtype. If ``None``, chosen from
            the runtime.
        grad_clip: max L2 norm for gradient clipping, ``None`` to disable.
        log_every: print a training-metric line every N optimizer steps.
        checkpoint_every: write a checkpoint every N optimizer steps.
        checkpoint_dir: directory to write checkpoints into. Created if
            missing.
        ema_beta: if not ``None``, maintain an EMA copy of the model with
            this decay rate.
        device: torch device string, defaults to CUDA if available.
        resume_from: optional path to a checkpoint saved by this loop.
    """

    max_steps: int = 1_000
    grad_accum_steps: int = 1
    mixed_precision: bool = True
    amp_dtype: Optional[torch.dtype] = None
    grad_clip: Optional[float] = 1.0
    log_every: int = 50
    checkpoint_every: int = 500
    checkpoint_dir: Union[str, Path] = "checkpoints"
    ema_beta: Optional[float] = 0.999
    device: Optional[str] = None
    resume_from: Optional[Union[str, Path]] = None
    log_fn: Callable[[str], None] = field(default=print)


def _select_amp_dtype(device: torch.device, override: Optional[torch.dtype]) -> torch.dtype:
    if override is not None:
        return override
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _resolve_device(spec: Optional[str]) -> torch.device:
    if spec is not None:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_x1(batch) -> torch.Tensor:
    """Get the data tensor from a batch that may also carry labels/paths."""
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], torch.Tensor):
        return batch[0]
    if isinstance(batch, dict):
        for key in ("x", "image", "data"):
            if key in batch:
                return batch[key]
    raise TypeError(f"Cannot extract x1 tensor from batch of type {type(batch).__name__}")


def _infinite(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    ema_model: Optional[nn.Module] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> None:
    """Write a checkpoint dictionary to ``path`` (creates parent dir)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if ema_model is not None:
        ckpt["ema_model"] = ema_model.state_dict()
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    ema_model: Optional[nn.Module] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> int:
    """Load a checkpoint written by `save_checkpoint`. Returns the step."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema_model is not None and "ema_model" in ckpt:
        ema_model.load_state_dict(ckpt["ema_model"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0))


def train(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: BaseLoss,
    dataloader: DataLoader,
    config: Optional[TrainConfig] = None,
) -> nn.Module:
    """Train ``model`` in-place under ``loss_fn`` for ``config.max_steps`` steps.

    Returns the (possibly EMA-shadowed) model that was trained. The original
    ``model`` is always updated in place. If EMA is enabled, an EMA copy is
    also kept and written into checkpoints, and can be retrieved from the
    latest checkpoint's ``"ema_model"`` key.
    """
    cfg = config or TrainConfig()
    device = _resolve_device(cfg.device)
    model.to(device)

    amp_dtype = _select_amp_dtype(device, cfg.amp_dtype)
    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    ema = EMA(beta=cfg.ema_beta) if cfg.ema_beta is not None else None
    ema_model: Optional[nn.Module] = None
    if ema is not None:
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    start_step = 0
    if cfg.resume_from is not None:
        start_step = load_checkpoint(
            cfg.resume_from,
            model=model,
            optimizer=optimizer,
            ema_model=ema_model,
            scaler=scaler if use_amp else None,
            map_location=device,
        )
        cfg.log_fn(f"[deltaflow.trainer] resumed at step {start_step} from {cfg.resume_from}")

    ckpt_dir = Path(cfg.checkpoint_dir)
    data_iter = _infinite(dataloader)

    model.train()
    step = start_step
    accum_i = 0
    running_loss = 0.0
    t_last = time.time()

    optimizer.zero_grad(set_to_none=True)

    while step < cfg.max_steps:
        batch = next(data_iter)
        x1 = _extract_x1(batch).to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            loss = loss_fn(model, x1)
            loss = loss / cfg.grad_accum_steps

        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += loss.item() * cfg.grad_accum_steps
        accum_i += 1

        if accum_i < cfg.grad_accum_steps:
            continue
        accum_i = 0

        if cfg.grad_clip is not None:
            if use_amp and amp_dtype == torch.float16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        if use_amp and amp_dtype == torch.float16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if ema_model is not None:
            ema.update_model_average(ema_model, model)

        step += 1

        if step % cfg.log_every == 0:
            avg_loss = running_loss / (cfg.log_every * cfg.grad_accum_steps)
            dt = time.time() - t_last
            cfg.log_fn(
                f"[deltaflow.trainer] step {step:>7d} | loss {avg_loss:.4f} | {dt:.1f}s / {cfg.log_every} steps"
            )
            running_loss = 0.0
            t_last = time.time()

        if step % cfg.checkpoint_every == 0:
            save_checkpoint(
                ckpt_dir / f"step_{step:07d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                ema_model=ema_model,
                scaler=scaler if use_amp else None,
            )

    save_checkpoint(
        ckpt_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        step=step,
        ema_model=ema_model,
        scaler=scaler if use_amp else None,
    )
    return ema_model if ema_model is not None else model


__all__ = ["TrainConfig", "train", "save_checkpoint", "load_checkpoint"]
