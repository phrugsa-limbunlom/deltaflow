"""Image-scale training utilities: streaming datasets, coupling strategies,
and a mixed-precision training loop with checkpoint/resume."""

from .coupling import BaseCoupling, IndependentCoupling, OTCoupling
from .data import ImageFolderStream, build_loader
from .loop import TrainConfig, load_checkpoint, save_checkpoint, train

__all__ = [
    "BaseCoupling",
    "ImageFolderStream",
    "IndependentCoupling",
    "OTCoupling",
    "TrainConfig",
    "build_loader",
    "load_checkpoint",
    "save_checkpoint",
    "train",
]
