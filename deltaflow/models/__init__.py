"""Model components: projector heads and EMA weight averaging."""

from .ema import EMA
from .projector import MultiScaleProjector, ProjectorHead

__all__ = ["EMA", "MultiScaleProjector", "ProjectorHead"]
