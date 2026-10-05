"""Model components: velocity-field backbone wrappers, projector heads, EMA,
and the DiT transformer with adaLN-Zero conditioning."""

from .backbone import TinyVelocityField, WrappedBackbone
from .dit import DiT, DiTBlock, FinalLayer, LabelEmbedding, TimestepEmbedding, modulate
from .ema import EMA
from .projector import MultiScaleProjector, ProjectorHead

__all__ = [
    "DiT",
    "DiTBlock",
    "EMA",
    "FinalLayer",
    "LabelEmbedding",
    "MultiScaleProjector",
    "ProjectorHead",
    "TimestepEmbedding",
    "TinyVelocityField",
    "WrappedBackbone",
    "modulate",
]
