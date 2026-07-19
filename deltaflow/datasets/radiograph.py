"""
Generic radiograph dataset base class, plus thin subclasses for the three
common modalities (chest, cephalometric, hand). These wrap a flat directory
of grayscale images (optionally with a landmark annotation file) and are
intentionally minimal -- bring your own annotation parser for a specific
benchmark by subclassing and overriding :meth:`_load_landmarks`.
"""

from pathlib import Path
from typing import Callable, Optional, Union

import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class RadiographDataset(Dataset):
    """Base dataset over a flat directory of grayscale radiographs.

    Args:
        root: directory containing image files.
        image_size: if given, images are resized to ``(image_size, image_size)``.
        transform: optional callable applied to the loaded PIL image before
            conversion to a tensor. Receives and must return a PIL image.
        landmarks_file: optional path to a landmark annotation file; parsed
            by :meth:`_load_landmarks`, which subclasses may override for a
            specific benchmark's file format.
    """

    def __init__(
        self,
        root: Union[str, Path],
        image_size: Optional[int] = None,
        transform: Optional[Callable] = None,
        landmarks_file: Optional[Union[str, Path]] = None,
    ):
        if Image is None:
            raise ImportError("Pillow is required for RadiographDataset (pip install pillow)")

        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform
        self.image_paths = sorted(
            p for p in self.root.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        self.landmarks = self._load_landmarks(landmarks_file) if landmarks_file else None

    def _load_landmarks(self, landmarks_file: Union[str, Path]) -> dict:
        """Override in a subclass to parse a benchmark-specific annotation format.

        Must return a mapping ``{image_stem: tensor_of_shape_(n_landmarks, 2)}``.
        """
        raise NotImplementedError(
            "Provide a `landmarks_file` parser by subclassing RadiographDataset "
            "and overriding `_load_landmarks`."
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = Image.open(path).convert("L")
        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size))
        if self.transform is not None:
            img = self.transform(img)

        x = torch.from_numpy(_to_numpy(img)).float().unsqueeze(0) / 255.0

        if self.landmarks is None:
            return x
        return x, self.landmarks[path.stem]


def _to_numpy(img):
    import numpy as np

    return np.array(img)


class ChestXrayDataset(RadiographDataset):
    """Frontal chest radiographs (e.g. Shenzhen, NIH ChestX-ray14)."""


class CephalometricDataset(RadiographDataset):
    """Lateral cephalometric radiographs (e.g. ISBI2015)."""


class HandRadiographDataset(RadiographDataset):
    """Hand/wrist radiographs (e.g. DHA)."""
