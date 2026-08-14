"""Streaming image-dataset loaders for image-scale flow-matching training.

The primary class is `ImageFolderStream`, a thin
`torch.utils.data.Dataset` over a directory of image files that
opens each file on demand rather than pre-loading everything into memory.
It is intentionally minimal - if you already have a dataset (HF datasets,
WebDataset, an in-house one) just plug it into
`deltaflow.trainer.loop.train` directly.
"""

from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


class ImageFolderStream(Dataset):
    """Streaming dataset over a directory of images.

    Args:
        root: directory containing image files (searched recursively).
        image_size: if given, images are resized to ``(image_size, image_size)``.
        mode: PIL mode to convert to (``"L"`` for grayscale, ``"RGB"`` for
            colour, ``"F"`` for float).
        transform: optional PIL-image transform applied *before* tensor
            conversion. If it returns a `torch.Tensor`, that tensor
            is returned as-is (i.e. no default rescaling is applied).
        extensions: file extensions to include.
    """

    def __init__(
        self,
        root: Union[str, Path],
        image_size: Optional[int] = None,
        mode: str = "RGB",
        transform: Optional[Callable] = None,
        extensions: Sequence[str] = _IMAGE_EXTENSIONS,
    ):
        if Image is None:
            raise ImportError("Pillow is required for ImageFolderStream (pip install pillow)")
        self.root = Path(root)
        self.image_size = image_size
        self.mode = mode
        self.transform = transform
        exts = tuple(e.lower() for e in extensions)
        self.paths = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in exts)
        if not self.paths:
            raise ValueError(f"No images with extensions {exts} found under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert(self.mode)
        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size))
        if self.transform is not None:
            out = self.transform(img)
            if isinstance(out, torch.Tensor):
                return out
            img = out
        import numpy as np

        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[..., None]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
        # Rescale to [-1, 1] so the target distribution matches a standard-normal source.
        return tensor * 2.0 - 1.0

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._load(self.paths[idx])


def build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 2,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Convenience wrapper around `torch.utils.data.DataLoader` with
    sensible defaults for image-scale flow-matching training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


__all__ = ["ImageFolderStream", "build_loader"]
