"""Image file I/O shared by the image datasets.

Pillow is an optional extra (``pip install torchdeltaflow[images]``), so it is
the only dependency resolved lazily. Every image-loading module goes through
this adapter rather than importing Pillow itself, which keeps the optional
dependency handling in exactly one place.
"""

import importlib
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Sequence, Union

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

_PILLOW_INSTALL_HINT = (
    "Pillow is required for image loading. Install it with "
    "`pip install torchdeltaflow[images]` or `pip install pillow`."
)


def require_pillow() -> ModuleType:
    """Return the ``PIL.Image`` module, or raise an actionable `ImportError`."""
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as err:
        raise ImportError(_PILLOW_INSTALL_HINT) from err


def list_images(root: Union[str, Path], extensions: Sequence[str] = IMAGE_EXTENSIONS) -> List[Path]:
    """Recursively list image files under ``root`` in a deterministic order."""
    exts = tuple(e.lower() for e in extensions)
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in exts)


def open_image(path: Union[str, Path], mode: str, size: Optional[int] = None):
    """Open ``path`` as a PIL image in ``mode``, optionally resized to ``size x size``."""
    img = require_pillow().open(path).convert(mode)
    if size is not None:
        img = img.resize((size, size))
    return img


__all__ = ["IMAGE_EXTENSIONS", "list_images", "open_image", "require_pillow"]
