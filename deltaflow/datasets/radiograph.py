"""
Generic radiograph dataset base class, plus thin subclasses for the three
common modalities (chest, cephalometric, hand). These wrap a flat directory
of grayscale images (optionally with a landmark annotation file) and are
intentionally minimal, bring your own annotation parser for a specific
benchmark by subclassing and overriding `_load_landmarks`.
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
        landmarks_file: optional path to a landmark annotation file, parsed
            by `_load_landmarks`, which subclasses may override for a
            specific benchmark's file format. Landmarks must be returned in
            the *original* image's pixel coordinates (row/col or x/y as the
            subclass defines, consistently), the base class rescales them to
            match ``image_size`` and (optionally) normalises them.
        normalize_landmarks: if ``True`` (default), landmarks returned by
            ``__getitem__`` are mapped from resized-image pixel coordinates
            to ``[-1, 1]``, matching the convention used by every other
            DeltaFlow example (e.g. ``examples/90-showcase/07-landmark-detection``).
            Set to ``False`` to keep them in resized-image pixel units.
    """

    def __init__(
        self,
        root: Union[str, Path],
        image_size: Optional[int] = None,
        transform: Optional[Callable] = None,
        landmarks_file: Optional[Union[str, Path]] = None,
        normalize_landmarks: bool = True,
    ):
        if Image is None:
            raise ImportError("Pillow is required for RadiographDataset (pip install pillow)")

        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform
        self.normalize_landmarks = normalize_landmarks
        self.image_paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        self.landmarks = self._load_landmarks(landmarks_file) if landmarks_file else None
        if self.landmarks is not None:
            # Keep only images that actually carry an annotation. Real
            # benchmarks ship official train/test splits as separate
            # annotation files, so ``root`` typically holds more images than
            # any single split annotates; dropping the rest is what lets one
            # image directory be reused across splits.
            self.image_paths = [p for p in self.image_paths if p.stem in self.landmarks]
            if not self.image_paths:
                raise ValueError(
                    f"None of the images under {self.root} matched an annotation "
                    "entry (checked by file stem). Check that image filenames "
                    "line up with the annotation keys."
                )

    def _load_landmarks(self, landmarks_file: Union[str, Path]) -> dict:
        """Override in a subclass to parse a benchmark-specific annotation format.

        Must return a mapping ``{image_stem: tensor_of_shape_(n_landmarks, 2)}``,
        with each row an ``(x, y)`` pair in the *original* (pre-resize) image's
        pixel coordinates.
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
        orig_w, orig_h = img.size
        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size))
        if self.transform is not None:
            img = self.transform(img)

        x = torch.from_numpy(_to_numpy(img)).float().unsqueeze(0) / 255.0

        if self.landmarks is None:
            return x

        lm = self.landmarks[path.stem].clone().float()  # (K, 2) in (x, y)
        if self.image_size is not None:
            # Rescale from the original image's pixel grid to the resized
            # image's pixel grid. Radiographs are rarely square, so this is
            # two independent (possibly different) scale factors, not one.
            scale = torch.tensor(
                [self.image_size / orig_w, self.image_size / orig_h], dtype=lm.dtype
            )
            lm = lm * scale
            if self.normalize_landmarks:
                lm = (lm / (self.image_size - 1)) * 2.0 - 1.0
        return x, lm.reshape(-1)


def _to_numpy(img):
    import numpy as np

    return np.array(img)


class ChestXrayDataset(RadiographDataset):
    """Frontal chest radiographs (e.g. Shenzhen, NIH ChestX-ray14)."""


class CephalometricDataset(RadiographDataset):
    """Lateral cephalometric radiographs (e.g. ISBI2015)."""


class ISBI2015CephalometricDataset(CephalometricDataset):
    """ISBI2015 "Automatic Cephalometric X-Ray Landmark Detection" benchmark.

    Wang et al., "A benchmark for comparison of dental radiography analysis
    algorithms", Medical Image Analysis (2016). 400 lateral cephalograms
    (``1935x2400`` px, ``0.1`` mm/px), each with 19 anatomical landmarks
    annotated independently by a senior and a junior rater.

    Two annotation layouts are recognised, picked automatically from
    ``landmarks_file``:

    * **Per-image text files** (the original figshare release). Point
      ``landmarks_file`` at a *directory* of ``.txt`` files named after each
      image stem (e.g. ``001.txt``), one ``x,y`` pixel pair per line in a
      fixed anatomical order. Some releases append extra classification lines
      after the 19th point, which are ignored. Pass ``landmarks_file_2`` for a
      second rater directory to average the two raters into one ground truth.
    * **Consolidated CSV** (common on Kaggle mirrors, e.g.
      ``jiahongqian/cephalometric-landmarks``). Point ``landmarks_file`` at a
      ``.csv`` whose header is ``image_path,1_x,1_y,2_x,2_y,...`` and whose
      rows give one image's filename followed by the landmark pixel
      coordinates. Only the images listed in the CSV are kept, which is how
      the dataset's official train/test splits (separate CSV files) are
      honoured while every image lives in one directory.

    Args:
        root: directory of radiograph images (searched recursively).
        image_size: images (and landmarks) are rescaled to this square size.
        landmarks_file: a directory of per-image ``.txt`` annotations, or a
            single ``.csv`` split file (see above).
        landmarks_file_2: optional second rater's ``.txt`` directory (e.g.
            ``400_junior``). If given, the two raters' points are averaged
            per landmark. Only applies to the per-image text layout.
        n_landmarks: landmarks per image (19 for the ISBI2015 challenge).
    """

    def __init__(
        self,
        root: Union[str, Path],
        image_size: Optional[int] = None,
        transform: Optional[Callable] = None,
        landmarks_file: Optional[Union[str, Path]] = None,
        landmarks_file_2: Optional[Union[str, Path]] = None,
        normalize_landmarks: bool = True,
        n_landmarks: int = 19,
    ):
        self.n_landmarks = n_landmarks
        self._landmarks_file_2 = Path(landmarks_file_2) if landmarks_file_2 else None
        super().__init__(
            root,
            image_size=image_size,
            transform=transform,
            landmarks_file=landmarks_file,
            normalize_landmarks=normalize_landmarks,
        )

    def _parse_points(self, txt_path: Union[str, Path]) -> torch.Tensor:
        """Parse the first ``n_landmarks`` ``x,y`` lines of one annotation file."""
        pts = []
        for line in Path(txt_path).read_text().splitlines():
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue  # trailing classification/measurement lines, not points
            if len(pts) == self.n_landmarks:
                break
        if len(pts) != self.n_landmarks:
            raise ValueError(
                f"Expected {self.n_landmarks} landmark points in {txt_path}, found {len(pts)}."
            )
        return torch.tensor(pts, dtype=torch.float32)

    def _load_landmarks_csv(self, csv_path: Path) -> dict:
        """Parse a wide ``image_path,1_x,1_y,...`` CSV split file."""
        import csv as _csv

        landmarks = {}
        with open(csv_path, newline="") as f:
            reader = _csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"Empty CSV annotation file: {csv_path}")
            for row in reader:
                if not row or not row[0].strip():
                    continue
                coords = [float(v) for v in row[1 : 1 + 2 * self.n_landmarks]]
                if len(coords) != 2 * self.n_landmarks:
                    raise ValueError(
                        f"Row for {row[0]!r} in {csv_path} has {len(coords)} "
                        f"coordinates, expected {2 * self.n_landmarks}."
                    )
                stem = Path(row[0].strip()).stem
                landmarks[stem] = torch.tensor(coords, dtype=torch.float32).reshape(
                    self.n_landmarks, 2
                )
        return landmarks

    def _load_landmarks(self, landmarks_file: Union[str, Path]) -> dict:
        path = Path(landmarks_file)
        if path.is_file() and path.suffix.lower() == ".csv":
            return self._load_landmarks_csv(path)

        landmarks = {}
        for txt_path in sorted(path.glob("*.txt")):
            pts = self._parse_points(txt_path)
            if self._landmarks_file_2 is not None:
                pts_2 = self._parse_points(self._landmarks_file_2 / txt_path.name)
                pts = (pts + pts_2) / 2.0
            landmarks[txt_path.stem] = pts
        return landmarks


class HandRadiographDataset(RadiographDataset):
    """Hand/wrist radiographs (e.g. DHA)."""
