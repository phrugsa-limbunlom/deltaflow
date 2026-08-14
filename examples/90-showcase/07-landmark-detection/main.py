"""90-showcase/07-landmark-detection: image-conditioned landmark detection as
conditional flow matching.

Landmark detection maps an image (here an X-ray) to a set of coordinate
points. This is *not* a linear inverse problem, so the posterior sampler
(:class:`~deltaflow.solvers.posterior_solver.PosteriorSolver`) is the wrong
tool -- it would need a known differentiable operator ``A: landmarks -> image``,
which does not exist. Instead we model the conditional distribution

    p(landmarks | image)

directly with conditional flow matching: the *data* being generated is the
vector of landmark coordinates ``x1`` (shape ``(B, 2K)``), and the X-ray image
is passed as conditioning to the velocity field. Exactly the same training loop
and loss as every other DeltaFlow example are reused; only the data and the
conditioning change.

Because the model is generative, sampling it several times from different noise
seeds yields a *distribution* over each landmark position -- built-in
uncertainty that plain coordinate regression cannot provide.

By default this script is self-contained: it builds a synthetic task
(grayscale images with a few bright Gaussian blobs; the landmarks are the
blob centres) so it runs end to end with no data download. Pass ``--data-root``
and ``--landmarks-root`` to train on a real dataset instead, e.g. the
ISBI2015 lateral cephalometric benchmark via
:class:`~deltaflow.datasets.ISBI2015CephalometricDataset`. Everything else
(model, training loop, sampling, visualisation) is unchanged, only where the
``(image, landmarks)`` pairs come from differs.

Run::

    python examples/90-showcase/07-landmark-detection/main.py

    # or, on the original ISBI2015 figshare layout (per-image .txt files,
    # RawImage/ + AnnotationsByMD/400_{senior,junior}):
    python examples/90-showcase/07-landmark-detection/main.py \\
        --data-root path/to/RawImage \\
        --landmarks-root path/to/AnnotationsByMD/400_senior \\
        --landmarks-root-2 path/to/AnnotationsByMD/400_junior \\
        --n-landmarks 19 --image-size 64

    # or, on the Kaggle mirror (consolidated CSV splits,
    # kaggle datasets download -d jiahongqian/cephalometric-landmarks):
    python examples/90-showcase/07-landmark-detection/main.py \\
        --data-root path/to/cepha400 \\
        --landmarks-root path/to/train_senior.csv \\
        --test-landmarks-root path/to/test1_senior.csv \\
        --n-landmarks 19 --image-size 64

Outputs are written to ``outputs/landmark_detection/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - install hint only
    raise SystemExit(
        "This example needs matplotlib. Install it with:\n" "    pip install matplotlib"
    ) from exc

from deltaflow.core.base import BaseVelocityField
from deltaflow.interpolants import LinearInterpolant
from deltaflow.losses import ConditionalFlowMatchingLoss
from deltaflow.solvers import EulerSolver

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE = 48  # synthetic image resolution (square)
N_LANDMARKS = 3  # number of landmarks (blob centres) per image
DATA_DIM = 2 * N_LANDMARKS
BLOB_SIGMA = 3.0  # blob radius in pixels
N_TRAIN = 2000
N_STEPS = 2500
BATCH = 128
LR = 2e-3
SAMPLES_PER_IMAGE = 40  # draws per test image, to visualise uncertainty
SEED = 0

C_GT = "#e07a3f"  # ground-truth landmark markers (warm)
C_PRED = "#0f9bab"  # predicted samples (teal)
C_MEAN = "#155e63"  # predicted mean (deep teal)


# ---------------------------------------------------------------------------
# Synthetic data: images with bright blobs; landmarks are the blob centres
# ---------------------------------------------------------------------------
def make_dataset(n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(images, landmarks)``.

    ``images``: ``(n, 1, IMG_SIZE, IMG_SIZE)`` in ``[0, 1]``.
    ``landmarks``: ``(n, 2K)`` blob centres, normalised to ``[-1, 1]``.
    """
    # Landmark centres in pixel coordinates, kept away from the border.
    margin = int(3 * BLOB_SIGMA)
    lo, hi = margin, IMG_SIZE - margin
    centres_px = torch.randint(
        lo, hi, size=(n, N_LANDMARKS, 2), generator=generator
    ).float()  # (n, K, 2) as (row, col)

    # Give the landmarks a canonical, image-derivable order (left -> right by
    # column). Without this the target ordering is arbitrary relative to image
    # content, making p(landmarks|image) permutation-multimodal and the mean
    # error meaningless. Sorting removes that ambiguity.
    order = torch.argsort(centres_px[..., 1], dim=1)  # by column (x)
    centres_px = torch.gather(centres_px, 1, order.unsqueeze(-1).expand(-1, -1, 2))

    ys = torch.arange(IMG_SIZE).view(1, 1, IMG_SIZE, 1).float()
    xs = torch.arange(IMG_SIZE).view(1, 1, 1, IMG_SIZE).float()
    cy = centres_px[..., 0].view(n, N_LANDMARKS, 1, 1)
    cx = centres_px[..., 1].view(n, N_LANDMARKS, 1, 1)

    # Sum of Gaussian blobs, one per landmark.
    blobs = torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * BLOB_SIGMA**2))
    img = blobs.sum(dim=1, keepdim=True).clamp(max=1.0)  # (n, 1, H, W)

    # Mild speckle so the task is not perfectly noise-free.
    noise = 0.05 * torch.rand(img.shape, generator=generator)
    img = (img + noise).clamp(0.0, 1.0)

    # Normalise landmark coordinates to [-1, 1] as (x, y) = (col, row).
    centres_xy = centres_px.flip(-1)  # (row, col) -> (col, row) i.e. (x, y)
    landmarks = (centres_xy / (IMG_SIZE - 1)) * 2.0 - 1.0
    return img, landmarks.reshape(n, DATA_DIM)


# ---------------------------------------------------------------------------
# Image-conditioned velocity field over landmark coordinates
# ---------------------------------------------------------------------------
class LandmarkVelocityField(BaseVelocityField):
    """Predict a velocity in landmark space from ``(x_t, t, image)``.

    A small CNN encodes the conditioning image into a feature vector, which is
    concatenated with the noised landmark vector ``x_t`` and the time ``t`` and
    fed to an MLP that outputs the velocity.
    """

    def __init__(self, data_dim: int = DATA_DIM, cond_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # H/2
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # H/4
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # H/8
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Keep a coarse spatial grid (4x4) rather than a single vector:
            # global pooling would discard *where* the blobs are, which is
            # exactly the signal a localiser needs.
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, cond_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1 + cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, image: torch.Tensor = None, **cond
    ) -> torch.Tensor:
        if image is None:
            raise ValueError("LandmarkVelocityField requires an `image` conditioning tensor.")
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        c = self.encoder(image)  # (B, cond_dim)
        h = torch.cat([x, t.view(-1, 1).to(x.dtype), c], dim=1)
        return self.net(h)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_flow_loss(model, images, landmarks, device, loss_fn, n_draws=8, batch=256, seed=1234):
    """Average the flow-matching loss over a dataset (held-out or train).

    This is the *same* training objective evaluated on other data, averaged
    over several random ``(t, noise)`` draws with a fixed seed so the number
    is low-variance and comparable across calls. A widening train-vs-test gap
    is the overfitting signal to watch, which matters here because the real
    cephalometric split has only ~150 training images.
    """
    was_training = model.training
    model.eval()
    rng_state = torch.get_rng_state()  # keep eval RNG from perturbing training
    torch.manual_seed(seed)
    n = images.shape[0]
    total, count = 0.0, 0
    for _ in range(n_draws):
        for start in range(0, n, batch):
            img_b = images[start : start + batch].to(device)
            lm_b = landmarks[start : start + batch].to(device)
            total += loss_fn(model, lm_b, image=img_b).item() * img_b.shape[0]
            count += img_b.shape[0]
    torch.set_rng_state(rng_state)
    if was_training:
        model.train()
    return total / count


def train(model, images, landmarks, generator, device, test_images=None, test_landmarks=None):

    loss_fn = ConditionalFlowMatchingLoss(interpolant=LinearInterpolant())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = images.shape[0]

    model.train()
    for step in range(N_STEPS):
        idx = torch.randint(0, n, (BATCH,), generator=generator)
        img_b = images[idx].to(device)
        lm_b = landmarks[idx].to(device)

        loss = loss_fn(model, lm_b, image=img_b)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 250 == 0 or step == N_STEPS - 1:
            msg = f"  step {step:4d}/{N_STEPS}  train batch loss {loss.item():.4f}"
            if test_images is not None:
                tr = eval_flow_loss(model, images, landmarks, device, loss_fn)
                te = eval_flow_loss(model, test_images, test_landmarks, device, loss_fn)
                msg += f"  |  train loss {tr:.4f}  test loss {te:.4f}  (gap {te - tr:+.4f})"
            print(msg)
    model.eval()


# ---------------------------------------------------------------------------
# Sampling with uncertainty
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_landmarks(model, image, n_samples, device, generator, n_landmarks=N_LANDMARKS):
    """Draw ``n_samples`` landmark predictions for a single ``image``.

    Returns an array of shape ``(n_samples, K, 2)`` in ``[-1, 1]`` (x, y).
    """
    data_dim = 2 * n_landmarks
    img_rep = image.unsqueeze(0).repeat(n_samples, 1, 1, 1).to(device)
    x0 = torch.randn(n_samples, data_dim, generator=generator).to(device)
    solver = EulerSolver(model)
    pred = solver.sample(x0, n_steps=100, image=img_rep, show_progress=False)
    return pred.cpu().reshape(n_samples, n_landmarks, 2).numpy()


def _to_pixels(coords_xy: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """Map normalised ``[-1, 1]`` (x, y) to pixel coordinates for plotting."""
    return (coords_xy + 1.0) / 2.0 * (img_size - 1)


def visualise(model, images, landmarks, out_dir, device, generator, n_landmarks=N_LANDMARKS):
    img_size = images.shape[-1]
    n_show = 4
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4.2))

    for ax, i in zip(axes, range(n_show)):
        img = images[i]
        gt = landmarks[i].reshape(n_landmarks, 2).numpy()
        samples = sample_landmarks(model, img, SAMPLES_PER_IMAGE, device, generator, n_landmarks)

        ax.imshow(img.squeeze(0).numpy(), cmap="gray", origin="upper")

        # Predicted sample cloud (uncertainty).
        for k in range(n_landmarks):
            pts = _to_pixels(samples[:, k, :], img_size)
            ax.scatter(pts[:, 0], pts[:, 1], s=8, color=C_PRED, alpha=0.35, edgecolors="none")
            mean = pts.mean(axis=0)
            ax.scatter(*mean, s=90, marker="+", color=C_MEAN, linewidths=2.0, zorder=5)

        # Ground-truth landmarks.
        gt_px = _to_pixels(gt, img_size)
        ax.scatter(gt_px[:, 0], gt_px[:, 1], s=70, marker="x", color=C_GT, linewidths=2.0, zorder=6)

        ax.set_title(f"test image {i}")
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        plt.Line2D(
            [], [], marker="x", color=C_GT, linestyle="none", markersize=9, label="ground truth"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            color=C_PRED,
            linestyle="none",
            markersize=7,
            label="samples p(landmarks|image)",
        ),
        plt.Line2D(
            [],
            [],
            marker="+",
            color=C_MEAN,
            linestyle="none",
            markersize=10,
            label="predicted mean",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Landmark detection as conditional flow matching  -  sample spread shows uncertainty",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    out_path = out_dir / "landmark_detection.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}")


def report_error(model, images, landmarks, device, generator, n_landmarks=N_LANDMARKS):
    """Print mean per-landmark error (in pixels) using the sample mean."""
    img_size = images.shape[-1]
    errs = []
    for i in range(images.shape[0]):
        gt = landmarks[i].reshape(n_landmarks, 2).numpy()
        samples = sample_landmarks(
            model, images[i], SAMPLES_PER_IMAGE, device, generator, n_landmarks
        )
        mean = samples.mean(axis=0)
        errs.append(np.linalg.norm(_to_pixels(mean, img_size) - _to_pixels(gt, img_size), axis=1))
    errs = np.concatenate(errs)
    print(f"  mean landmark error: {errs.mean():.2f} px  (image is {img_size}x{img_size})")


# ---------------------------------------------------------------------------
# Using real cephalometric data instead of the synthetic task
# ---------------------------------------------------------------------------
def load_real_dataset(args, generator):
    """Load a real ``(image, landmarks)`` dataset and produce train/test splits.

    Uses :class:`~deltaflow.datasets.ISBI2015CephalometricDataset`, which
    accepts either per-image ``.txt`` annotations or a consolidated ``.csv``
    split file. Landmarks come back already rescaled to ``args.image_size``
    and normalised to ``[-1, 1]`` by the base
    :class:`~deltaflow.datasets.RadiographDataset`, so nothing downstream
    (``train``/``sample_landmarks``/``visualise``) needs to change.

    If ``--test-landmarks-root`` is given, it is loaded as an explicit test
    split (the standard protocol for benchmarks that ship separate train and
    test annotation files). Otherwise a fraction ``--val-fraction`` of the
    training annotations is held out at random.
    """
    from deltaflow.datasets import ISBI2015CephalometricDataset

    def _load(landmarks_file):
        ds = ISBI2015CephalometricDataset(
            root=args.data_root,
            image_size=args.image_size,
            landmarks_file=landmarks_file,
            landmarks_file_2=args.landmarks_root_2,
            n_landmarks=args.n_landmarks,
        )
        if ds.landmarks is None or len(ds) == 0:
            raise ValueError(f"No annotated images found for {landmarks_file!r}.")
        images = torch.stack([ds[i][0] for i in range(len(ds))])
        landmarks = torch.stack([ds[i][1] for i in range(len(ds))])
        return images, landmarks

    images, landmarks = _load(args.landmarks_root)

    if args.test_landmarks_root:
        test_images, test_landmarks = _load(args.test_landmarks_root)
        return images, landmarks, test_images, test_landmarks

    n = images.shape[0]
    n_test = max(1, int(round(n * args.val_fraction)))
    perm = torch.randperm(n, generator=generator)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return images[train_idx], landmarks[train_idx], images[test_idx], landmarks[test_idx]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Directory of real radiograph images (searched recursively). "
        "If omitted, the synthetic blob dataset is used instead.",
    )
    p.add_argument(
        "--landmarks-root",
        type=str,
        default=None,
        help="Landmark annotations for training: either a directory of per-image "
        ".txt files (e.g. ISBI2015 400_senior) or a consolidated .csv split file "
        "(e.g. train_senior.csv). Required if --data-root is given.",
    )
    p.add_argument(
        "--test-landmarks-root",
        type=str,
        default=None,
        help="Optional separate test-split annotations (directory or .csv). If given, "
        "used directly as the test set instead of holding out --val-fraction of train.",
    )
    p.add_argument(
        "--landmarks-root-2",
        type=str,
        default=None,
        help="Optional second rater's .txt annotation directory (e.g. ISBI2015 400_junior); "
        "if given, landmarks are averaged across both raters (per-image .txt layout only).",
    )
    p.add_argument(
        "--n-landmarks", type=int, default=19, help="Landmarks per image in the real dataset."
    )
    p.add_argument("--image-size", type=int, default=IMG_SIZE, help="Square resize resolution.")
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of the training annotations held out for testing when "
        "--test-landmarks-root is not given.",
    )
    args = p.parse_args()
    if args.data_root and not args.landmarks_root:
        p.error("--landmarks-root is required when --data-root is given.")
    return args


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path("outputs/landmark_detection")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_root:
        print(f"Loading real radiograph dataset from {args.data_root} ...")
        images, landmarks, test_images, test_landmarks = load_real_dataset(args, gen)
        n_landmarks = args.n_landmarks
        print(
            f"  {images.shape[0]} train / {test_images.shape[0]} test images, {n_landmarks} landmarks each"
        )
    else:
        print("Building synthetic X-ray-like dataset...")
        images, landmarks = make_dataset(N_TRAIN, gen)
        test_images, test_landmarks = make_dataset(64, gen)
        n_landmarks = N_LANDMARKS

    print("Training image-conditioned landmark flow...")
    model = LandmarkVelocityField(data_dim=2 * n_landmarks).to(device)
    train(model, images, landmarks, gen, device, test_images, test_landmarks)

    print("Evaluating and visualising...")
    report_error(model, test_images, test_landmarks, device, gen, n_landmarks)
    visualise(model, test_images, test_landmarks, out_dir, device, gen, n_landmarks)
    print("Done.")


if __name__ == "__main__":
    main()
