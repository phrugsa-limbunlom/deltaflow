"""90-showcase/08-guidance-alignment-pretraining: CDPM-style representation
learning with a guidance-aligned image flow.

This is the *representation-learning* half of the DeltaFlow story, the
counterpart to the landmark *detector* in ``07-landmark-detection``. It
reproduces, at toy scale, the pretraining phase of CDPM-Align (Multi-Scale
Guidance-Aligned Diffusion Pretraining for few-shot anatomical landmark
detection). The key design point, and the answer to "condition on the image
or on the landmark?", is that neither the image nor the landmark is the
generative *target* here:

* The generative target is the **image** itself. A flow-matching model learns
  to transport noise onto images, i.e. it models ``p(image | y)`` where ``y``
  is a **dataset/class label** (the "which dataset did this come from" index),
  not a landmark. Landmark labels are not used at all in this phase, this is
  self-supervised generative representation learning.
* Classifier-free guidance gives, at every hierarchy level ``l``, a guidance
  difference ``delta_h_l = h_cond_l - h_uncond_l`` (features with the label
  minus features without it). This isolates *what the conditioning changed*
  and largely cancels the shared anatomy.
* The **delta-alignment** loss (the "Delta" in DeltaFlow) enforces that
  ``delta_h`` is consistent across two independent noise levels of the same
  image, pushing the encoder toward a guidance representation invariant to
  noise level. See :class:`~deltaflow.losses.delta_alignment.DeltaAlignmentLoss`
  and :class:`~deltaflow.models.projector.MultiScaleProjector`.

After pretraining, we *freeze* the backbone and read out its features with a
tiny linear probe to check that the learned representation has become
class/anatomy-discriminative (compared against a random-init backbone). That
readout is the stand-in for the downstream landmark head, a good generative
representation is what makes few-shot detection work.

The task is synthetic (four "datasets", each a distinct oriented-texture or
ring pattern standing in for a distinct anatomy) so the script runs end to end
on CPU with no download.

Note on the alignment term: the two "views" here are two independent noise
levels of the same image (as in the CDPM paper), and the alignment loss is a
light *consistency* regularizer on ``delta_h``. On this easy synthetic task it
sits near zero (the guidance difference is already noise-level-consistent), the
flow objective does the representational heavy lifting. Its role grows on
harder, more heterogeneous multi-dataset data. The point of this example is to
wire the pieces together correctly and show that guided generative pretraining
yields a class/anatomy-discriminative representation.

Run::

    python examples/90-showcase/08-guidance-alignment-pretraining/main.py

Outputs are written to ``outputs/guidance_alignment_pretraining/``.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - install hint only
    raise SystemExit(
        "This example needs matplotlib. Install it with:\n    pip install matplotlib"
    ) from exc

from deltaflow.interpolants import LinearInterpolant
from deltaflow.losses.delta_alignment import DeltaAlignmentLoss
from deltaflow.models.projector import MultiScaleProjector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE = 32
N_CLASSES = 4  # number of "datasets" (distinct synthetic anatomies)
N_PER_CLASS_TRAIN = 120
N_PER_CLASS_TEST = 40
BASE_CH = 16
EMB_DIM = 64
N_STEPS = 300
BATCH = 32
LR = 2e-3
LAMBDA_ALIGN = 5.0  # weight of the delta-alignment term
T_LOW, T_HIGH = 0.25, 0.75  # mid-range timestep bias (paper's [T/4, 3T/4])
PROBE_STEPS = 300
SEED = 0

# Non-primary companion palette (never the docs' blue primary).
CLASS_COLORS = ["#3f9e73", "#7a5cc0", "#e0b13c", "#0f9bab"]  # green, purple, yellow, teal


# ---------------------------------------------------------------------------
# Synthetic multi-"dataset" images: each class is a distinct texture
# ---------------------------------------------------------------------------
def make_dataset(n_per_class: int, generator: torch.Generator):
    """Return ``(images, labels)`` with per-class distinct textures.

    ``images``: ``(N, 1, IMG_SIZE, IMG_SIZE)`` in ``[-1, 1]``.
    ``labels``: ``(N,)`` integer class ids in ``[0, N_CLASSES)``.
    """
    ys = torch.linspace(-1, 1, IMG_SIZE).view(IMG_SIZE, 1)
    xs = torch.linspace(-1, 1, IMG_SIZE).view(1, IMG_SIZE)
    imgs, labels = [], []
    for c in range(N_CLASSES):
        for _ in range(n_per_class):
            freq = 3.0 + 4.0 * torch.rand(1, generator=generator).item()
            phase = 2 * math.pi * torch.rand(1, generator=generator).item()
            if c == 0:  # vertical gratings
                field = torch.sin(freq * math.pi * xs + phase).expand(IMG_SIZE, IMG_SIZE)
            elif c == 1:  # horizontal gratings
                field = torch.sin(freq * math.pi * ys + phase).expand(IMG_SIZE, IMG_SIZE)
            elif c == 2:  # diagonal gratings
                field = torch.sin(freq * math.pi * (xs + ys) / 1.4 + phase)
            else:  # concentric rings
                r = torch.sqrt(xs**2 + ys**2)
                field = torch.sin(freq * math.pi * r + phase)
            img = field + 0.15 * torch.randn(IMG_SIZE, IMG_SIZE, generator=generator)
            imgs.append(img.clamp(-1, 1).unsqueeze(0))
            labels.append(c)
    images = torch.stack(imgs)
    labels = torch.tensor(labels, dtype=torch.long)
    perm = torch.randperm(images.shape[0], generator=generator)
    return images[perm], labels[perm]


# ---------------------------------------------------------------------------
# Conditional UNet velocity field that also exposes multi-scale features
# ---------------------------------------------------------------------------
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of a continuous ``t in [0, 1]``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().view(-1, 1) * freqs.view(1, -1) * 1000.0
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(groups, cout)
        self.emb = nn.Linear(emb_dim, cout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.conv(x))
        h = h + self.emb(emb)[:, :, None, None]
        return self.act(h)


class GuidedUNet(nn.Module):
    """Small conditional UNet velocity field with a null class for CFG.

    ``forward(x, t, y)`` returns ``(velocity, feats)`` where ``feats`` holds
    the four hierarchy levels the delta-alignment projector consumes. Passing
    ``y = N_CLASSES`` selects the learned *null* embedding (the unconditional
    pass required to form ``delta_h``).
    """

    NULL_CLASS = N_CLASSES

    def __init__(self, base: int = BASE_CH, emb_dim: int = EMB_DIM):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.class_emb = nn.Embedding(N_CLASSES + 1, emb_dim)

        self.stem = nn.Conv2d(1, base, 3, padding=1)
        self.enc1 = ConvBlock(base, base, emb_dim)  # 32
        self.down1 = nn.Conv2d(base, base, 4, 2, 1)  # -> 16
        self.enc2 = ConvBlock(base, 2 * base, emb_dim)  # 16
        self.down2 = nn.Conv2d(2 * base, 2 * base, 4, 2, 1)  # -> 8
        self.enc3 = ConvBlock(2 * base, 2 * base, emb_dim)  # 8  (enc_1_4)
        self.down3 = nn.Conv2d(2 * base, 4 * base, 4, 2, 1)  # -> 4
        self.enc4 = ConvBlock(4 * base, 4 * base, emb_dim)  # 4  (enc_1_8)
        self.bottleneck = ConvBlock(4 * base, 4 * base, emb_dim)  # 4 (bottleneck)
        self.dec8 = ConvBlock(8 * base, 2 * base, emb_dim)  # 4  (dec_1_8)
        self.up1 = nn.ConvTranspose2d(2 * base, 2 * base, 4, 2, 1)  # -> 8
        self.dec_u1 = ConvBlock(4 * base, 2 * base, emb_dim)
        self.up2 = nn.ConvTranspose2d(2 * base, base, 4, 2, 1)  # -> 16
        self.dec_u2 = ConvBlock(3 * base, base, emb_dim)
        self.up3 = nn.ConvTranspose2d(base, base, 4, 2, 1)  # -> 32
        self.dec_u3 = ConvBlock(2 * base, base, emb_dim)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    @property
    def feature_dims(self):
        b = BASE_CH
        return {"enc_1_4": 2 * b, "enc_1_8": 4 * b, "bottleneck": 4 * b, "dec_1_8": 2 * b}

    def forward(self, x, t, y):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device)
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        emb = self.time_mlp(timestep_embedding(t, self.emb_dim)) + self.class_emb(y)

        h0 = self.stem(x)
        h1 = self.enc1(h0, emb)
        h2 = self.enc2(self.down1(h1), emb)
        h3 = self.enc3(self.down2(h2), emb)  # 8, enc_1_4
        h4 = self.enc4(self.down3(h3), emb)  # 4, enc_1_8
        b = self.bottleneck(h4, emb)  # 4, bottleneck
        d8 = self.dec8(torch.cat([b, h4], dim=1), emb)  # 4, dec_1_8
        u1 = self.dec_u1(torch.cat([self.up1(d8), h3], dim=1), emb)  # 8
        u2 = self.dec_u2(torch.cat([self.up2(u1), h2], dim=1), emb)  # 16
        u3 = self.dec_u3(torch.cat([self.up3(u2), h1], dim=1), emb)  # 32
        v = self.out(u3)
        feats = {"enc_1_4": h3, "enc_1_8": h4, "bottleneck": b, "dec_1_8": d8}
        return v, feats


# ---------------------------------------------------------------------------
# CDPM-style guidance-aligned pretraining
# ---------------------------------------------------------------------------
def pretrain(unet, images, labels, generator, device):
    interp = LinearInterpolant()
    projector = MultiScaleProjector(feature_dims=unet.feature_dims, hidden_dim=128, out_dim=64).to(
        device
    )
    loss_fn = DeltaAlignmentLoss(projector, lambda_flow=1.0, lambda_align=LAMBDA_ALIGN)
    params = list(unet.parameters()) + list(projector.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    n = images.shape[0]
    null = torch.full((BATCH,), GuidedUNet.NULL_CLASS, device=device, dtype=torch.long)

    unet.train()
    for step in range(N_STEPS):
        idx = torch.randint(0, n, (BATCH,), generator=generator)
        x1 = images[idx].to(device)
        y = labels[idx].to(device)

        # Two independent noise levels (views) of the same images, biased to
        # the mid-range where the guidance difference is most informative.
        outs = []
        for _ in range(2):
            t = T_LOW + (T_HIGH - T_LOW) * torch.rand(BATCH, device=device)
            x0 = torch.randn_like(x1)
            x_t, target_v = interp.interpolate(x1, t, x0=x0)
            v_c, feats_c = unet(x_t, t, y)
            v_u, feats_u = unet(x_t, t, null)
            outs.append((v_c, v_u, target_v, feats_u, feats_c))

        (v_c1, v_u1, tv1, fu1, fc1), (v_c2, v_u2, tv2, fu2, fc2) = outs
        total, parts = loss_fn(v_c1, v_u1, tv1, v_c2, v_u2, tv2, fu1, fc1, fu2, fc2)
        opt.zero_grad()
        total.backward()
        opt.step()

        if step % 50 == 0 or step == N_STEPS - 1:
            print(
                f"  step {step:4d}/{N_STEPS}  total {parts['loss_total']:.4f}  "
                f"flow {parts['loss_flow']:.4f}  align {parts['loss_align']:.4f}"
            )
    unet.eval()
    return projector


# ---------------------------------------------------------------------------
# Frozen-feature readout (stand-in for the downstream landmark head)
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(unet, images, device, seed=7, batch=128):
    """Global-average-pooled bottleneck features from an unconditional pass.

    Uses a fixed mid-range timestep and fixed noise so the representation is
    deterministic. This is the image-only ``h_uncond`` feature the downstream
    head would build on.
    """
    unet.eval()
    g = torch.Generator().manual_seed(seed)
    interp = LinearInterpolant()
    feats = []
    n = images.shape[0]
    for start in range(0, n, batch):
        x1 = images[start : start + batch].to(device)
        bsz = x1.shape[0]
        t = torch.full((bsz,), 0.5, device=device)
        x0 = torch.randn(x1.shape, generator=g).to(device)
        x_t, _ = interp.interpolate(x1, t, x0=x0)
        null = torch.full((bsz,), GuidedUNet.NULL_CLASS, device=device, dtype=torch.long)
        _, f = unet(x_t, t, null)
        feats.append(f["bottleneck"].mean(dim=(2, 3)).cpu())
    return torch.cat(feats)


def linear_probe(train_x, train_y, test_x, test_y, n_classes, device, steps=PROBE_STEPS):
    """Train a linear classifier on frozen features; return test accuracy."""
    mean, std = train_x.mean(0, keepdim=True), train_x.std(0, keepdim=True) + 1e-6
    tr = ((train_x - mean) / std).to(device)
    te = ((test_x - mean) / std).to(device)
    clf = nn.Linear(tr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    tyd, teyd = train_y.to(device), test_y.to(device)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(clf(tr), tyd).backward()
        opt.step()
    with torch.no_grad():
        acc = (clf(te).argmax(1) == teyd).float().mean().item()
    return acc


def visualise(feats, labels, out_dir, title):
    """2D PCA scatter of frozen features, coloured by class."""
    x = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    x = x - x.mean(0)
    _, _, v = torch.pca_lowrank(x, q=2)
    proj = (x @ v[:, :2]).numpy()

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for c in range(N_CLASSES):
        m = labels.numpy() == c
        ax.scatter(
            proj[m, 0],
            proj[m, 1],
            s=18,
            color=CLASS_COLORS[c],
            alpha=0.75,
            edgecolors="none",
            label=f"dataset {c}",
        )
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    out_path = out_dir / "guidance_alignment_features.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path("outputs/guidance_alignment_pretraining")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building synthetic multi-dataset images...")
    images, labels = make_dataset(N_PER_CLASS_TRAIN, gen)
    test_images, test_labels = make_dataset(N_PER_CLASS_TEST, gen)

    # Baseline: probe a random-init backbone, to show what pretraining adds.
    print("Probing a RANDOM-INIT backbone (baseline)...")
    rand_unet = GuidedUNet().to(device)
    base_tr = extract_features(rand_unet, images, device)
    base_te = extract_features(rand_unet, test_images, device)
    base_acc = linear_probe(base_tr, labels, base_te, test_labels, N_CLASSES, device)
    print(f"  random-init linear-probe accuracy: {base_acc:.3f}")

    print("Guidance-aligned generative pretraining (flow + delta-alignment)...")
    unet = GuidedUNet().to(device)
    pretrain(unet, images, labels, gen, device)

    print("Probing the PRETRAINED backbone...")
    tr_feats = extract_features(unet, images, device)
    te_feats = extract_features(unet, test_images, device)
    acc = linear_probe(tr_feats, labels, te_feats, test_labels, N_CLASSES, device)
    print(f"  pretrained linear-probe accuracy:  {acc:.3f}  (chance = {1 / N_CLASSES:.3f})")
    print(f"  improvement over random init:      {acc - base_acc:+.3f}")

    print("Visualising learned representation...")
    visualise(te_feats, test_labels, out_dir, "Pretrained backbone features (PCA)")
    print("Done.")


if __name__ == "__main__":
    main()
