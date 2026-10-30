"""DiT tensor walkthrough, a visual trace of one forward pass.

Trains a small three-channel `DiT` velocity field on an RGB variant of the 8x8
toy from ``20-training/03-conditional-dit`` (2x2 patches give a readable
16-token sequence), then replays ``DiT.forward`` step by step with the
model's own submodules. The replay is checked against ``field(x, t, y)`` so every
figure shows the tensors the real forward pass computes.

The point being made is the split between the two paths in a DiT. The image
becomes a sequence of N patch tokens, while time and label collapse into one
conditioning vector ``c`` per image. The two never concatenate. ``c`` only emits
per-channel ``(shift, scale, gate)`` vectors (adaLN-Zero) that are broadcast to
all tokens, and attention is the only operation that mixes tokens.

Outputs (written to ``outputs/dit_tensor_viz/``)::

    patchify.png      x_t -> patches -> token matrix (B, N, D), + pos_embed
    conditioning.png  t and y -> sinusoids, embeddings, c = t_emb + y_emb
    adaln.png         per-block (shift, scale, gate) emitted from c, and how y moves them
    attention.png     head-averaged attention maps and one query on the image
    unpatchify.png    tokens -> patch pixels -> velocity, versus target x1 - x0
    cfg.png           velocity and endpoint estimate from pure noise under
                      y=0, y=1, the null token, and CFG
    dit_trace.json    every intermediate tensor for the interactive docs
                      walkthrough (copy to docs/assets/dit-walkthrough/)

Run::

    python examples/90-showcase/10-dit-tensor-viz/main.py

References:
    Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023).
    Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
except ImportError as exc:  # pragma: no cover - install hint only
    raise SystemExit(
        "This example needs matplotlib. Install it with:\n    pip install matplotlib"
    ) from exc

from deltaflow.losses import ConditionalFlowMatchingLoss
from deltaflow.models import DiT
from deltaflow.models.dit import modulate

IMG = 8
CHANNELS = 3
PATCH = 2
HIDDEN = 64
DEPTH = 4
HEADS = 4
N_CLASSES = 2
T_VIEW = 0.5

# Palette, never the docs primary blue.
C_GREEN = "#3f9e73"
C_PURPLE = "#7a5cc0"
C_YELLOW = "#e0b13c"
C_TEAL = "#0f9bab"
C_NEUTRAL = "#d6dbe6"
SEQ = "viridis"
DIV = LinearSegmentedColormap.from_list("df_div", [C_PURPLE, "#f4f5f8", C_GREEN])


# --------------------------------------------------------------------------- #
# Data and training
# --------------------------------------------------------------------------- #


def class_images(y: torch.Tensor, noise: float = 0.05) -> torch.Tensor:
    """Class 0 is green above purple, class 1 swaps the two RGB halves."""
    green = torch.tensor([0x3F, 0x9E, 0x73], dtype=torch.float32) / 255
    purple = torch.tensor([0x7A, 0x5C, 0xC0], dtype=torch.float32) / 255
    x = purple[None, :, None, None].expand(y.shape[0], CHANNELS, IMG, IMG).clone()
    half = IMG // 2
    x[y == 0, :, :half, :] = green[:, None, None]
    x[y == 1, :, half:, :] = green[:, None, None]
    return x + noise * torch.randn_like(x)


def train(field: DiT, steps: int = 400, batch: int = 64) -> None:
    loss_fn = ConditionalFlowMatchingLoss()
    opt = torch.optim.Adam(field.parameters(), lr=2e-3)
    field.train()
    for step in range(steps):
        y = torch.randint(0, N_CLASSES, (batch,))
        loss = loss_fn(field, class_images(y), y=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"  step={step:4d}  loss={loss.item():.4f}")
    field.eval()


# --------------------------------------------------------------------------- #
# Step-by-step replay of DiT.forward
# --------------------------------------------------------------------------- #


def attention_weights(attn, x: torch.Tensor) -> torch.Tensor:
    """Recompute softmax(q k^T / sqrt(d)), which SDPA keeps internal. Shape (B, heads, N, N)."""
    B, N, _ = x.shape
    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    return torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim), dim=-1)


@torch.no_grad()
def trace(field: DiT, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> dict:
    """Mirror ``DiT.forward`` and keep every intermediate tensor."""
    s: dict = {"x": x}
    s["patch_conv"] = field.patch_embed(x)  # (B, D, n, n)
    s["flat"] = s["patch_conv"].flatten(2)  # (B, D, N)
    s["tokens"] = s["flat"].transpose(1, 2)  # (B, N, D)
    h = s["tokens"] + field.pos_embed
    s["tokens_pos"] = h

    s["sinusoid"] = field.t_embedder._sinusoidal(t * field.t_embedder.time_scale)
    s["t_emb"] = field.t_embedder(t)
    s["y_emb"] = field.y_embedder(y, train=False)
    c = s["t_emb"] + s["y_emb"]
    s["c"] = c

    s["mod"], s["attn"], s["blocks"], s["block_states"] = [], [], [], []
    for blk in field.blocks:
        mod = blk.adaLN_modulation(c)  # (B, 6D)
        s["mod"].append(mod)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        a_in = modulate(blk.norm1(h), shift_msa, scale_msa)
        s["attn"].append(attention_weights(blk.attn, a_in))
        h = h + gate_msa.unsqueeze(1) * blk.attn(a_in)
        after_attention = h
        m_in = modulate(blk.norm2(h), shift_mlp, scale_mlp)
        m_hidden = blk.mlp[1](blk.mlp[0](m_in))
        h = h + gate_mlp.unsqueeze(1) * blk.mlp[2](m_hidden)
        s["block_states"].append(
            {
                "attention_input": a_in,
                "after_attention": after_attention,
                "mlp_input": m_in,
                "mlp_hidden": m_hidden,
                "output": h,
            }
        )
        s["blocks"].append(h)

    shift, scale = field.final_layer.adaLN_modulation(c).chunk(2, dim=-1)
    s["final_input"] = modulate(field.final_layer.norm_final(h), shift, scale)
    s["final"] = field.final_layer.linear(s["final_input"])  # (B, N, p*p*C)
    s["v"] = field._unpatchify(s["final"])  # (B, C, H, W)

    ref = field(x, t, y=y)
    err = (s["v"] - ref).abs().max().item()
    assert err < 1e-5, f"replay diverged from DiT.forward (max |diff| = {err:.2e})"
    return s


def print_shapes(s: dict) -> None:
    rows = [
        ("x_t", s["x"]),
        ("patch_embed(x)", s["patch_conv"]),
        (".flatten(2)", s["flat"]),
        (".transpose(1, 2)", s["tokens"]),
        ("+ pos_embed", s["tokens_pos"]),
        ("t_embedder(t)", s["t_emb"]),
        ("y_embedder(y)", s["y_emb"]),
        ("c = t_emb + y_emb", s["c"]),
        ("adaLN_modulation(c)", s["mod"][0]),
        ("attention weights", s["attn"][0]),
        ("block output", s["blocks"][-1]),
        ("final_layer(h, c)", s["final"]),
        ("_unpatchify -> v", s["v"]),
    ]
    print("[demo] tensor shapes along the forward pass")
    for name, tensor in rows:
        print(f"  {name:<22} {tuple(tensor.shape)}")


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #


def _clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def _sym(a: np.ndarray) -> float:
    return float(np.abs(a).max()) or 1.0


def _rgb(a, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Display CHW tensors as RGB, clipping only the display, not the trace."""
    return np.clip((np.moveaxis(np.asarray(a), 0, -1) - lo) / (hi - lo), 0, 1)


def _grid_lines(ax, n: int, p: int, color: str = "white") -> None:
    for k in range(1, n):
        ax.axhline(k * p - 0.5, color=color, lw=1.0)
        ax.axvline(k * p - 0.5, color=color, lw=1.0)


def plot_patchify(s: dict, out: Path) -> Path:
    x = _rgb(s["x"][0])
    n = IMG // PATCH
    tokens = s["tokens"][0].numpy()
    tokens_pos = s["tokens_pos"][0].numpy()

    fig = plt.figure(figsize=(16, 4.6), dpi=130, layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.6, 1.5, 1.5])

    ax = fig.add_subplot(gs[0])
    ax.imshow(x)
    _grid_lines(ax, n, PATCH)
    for i in range(n):
        for j in range(n):
            ax.text(
                j * PATCH + PATCH / 2 - 0.5,
                i * PATCH + PATCH / 2 - 0.5,
                str(i * n + j),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                weight="bold",
            )
    ax.set_title(
        f"RGB x_t  ({CHANNELS}, {IMG}, {IMG})\n{n}x{n} grid of {PATCH}x{PATCH} patches",
        fontsize=9,
    )
    _clean(ax)

    ax = fig.add_subplot(gs[1])
    per_row = n * n // 2
    gap = PATCH + 1
    strip = np.ones((2 * gap, per_row * gap - 1, CHANNELS))
    for k in range(n * n):
        i, j = divmod(k, n)
        r, col = divmod(k, per_row)
        strip[r * (gap + 1) : r * (gap + 1) + PATCH, col * gap : col * gap + PATCH] = x[
            i * PATCH : (i + 1) * PATCH, j * PATCH : (j + 1) * PATCH
        ]
        ax.text(
            col * gap + PATCH / 2 - 0.5,
            r * (gap + 1) - 0.75,
            str(k),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.imshow(strip)
    ax.set_ylim(strip.shape[0] - 0.5, -1.6)
    ax.axis("off")
    ax.set_title(
        f"row-major patch order, N = ({IMG}/{PATCH})^2 = {n * n} tokens\n"
        f"each patch is {PATCH}x{PATCH}x{CHANNELS} = {PATCH * PATCH * CHANNELS} values",
        fontsize=9,
    )

    lim = _sym(tokens_pos)
    for idx, (mat, title) in enumerate(
        [
            (tokens, f"patch_embed -> flatten -> transpose\nh  (N, D) = ({n * n}, {HIDDEN})"),
            (tokens_pos, f"h + pos_embed  ({n * n}, {HIDDEN})\nposition is now part of each token"),
        ]
    ):
        ax = fig.add_subplot(gs[2 + idx])
        im = ax.imshow(mat, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xlabel(f"channel d (0..{HIDDEN - 1})", fontsize=8)
        ax.set_ylabel("token n", fontsize=8)
        ax.set_yticks(range(0, n * n, 3))
        ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        "Image path. Conv2d(k=s=p) is a per-patch linear map, so one patch "
        "becomes one D-dim token",
        fontsize=11,
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_conditioning(field: DiT, s: dict, out: Path) -> Path:
    sinus = s["sinusoid"][0].numpy()
    table = field.y_embedder.embedding_table.weight.detach().numpy()
    rows = np.stack([s["t_emb"][0].numpy(), s["c"][0].numpy(), s["c"][1].numpy()])
    labels = [f"t_emb (t={T_VIEW:g})", "c (y=0)", "c (y=1)"]
    diff = (s["c"][1] - s["c"][0]).numpy()[None]
    ratio = np.linalg.norm(diff) / np.linalg.norm(rows[0])

    fig = plt.figure(figsize=(17, 4.6), dpi=130, layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.8], height_ratios=[3, 1.3])

    ax = fig.add_subplot(gs[:, 0])
    half = sinus.shape[0] // 2
    ax.plot(sinus[:half], color=C_GREEN, lw=1.0, label="cos")
    ax.plot(sinus[half:], color=C_PURPLE, lw=1.0, label="sin")
    ax.set_xlabel("frequency index (high to low)", fontsize=8)
    ax.set_title(
        f"sinusoids of t x {field.t_embedder.time_scale:g}  ({sinus.shape[0]},)\n"
        f"then MLP {sinus.shape[0]} -> D -> D gives t_emb",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[:, 1])
    lim = _sym(table)
    im = ax.imshow(table, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([f"class {k}" for k in range(N_CLASSES)] + ["null (CFG)"], fontsize=8)
    ax.set_xlabel("channel d", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"embedding_table  ({table.shape[0]}, {HIDDEN})\ny_emb is a row lookup", fontsize=9
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = fig.add_subplot(gs[0, 2])
    lim = _sym(rows)
    im = ax.imshow(rows, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks([])
    ax.set_title(f"c = t_emb + y_emb, one vector ({HIDDEN},) per image", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = fig.add_subplot(gs[1, 2])
    lim = _sym(diff)
    im = ax.imshow(diff, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks([0])
    ax.set_yticklabels(["c(y=1) - c(y=0)"], fontsize=8)
    ax.set_xlabel("channel d", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"the label's entire contribution (own colour scale, " f"|diff| / |t_emb| = {ratio:.2f})",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        "Conditioning path. Time and label are summed into a single vector c "
        "(no tokens, no concatenation)",
        fontsize=11,
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_adaln(s: dict, out: Path) -> Path:
    names = ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]
    mods = [m.reshape(m.shape[0], 6, HIDDEN).numpy() for m in s["mod"]]
    base = [m[0] for m in mods]
    diff = [m[1] - m[0] for m in mods]
    lim_b = max(_sym(m) for m in base)
    lim_d = max(_sym(m) for m in diff)

    fig, axes = plt.subplots(
        DEPTH,
        2,
        figsize=(12, 1.55 * DEPTH + 1.2),
        dpi=130,
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    for b in range(DEPTH):
        for col, (mat, lim) in enumerate([(base[b], lim_b), (diff[b], lim_d)]):
            ax = axes[b, col]
            im = ax.imshow(mat, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
            ax.set_yticks(range(6))
            ax.set_yticklabels(names, fontsize=6)
            ax.tick_params(axis="x", labelsize=7)
            if col == 0:
                ax.set_ylabel(f"block {b}", fontsize=8)
            if b == DEPTH - 1:
                ax.set_xlabel("channel d", fontsize=8)
                fig.colorbar(im, ax=axes[:, col], fraction=0.05, pad=0.02)
    axes[0, 0].set_title("modulation for y = 0", fontsize=10, color=C_TEAL, weight="bold")
    axes[0, 1].set_title(
        "change when y = 0 -> 1 (own colour scale)", fontsize=10, color=C_PURPLE, weight="bold"
    )
    fig.suptitle(
        "adaLN-Zero. Each block maps c -> (6, D) via SiLU + Linear (zero-initialised).\n"
        "Every row is one (D,) vector broadcast over all tokens as (B, 1, D)",
        fontsize=10,
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_attention(s: dict, out: Path, query: int = 5) -> Path:
    n = IMG // PATCH
    x = _rgb(s["x"][0])
    maps = [a[0].mean(0).numpy() for a in s["attn"]]  # head average, (N, N)
    vmax = max(m.max() for m in maps)

    fig, axes = plt.subplots(2, DEPTH, figsize=(3.3 * DEPTH, 6.8), dpi=130, layout="constrained")
    for b in range(DEPTH):
        ax = axes[0, b]
        im = ax.imshow(maps[b], cmap=SEQ, vmin=0, vmax=vmax)
        ax.axhline(query, color=C_YELLOW, lw=1.0)
        ax.set_title(f"block {b}, mean over {HEADS} heads\n(N, N) = ({n * n}, {n * n})", fontsize=9)
        ax.set_xlabel("key token", fontsize=8)
        ax.set_ylabel("query token", fontsize=8)
        ax.tick_params(labelsize=7)

        ax = axes[1, b]
        ax.imshow(x)
        heat = np.kron(maps[b][query].reshape(n, n), np.ones((PATCH, PATCH)))
        ax.imshow(heat, cmap=SEQ, alpha=0.65)
        qi, qj = divmod(query, n)
        ax.add_patch(
            Rectangle(
                (qj * PATCH - 0.5, qi * PATCH - 0.5), PATCH, PATCH, fill=False, ec=C_YELLOW, lw=2.0
            )
        )
        _grid_lines(ax, n, PATCH, color=C_NEUTRAL)
        ax.set_title(f"where token {query} looks (block {b})", fontsize=9)
        _clean(ax)
    fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.02)
    fig.suptitle(
        "Self-attention is the only step that mixes tokens. "
        "Each row is one query's softmax over all keys",
        fontsize=11,
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_unpatchify(s: dict, x0: torch.Tensor, x1: torch.Tensor, out: Path) -> Path:
    final = s["final"][0].numpy()
    v = s["v"][0].numpy()
    delta = (x1 - x0)[0].numpy()
    x_t = s["x"][0].numpy()
    x1_hat = x_t + (1 - T_VIEW) * v
    n = IMG // PATCH
    lim = max(_sym(v), _sym(delta))

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(17, 3.9),
        dpi=130,
        layout="constrained",
        gridspec_kw={"width_ratios": [1.3, 1, 1, 1, 1]},
    )
    ax = axes[0]
    im = ax.imshow(final, cmap=DIV, vmin=-_sym(final), vmax=_sym(final), aspect="auto")
    ax.set_xlabel(f"p*p*C = {PATCH * PATCH * CHANNELS} values (pixel, then R/G/B)", fontsize=8)
    ax.set_xticks(range(0, PATCH * PATCH * CHANNELS, CHANNELS))
    ax.set_ylabel("token n", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"final_layer(h, c)  ({n * n}, {PATCH * PATCH * CHANNELS})\neach token -> its own RGB patch",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    panels = [
        (
            _rgb(v, -lim, lim),
            True,
            f"v = unpatchify  ({CHANNELS}, {IMG}, {IMG})\nRGB, zero velocity is mid-gray",
        ),
        (_rgb(delta, -lim, lim), False, "target  x1 - x0\n(same signed RGB scale)"),
        (_rgb(x_t), False, f"RGB x_t  at t = {T_VIEW:g}"),
        (_rgb(x1_hat), False, "x_t + (1 - t) v\n(RGB endpoint estimate)"),
    ]
    for ax, (img, grid, title) in zip(axes[1:], panels):
        ax.imshow(img)
        if grid:
            _grid_lines(ax, n, PATCH, color=C_NEUTRAL)
        ax.set_title(title, fontsize=9)
        _clean(ax)
    fig.suptitle(
        "Back to pixels. The velocity has the shape of x and regresses the " "displacement x1 - x0",
        fontsize=11,
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


@torch.no_grad()
def plot_cfg(field: DiT, x_t: torch.Tensor, t: torch.Tensor, out: Path, scale: float = 2.0) -> Path:
    null = torch.full((1,), field.y_embedder.null_index)
    cases = [
        ("y = 0", field(x_t, t, y=torch.tensor([0]))),
        ("y = 1", field(x_t, t, y=torch.tensor([1]))),
        ("null token", field(x_t, t, y=null)),
        (f"CFG y = 1, s = {scale:g}", field.forward_with_cfg(x_t, t, torch.tensor([1]), scale)),
    ]
    lim = max(_sym(v.numpy()) for _, v in cases)
    t_val = t.item()

    fig, axes = plt.subplots(
        2, len(cases) + 1, figsize=(3.0 * (len(cases) + 1), 6.2), dpi=130, layout="constrained"
    )
    axes[0, 0].imshow(_rgb(x_t[0]))
    axes[0, 0].set_title(f"same x_t (t = {t_val:g}, pure noise)", fontsize=9)
    _clean(axes[0, 0])
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.0,
        0.5,
        "top row\nsigned RGB velocity\nzero is mid-gray\n\nbottom row\nRGB endpoint",
        fontsize=9,
        va="center",
    )
    for k, (name, v) in enumerate(cases, start=1):
        axes[0, k].imshow(_rgb(v[0], -lim, lim))
        axes[0, k].set_title(name, fontsize=9)
        axes[1, k].imshow(_rgb((x_t + (1 - t_val) * v)[0]))
        _clean(axes[0, k])
        _clean(axes[1, k])
    fig.suptitle(
        "The label only changes c, yet it redirects the whole velocity field. "
        "CFG extrapolates v_null + s (v_cond - v_null)",
        fontsize=11,
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Interactive-walkthrough export
# --------------------------------------------------------------------------- #

EXPORT_T = (0.0, 0.5, 0.9)
EXPORT_CFG = (0.0, 1.0, 2.0, 4.0)
SAMPLE_STEPS = 20


def _r(a, nd: int = 3):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def _image(a: torch.Tensor, nd: int = 3) -> list:
    """Export HWC RGB values for canvas rendering without clipping."""
    return _r(a.permute(1, 2, 0), nd)


@torch.no_grad()
def _sample(field: DiT, x0: torch.Tensor, label: int, scale: float) -> list:
    """Euler trajectory from ``x0`` with CFG scale ``scale`` (``0`` = null token only)."""
    null = torch.full((1,), field.y_embedder.null_index)
    x, frames = x0.clone(), [x0[0]]
    for i in range(SAMPLE_STEPS):
        t = torch.full((1,), i / SAMPLE_STEPS)
        if scale == 0.0:
            v = field(x, t, y=null)
        else:
            v = field.forward_with_cfg(x, t, torch.tensor([label]), cfg_scale=scale)
        x = x + v / SAMPLE_STEPS
        frames.append(x[0])
    return [_image(f, 2) for f in frames]


@torch.no_grad()
def export_trace(field: DiT, out: Path) -> Path:
    """Write every intermediate tensor the interactive docs walkthrough renders.

    One shared noise draw ``x0`` is combined with each class image ``x1`` at each
    time in ``EXPORT_T``, and every such ``x_t`` is traced under the three labels
    (0, 1, null). Keeping ``x0`` fixed means changing a control changes exactly one
    factor, which is what lets the page attribute differences to that factor.
    """
    import json

    labels = [0, 1, field.y_embedder.null_index]
    x0 = torch.randn(1, CHANNELS, IMG, IMG)
    x1 = class_images(torch.arange(N_CLASSES), noise=0.0)

    keys, xs, ts, ys = [], [], [], []
    for img in range(N_CLASSES):
        for ti, t in enumerate(EXPORT_T):
            for li, lab in enumerate(labels):
                keys.append(f"{img}|{ti}|{li}")
                xs.append((1 - t) * x0[0] + t * x1[img])
                ts.append(t)
                ys.append(lab)
    s = trace(field, torch.stack(xs), torch.tensor(ts), torch.tensor(ys))

    traces = {}
    for b, key in enumerate(keys):
        hs = [s["tokens_pos"][b]] + [h[b] for h in s["blocks"]]
        traces[key] = {
            "x_t": _image(s["x"][b]),
            "tokens": _r(s["tokens"][b]),
            "t_emb": _r(s["t_emb"][b]),
            "y_emb": _r(s["y_emb"][b]),
            "c": _r(s["c"][b]),
            "mods": [_r(m[b].reshape(6, HIDDEN)) for m in s["mod"]],
            "attn": [_r(a[b]) for a in s["attn"]],
            "tok_norm": [_r(h.norm(dim=-1)) for h in hs],
            "block_states": [
                {name: _r(value[b]) for name, value in block.items()} for block in s["block_states"]
            ],
            "final_input": _r(s["final_input"][b]),
            "final": _r(s["final"][b]),
            "v": _image(s["v"][b]),
        }

    sinus = {
        str(ti): _r(
            field.t_embedder._sinusoidal(torch.tensor([t]) * field.t_embedder.time_scale)[0]
        )
        for ti, t in enumerate(EXPORT_T)
    }
    sampling = {
        f"{lab}|{si}": _sample(field, x0, lab, sc)
        for lab in range(N_CLASSES)
        for si, sc in enumerate(EXPORT_CFG)
    }
    data = {
        "schema_version": 3,
        "config": {
            "img": IMG,
            "channels": CHANNELS,
            "patch": PATCH,
            "grid": IMG // PATCH,
            "N": (IMG // PATCH) ** 2,
            "D": HIDDEN,
            "depth": DEPTH,
            "heads": HEADS,
            "time_scale": field.t_embedder.time_scale,
            "freq_dim": field.t_embedder.frequency_dim,
            "params": sum(p.numel() for p in field.parameters()),
        },
        "t_values": list(EXPORT_T),
        "cfg_values": list(EXPORT_CFG),
        "sample_steps": SAMPLE_STEPS,
        "x0": _image(x0[0]),
        "x1": [_image(x1[k]) for k in range(N_CLASSES)],
        "pos": _r(field.pos_embed[0]),
        "emb_table": _r(field.y_embedder.embedding_table.weight),
        "sinus": sinus,
        "traces": traces,
        "sampling": sampling,
    }
    out.write_text(json.dumps(data, separators=(",", ":")))
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    torch.manual_seed(0)
    field = DiT(
        input_size=IMG,
        patch_size=PATCH,
        in_channels=CHANNELS,
        hidden_size=HIDDEN,
        depth=DEPTH,
        num_heads=HEADS,
        num_classes=N_CLASSES,
        class_dropout_prob=0.1,
    )
    print(f"[demo] training DiT ({sum(p.numel() for p in field.parameters()):,} params) ...")
    train(field)

    # One example per class on the straight path x_t = (1 - t) x0 + t x1.
    y = torch.tensor([0, 1])
    x1 = class_images(y, noise=0.0)
    x0 = torch.randn_like(x1)
    t = torch.full((2,), T_VIEW)
    x_t = (1 - T_VIEW) * x0 + T_VIEW * x1

    s = trace(field, x_t, t, y)
    print_shapes(s)

    out_dir = Path("outputs") / "dit_tensor_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in [
        plot_patchify(s, out_dir / "patchify.png"),
        plot_conditioning(field, s, out_dir / "conditioning.png"),
        plot_adaln(s, out_dir / "adaln.png"),
        plot_attention(s, out_dir / "attention.png"),
        plot_unpatchify(s, x0, x1, out_dir / "unpatchify.png"),
    ]:
        print(f"[demo] wrote {path}")

    # At t = 0 the input is pure noise and carries no class information,
    # so any difference between the columns comes from the label alone.
    x_noise = torch.randn(1, CHANNELS, IMG, IMG)
    print(f"[demo] wrote {plot_cfg(field, x_noise, torch.tensor([0.0]), out_dir / 'cfg.png')}")

    print(f"[demo] wrote {export_trace(field, out_dir / 'dit_trace.json')}")


if __name__ == "__main__":
    main()
