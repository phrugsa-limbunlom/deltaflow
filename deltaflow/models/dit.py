"""Diffusion-Transformer (DiT) velocity field with adaLN-Zero conditioning.

This module provides a native transformer backbone for DeltaFlow, conditioned
through **Adaptive Layer Normalization with Zero initialization (adaLN-Zero)**.
The displacement/transport framing still holds: the network regresses the
flow-matching target velocity ``Delta = x1 - x0`` (see
`deltaflow.losses.ConditionalFlowMatchingLoss`), only now the field is a
sequence model over image patches rather than a convolutional UNet.

The conditioning path follows Peebles & Xie (2023). A shared conditioning
embedding ``c`` (time embedding plus, optionally, a class embedding) drives a
small MLP per block that emits per-channel ``(shift, scale, gate)`` modulation
parameters. The gates, and the final output projection, are zero-initialized, so
every residual branch starts as the identity and the whole field starts at
``v = 0``. Training then departs smoothly from that stable fixed point, which is
the stability trick that makes deep DiTs trainable without warmup tricks.

Classifier-free guidance (CFG) is supported through a learned *null* class token
in `LabelEmbedding`: at train time a fraction of labels are dropped to the null
token, and at sample time `DiT.forward_with_cfg` extrapolates between the
conditional and unconditional velocity.

References:
    Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023),
    https://arxiv.org/abs/2212.09748.
    Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022),
    https://arxiv.org/abs/2207.12598.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.base_velocity_field import BaseVelocityField


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaLN modulation ``x * (1 + scale) + shift`` with broadcasting.

    ``x`` has shape ``(B, N, D)`` (batch, tokens, channels), while ``shift`` and
    ``scale`` have shape ``(B, D)``. The ``1 +`` keeps the transform centred on
    the identity, so a zero-initialized modulation MLP leaves ``x`` unchanged.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedding(nn.Module):
    """Reusable sinusoidal timestep embedding followed by a small MLP.

    The continuous flow-matching time ``t in [0, 1]`` (DeltaFlow convention:
    ``t=0`` is noise, ``t=1`` is data) is first multiplied by ``time_scale`` to
    spread it across the sinusoidal frequency band, then embedded with the usual
    transformer sinusoids and projected by a two-layer MLP.

    Args:
        hidden_size: output (and MLP) width.
        frequency_dim: width of the raw sinusoidal features before the MLP.
        time_scale: multiplier applied to ``t`` before the sinusoids. The
            default of ``1000`` mirrors the diffusion-timestep range DiT was
            tuned on and gives continuous ``t in [0, 1]`` enough resolution.
        max_period: controls the lowest sinusoidal frequency.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_dim: int = 256,
        time_scale: float = 1000.0,
        max_period: int = 10000,
    ):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.time_scale = time_scale
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(1)
        emb = self._sinusoidal(t * self.time_scale)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class LabelEmbedding(nn.Module):
    """Class-label embedding with a learned null token for guidance.

    The embedding table has ``num_classes + 1`` rows; the extra row is the
    *null* (unconditional) token used by classifier-free guidance. During
    training a fraction ``dropout_prob`` of labels are replaced by the null
    token, teaching the field both the conditional and unconditional velocity
    with one set of weights.

    Args:
        num_classes: number of real classes.
        hidden_size: embedding width.
        dropout_prob: probability of dropping a label to the null token at
            train time (set ``0`` to disable CFG training).
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.null_index = num_classes
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Replace a random subset of ``labels`` with the null token.

        If ``force_drop_ids`` is given (a boolean/0-1 mask), those positions are
        dropped deterministically instead, which is how the unconditional branch
        is requested at sampling time.
        """
        if force_drop_ids is None:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop = force_drop_ids.to(torch.bool)
        return torch.where(drop, torch.full_like(labels, self.null_index), labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: Optional[bool] = None,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        train = self.training if train is None else train
        if (train and self.dropout_prob > 0) or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class _Attention(nn.Module):
    """Minimal multi-head self-attention over token sequences ``(B, N, D)``."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class DiTBlock(nn.Module):
    """A transformer block modulated by adaLN-Zero conditioning.

    Both sub-layers (self-attention and MLP) are wrapped as

    ``x = x + gate * sublayer(modulate(norm(x), shift, scale))``,

    where ``(shift, scale, gate)`` are produced per sub-layer from the shared
    conditioning embedding ``c``. The six modulation vectors come from a single
    ``SiLU -> Linear`` head whose weights are zero-initialized (see
    `DiT.initialize_weights`), so at initialization every gate is zero and the
    block is the identity map.

    Args:
        hidden_size: token channel width.
        num_heads: attention heads.
        mlp_ratio: hidden expansion of the feed-forward MLP.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = _Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """adaLN-Zero output head that maps tokens back to patch pixels.

    The final normalization is modulated by ``(shift, scale)`` from the
    conditioning embedding, then a linear layer projects each token to
    ``patch_size**2 * out_channels`` values. Both the modulation head and the
    output projection are zero-initialized so the field predicts ``v = 0`` at the
    start of training.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(BaseVelocityField):
    r"""A class-conditional Diffusion Transformer velocity field.

    The field patchifies an image ``x`` of shape ``(B, C, H, W)`` into a token
    sequence, adds a learned position embedding, runs it through ``depth``
    `DiTBlock` layers conditioned on ``time + class`` embeddings, and unpatchifies
    the output back to a velocity of the same shape as ``x``. It satisfies the
    `BaseVelocityField` contract, so it drops straight into DeltaFlow's losses
    and solvers.

    Conditioning is passed through the opaque ``**cond`` channel as ``cond["y"]``
    (integer class labels of shape ``(B,)``). When ``y`` is omitted the field
    runs unconditionally via the learned null token.

    Args:
        input_size: spatial size of the (square) input image.
        patch_size: side length of each square patch; must divide ``input_size``.
        in_channels: number of image channels.
        hidden_size: transformer token width.
        depth: number of `DiTBlock` layers.
        num_heads: attention heads per block.
        mlp_ratio: feed-forward expansion ratio.
        num_classes: number of conditioning classes (a null token is added on
            top for classifier-free guidance).
        class_dropout_prob: train-time label-dropout probability for CFG.
        time_scale: multiplier applied to ``t`` inside `TimestepEmbedding`.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 1,
        hidden_size: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        class_dropout_prob: float = 0.1,
        time_scale: float = 1000.0,
    ):
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by patch_size ({patch_size})"
            )
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_patches_side = input_size // patch_size
        self.num_patches = self.num_patches_side**2

        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.t_embedder = TimestepEmbedding(hidden_size, time_scale=time_scale)
        self.y_embedder = LabelEmbedding(num_classes, hidden_size, class_dropout_prob)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Xavier-init linear layers, then zero-init every adaLN-Zero gate."""

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        nn.init.normal_(self.pos_embed, std=0.02)

        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # adaLN-Zero: zero the modulation heads so blocks start as the identity.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        # Zero the final layer so the initial predicted velocity is exactly 0.
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, num_patches, p*p*C) -> (B, C, H, W)``."""
        c = self.out_channels
        p = self.patch_size
        n = self.num_patches_side
        x = x.reshape(x.shape[0], n, n, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(x.shape[0], c, n * p, n * p)

    def _conditioning(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
        force_drop_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        c = self.t_embedder(t)
        if y is None:
            y = torch.full((x.shape[0],), self.y_embedder.null_index, device=x.device)
            c = c + self.y_embedder(y, train=False)
        else:
            c = c + self.y_embedder(y.to(x.device), force_drop_ids=force_drop_ids)
        return c

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        force_drop_ids: Optional[torch.Tensor] = None,
        **cond,
    ) -> torch.Tensor:
        """Predict the velocity ``v_theta(x, t, y)``.

        Args:
            x: input image batch ``(B, C, H, W)``.
            t: time of shape ``(B,)`` or a scalar (DeltaFlow convention
                ``t=0`` noise, ``t=1`` data).
            y: optional integer class labels ``(B,)`` forwarded as ``cond["y"]``;
                omit for unconditional inference.
            force_drop_ids: optional mask selecting which labels to force to the
                null token (used by the unconditional branch of CFG).
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device)
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        h = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        h = h + self.pos_embed
        c = self._conditioning(x, t, y, force_drop_ids)
        for block in self.blocks:
            h = block(h, c)
        h = self.final_layer(h, c)
        return self._unpatchify(h)

    @torch.no_grad()
    def forward_with_cfg(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float = 4.0
    ) -> torch.Tensor:
        r"""Classifier-free-guided velocity for sampling.

        Runs the field once with the real labels and once with the null token,
        then extrapolates

        \[
        v_{\mathrm{cfg}} = v_\text{uncond}
            + s\,\bigl(v_\text{cond} - v_\text{uncond}\bigr),
        \]

        with guidance weight ``s = cfg_scale``. The two passes are batched
        together, so this costs one forward over a doubled batch.
        """
        half = x
        combined = torch.cat([half, half], dim=0)
        t_cat = torch.cat([t, t], dim=0) if t.dim() > 0 else t
        y_cat = torch.cat([y, y], dim=0)
        drop = torch.zeros(y_cat.shape[0], dtype=torch.bool, device=y.device)
        drop[y.shape[0] :] = True
        v = self.forward(combined, t_cat, y=y_cat, force_drop_ids=drop)
        v_cond, v_uncond = v.chunk(2, dim=0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)


__all__ = [
    "DiT",
    "DiTBlock",
    "FinalLayer",
    "LabelEmbedding",
    "TimestepEmbedding",
    "modulate",
]
