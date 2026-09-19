"""PINNsformer ``C_net`` (PRD §4.3 `models/pinnsformer.py`).

Encoder–decoder attention stack over the length-``L`` pseudo-sequence;
returns the concentration at the query token (index 0). Configurable
``d_model`` / layers / heads, optional encoder-free mode (literature
argues the encoder is redundant — kept as a flag for E3), and an
attention-weight extraction hook for the interpretability figure
(flagged boundary-hitting tokens vs attention mass).

Default config targets < 0.5 M parameters (PRD §9) and < 3 GB at L=5.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .activations import get_activation
from .embeddings import SpatioTemporalEmbedding


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block with accessible attention weights."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        activation: str = "wavelet",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self._last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        h = self.norm1(x)
        attn_out, attn_w = self.self_attn(
            h, h, h, need_weights=True, average_attn_weights=False
        )
        self._last_attn = attn_w.detach() if not torch.is_grad_enabled() else attn_w
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        if need_weights:
            return x, attn_w
        return x


class PINNsformer(nn.Module):
    """``C_net``: ``tokens (B, L, 3)`` (normalized) -> ``C (B, 1)`` (physical, ≥0).

    Parameters
    ----------
    d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward:
        Transformer geometry.
    activation:
        Feed-forward / head activation name (``wavelet`` default).
    encoder_free:
        If True, skip the encoder stack (decoder-only).
    in_dim:
        Token dim (3).
    fourier_features, fourier_scale:
        Passed to :class:`SpatioTemporalEmbedding`.
    dropout:
        Dropout rate.
    query_index:
        Which token to read out (0 = query, per :mod:`pseudoseq`).
    out_transform:
        ``"softplus"`` (default, non-negative ``C``) or ``"identity"``.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        activation: str = "wavelet",
        encoder_free: bool = False,
        in_dim: int = 3,
        fourier_features: int = 0,
        fourier_scale: float = 1.0,
        query_index: int = 0,
        out_transform: str = "softplus",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # YAML aliases from configs/model/pinnsformer.yaml (PRD §4.2):
        # n_encoder_layers/n_decoder_layers/n_heads/d_ff.
        if "n_encoder_layers" in kwargs and "num_encoder_layers" not in kwargs:
            num_encoder_layers = kwargs.pop("n_encoder_layers")
        if "n_decoder_layers" in kwargs and "num_decoder_layers" not in kwargs:
            num_decoder_layers = kwargs.pop("n_decoder_layers")
        if "n_heads" in kwargs and "nhead" not in kwargs:
            nhead = kwargs.pop("n_heads")
        if "d_ff" in kwargs and "dim_feedforward" not in kwargs:
            dim_feedforward = kwargs.pop("d_ff")
        kwargs.pop("param_target", None)  # budget note, not a constructor arg
        if kwargs:
            raise TypeError(f"PINNsformer got unexpected kwargs: {sorted(kwargs)}")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.encoder_free = bool(encoder_free)
        self.query_index = int(query_index)
        if out_transform not in ("softplus", "identity"):
            raise ValueError("out_transform must be softplus/identity")
        self.out_transform = out_transform

        self.embedding = SpatioTemporalEmbedding(
            in_dim=in_dim,
            d_model=self.d_model,
            fourier_features=int(fourier_features),
            fourier_scale=float(fourier_scale),
            dropout=float(dropout),
        )
        mk = lambda: TransformerBlock(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation=activation,
        )
        self.encoder = nn.ModuleList([mk() for _ in range(int(num_encoder_layers))])
        self.decoder = nn.ModuleList([mk() for _ in range(int(num_decoder_layers))])
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            get_activation(activation),
            nn.Linear(self.d_model // 2, 1),
        )
        self.softplus = nn.Softplus() if out_transform == "softplus" else nn.Identity()

    # ------------------------------------------------------------------
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Tokens ``(B, L, 3)`` normalized -> ``C (B, 1)`` physical ≥ 0."""
        if tokens.dim() != 3 or tokens.size(-1) != 3:
            raise ValueError(f"expected tokens (B, L, 3), got {tuple(tokens.shape)}")
        h = self.embedding(tokens)
        for blk in self.encoder:
            if self.encoder_free:
                break
            out = blk(h)
            h = out[0] if isinstance(out, tuple) else out
        for blk in self.decoder:
            out = blk(h)
            h = out[0] if isinstance(out, tuple) else out
        q = h[:, self.query_index, :]
        return self.softplus(self.head(q))

    # ------------------------------------------------------------------
    def get_attention_weights(self, tokens: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """Run a forward pass capturing per-layer attention weights.

        Returns
        -------
        ``{"encoder": [ (B, H, L, L), ... ], "decoder": [...]}`` with
        ``torch.no_grad`` semantics; used for the attention-attribution
        figure (boundary-hitting tokens highlighted).
        """
        was_training = self.training
        self.eval()
        enc_w: list[torch.Tensor] = []
        dec_w: list[torch.Tensor] = []
        with torch.no_grad():
            h = self.embedding(tokens)
            for blk in self.encoder:
                if self.encoder_free:
                    break
                h, w = blk(h, need_weights=True)  # type: ignore[misc]
                enc_w.append(w.detach().cpu())
            for blk in self.decoder:
                h, w = blk(h, need_weights=True)  # type: ignore[misc]
                dec_w.append(w.detach().cpu())
        if was_training:
            self.train()
        return {"encoder": enc_w, "decoder": dec_w}

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Trainable parameter count (default config must be < 0.5 M)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Small default (< 0.5 M params) per PRD §9."""
        return {
            "d_model": 64,
            "nhead": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.0,
            "activation": "wavelet",
            "encoder_free": False,
            "fourier_features": 0,
        }
