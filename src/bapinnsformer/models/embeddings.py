"""Spatiotemporal embeddings (PRD §4.3 `models/embeddings.py`).

``SpatioTemporalEmbedding`` maps pseudo-sequence tokens ``(x_hat, y_hat,
t_hat)`` in ``[-1, 1]^3`` to ``d_model`` vectors via a raw linear branch
plus an optional random-Fourier-feature branch (the ``sin``/``cos`` mix
of the PINNsformer line) and a sinusoidal positional term over the
sequence axis.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Random Fourier mapping ``v -> [sin(2π B v), cos(2π B v)]``.

    Parameters
    ----------
    in_dim:
        Input coordinate dim (3 for ``(x_hat, y_hat, t_hat)``).
    num_features:
        Number of Fourier basis frequencies ``M``; output dim is ``2 * M``.
    scale:
        Std of the Gaussian frequency matrix ``B``.
    learnable:
        If True, ``B`` is an ``nn.Parameter``; else a fixed buffer.
    seed:
        RNG seed for reproducibility (config + seed => same ``B``).
    """

    def __init__(
        self,
        in_dim: int = 3,
        num_features: int = 16,
        scale: float = 1.0,
        learnable: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        gen = torch.Generator().manual_seed(int(seed))
        B = torch.randn(in_dim, num_features, generator=gen) * float(scale)
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)
        self.in_dim = int(in_dim)
        self.num_features = int(num_features)
        self.scale = float(scale)

    @property
    def out_dim(self) -> int:
        return 2 * self.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim) -> (..., 2*M)
        proj = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"in_dim={self.in_dim}, num_features={self.num_features}, scale={self.scale}"


def _sinusoidal_positional(seq_len: int, d_model: int, device, dtype) -> torch.Tensor:
    """Classic sinusoidal positional encoding ``(L, d_model)``."""
    pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(d_model, device=device, dtype=dtype).unsqueeze(0)
    angle = pos / torch.pow(10000.0, (2 * (i // 2)) / d_model)
    pe = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe


class SpatioTemporalEmbedding(nn.Module):
    """Linear + Fourier + positional embedding for pseudo-sequences.

    Parameters
    ----------
    in_dim:
        Token dim (3 for ``(x_hat, y_hat, t_hat)``).
    d_model:
        Output width.
    fourier_features:
        ``M`` Fourier frequencies; 0 disables the Fourier branch.
    fourier_scale, seed:
        Passed to :class:`FourierFeatures`.
    use_raw:
        If True keep the raw linear branch; the two branches are summed.
        If False (and Fourier enabled) use Fourier only.
    use_positional:
        Add sinusoidal position codes over the ``L`` axis.
    dropout:
        Dropout after the sum (0.0 default, CPU-safe).
    """

    def __init__(
        self,
        in_dim: int = 3,
        d_model: int = 64,
        fourier_features: int = 0,
        fourier_scale: float = 1.0,
        seed: int = 0,
        use_raw: bool = True,
        use_positional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.use_raw = bool(use_raw)
        self.use_positional = bool(use_positional)

        self.raw_proj = nn.Linear(self.in_dim, self.d_model) if self.use_raw else None

        self.fourier: FourierFeatures | None = None
        self.fourier_proj: nn.Linear | None = None
        if int(fourier_features) > 0:
            self.fourier = FourierFeatures(
                in_dim=self.in_dim,
                num_features=int(fourier_features),
                scale=float(fourier_scale),
                seed=int(seed),
            )
            self.fourier_proj = nn.Linear(self.fourier.out_dim, self.d_model)

        if self.raw_proj is None and self.fourier_proj is None:
            raise ValueError("at least one of raw / fourier branches must be enabled")

        self.dropout = nn.Dropout(float(dropout))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed ``tokens (B, L, in_dim)`` -> ``(B, L, d_model)``."""
        if tokens.dim() != 3 or tokens.size(-1) != self.in_dim:
            raise ValueError(
                f"expected tokens of shape (B, L, {self.in_dim}), got {tuple(tokens.shape)}"
            )
        out: torch.Tensor | None = None
        if self.raw_proj is not None:
            out = self.raw_proj(tokens)
        if self.fourier is not None and self.fourier_proj is not None:
            f = self.fourier_proj(self.fourier(tokens))
            out = f if out is None else out + f
        assert out is not None
        if self.use_positional:
            pe = _sinusoidal_positional(
                tokens.size(1), self.d_model, tokens.device, out.dtype
            )
            out = out + pe.unsqueeze(0)
        return self.dropout(out)
