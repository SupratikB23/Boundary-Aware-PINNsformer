"""``S_net``: ``(x_hat, y_hat, t_hat) -> S`` (PRD §4.3 `models/source_net.py`).

Small MLP with non-negative (softplus) output and an L1/TV sparsity-friendly
parameterization. The field is a recovered nuisance quantity (PRD §1.4),
regularized in :mod:`bapinnsformer.train.losses`, not a validated inventory.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .activations import get_activation
from .embeddings import FourierFeatures

__all__ = ["SNet"]


class SNet(nn.Module):
    """Interior source net: normalized ``(x, y, t)`` -> ``S >= 0``."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        activation: str = "wavelet",
        fourier_features: int = 0,
        fourier_scale: float = 1.0,
        out_transform: str = "softplus",
        **kwargs,
    ) -> None:
        super().__init__()
        # YAML aliases from configs/model/s_net.yaml (PRD §4.2).
        if "depth" in kwargs and "num_layers" not in kwargs:
            num_layers = kwargs.pop("depth")
        if "width" in kwargs and "hidden_dim" not in kwargs:
            hidden_dim = kwargs.pop("width")
        if "output_transform" in kwargs and "out_transform" not in kwargs:
            out_transform = kwargs.pop("output_transform")
        kwargs.pop("l1_weight", None)  # loss weights, used by trainer
        kwargs.pop("tv_weight", None)
        if kwargs:
            raise TypeError(f"SNet got unexpected kwargs: {sorted(kwargs)}")
        if out_transform not in ("softplus", "identity"):
            raise ValueError("out_transform must be softplus/identity")
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.out_transform = out_transform

        self.fourier: FourierFeatures | None = None
        in_dim = 3
        if int(fourier_features) > 0:
            self.fourier = FourierFeatures(
                in_dim=3, num_features=int(fourier_features), scale=float(fourier_scale)
            )
            in_dim = self.fourier.out_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(self.num_layers):
            layers += [nn.Linear(prev, self.hidden_dim), get_activation(activation)]
            prev = self.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            get_activation(activation),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.softplus = nn.Softplus() if out_transform == "softplus" else nn.Identity()

    def forward(
        self,
        x_hat: torch.Tensor,
        y_hat: torch.Tensor,
        t_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``S (B, 1)`` ≥ 0."""
        from ..utils.dtype import infer_float_dtype

        dt = infer_float_dtype(x_hat, y_hat, t_hat)
        x = torch.as_tensor(x_hat, dtype=dt).reshape(-1, 1)
        y = torch.as_tensor(y_hat, dtype=dt).reshape(-1, 1)
        t = torch.as_tensor(t_hat, dtype=dt).reshape(-1, 1)
        if not (x.size(0) == y.size(0) == t.size(0)):
            raise ValueError("x_hat, y_hat, t_hat batch mismatch")
        h = torch.cat([x, y, t], dim=-1)
        if self.fourier is not None:
            h = self.fourier(h)
        return self.softplus(self.head(self.trunk(h)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
