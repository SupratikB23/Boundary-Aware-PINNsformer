"""``Cb_net``: ``(s_hat, t_hat) -> C_b`` (PRD §4.3 `models/boundary_net.py`).

Small MLP/SIREN with positive (softplus) output and temporal-smoothness
friendly parameterization. Optional per-pollutant heads share a trunk and
split only at the last layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .activations import get_activation
from .embeddings import FourierFeatures

__all__ = ["CbNet"]


class CbNet(nn.Module):
    """Boundary inflow net: normalized ``(s, t)`` -> ``C_b >= 0``.

    Parameters
    ----------
    hidden_dim, num_layers:
        MLP geometry (``num_layers`` hidden Linear+act blocks).
    activation:
        Hidden activation name.
    fourier_features, fourier_scale:
        Optional Fourier branch on the 2-D input (0 disables).
    pollutant_list:
        e.g. ``["PM1", "PM2.5", "PM10"]``. With ``per_pollutant_heads=True``
        a separate final head is kept per pollutant.
    per_pollutant_heads:
        If True, ``forward`` requires ``pollutant`` (name or index).
    out_transform:
        ``"softplus"`` (default) or ``"identity"`` (for debugging).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        activation: str = "wavelet",
        fourier_features: int = 0,
        fourier_scale: float = 1.0,
        pollutant_list: list[str] | None = None,
        per_pollutant_heads: bool = False,
        out_transform: str = "softplus",
        **kwargs,
    ) -> None:
        super().__init__()
        # YAML aliases from configs/model/cb_net.yaml (PRD §4.2):
        # depth/width/output_transform.
        if "depth" in kwargs and "num_layers" not in kwargs:
            num_layers = kwargs.pop("depth")
        if "width" in kwargs and "hidden_dim" not in kwargs:
            hidden_dim = kwargs.pop("width")
        if "output_transform" in kwargs and "out_transform" not in kwargs:
            out_transform = kwargs.pop("output_transform")
        kwargs.pop("temporal_smoothness", None)  # loss weight, used by trainer
        if kwargs:
            raise TypeError(f"CbNet got unexpected kwargs: {sorted(kwargs)}")
        if out_transform not in ("softplus", "identity"):
            raise ValueError("out_transform must be softplus/identity")
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.out_transform = out_transform
        self.pollutant_list = list(pollutant_list) if pollutant_list else None
        self.per_pollutant_heads = bool(per_pollutant_heads)
        if self.per_pollutant_heads and not self.pollutant_list:
            raise ValueError("per_pollutant_heads=True requires pollutant_list")

        self.fourier: FourierFeatures | None = None
        in_dim = 2
        if int(fourier_features) > 0:
            self.fourier = FourierFeatures(
                in_dim=2, num_features=int(fourier_features), scale=float(fourier_scale)
            )
            in_dim = self.fourier.out_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(self.num_layers):
            layers += [nn.Linear(prev, self.hidden_dim), get_activation(activation)]
            prev = self.hidden_dim
        self.trunk = nn.Sequential(*layers)

        def _head() -> nn.Module:
            return nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                get_activation(activation),
                nn.Linear(self.hidden_dim // 2, 1),
            )

        if self.per_pollutant_heads:
            assert self.pollutant_list is not None
            self._key = {p: str(p).replace(".", "_") for p in self.pollutant_list}
            self.heads = nn.ModuleDict({self._key[p]: _head() for p in self.pollutant_list})
            self.head_shared = None
        else:
            self._key = {}
            self.head_shared = _head()
            self.heads = None
        self.softplus = nn.Softplus() if out_transform == "softplus" else nn.Identity()

    # ------------------------------------------------------------------
    def _prep(self, s_hat, t_hat) -> torch.Tensor:
        from ..utils.dtype import infer_float_dtype

        dt = infer_float_dtype(s_hat, t_hat)
        s = torch.as_tensor(s_hat, dtype=dt).reshape(-1, 1)
        t = torch.as_tensor(t_hat, dtype=dt).reshape(-1, 1)
        if s.size(0) != t.size(0):
            raise ValueError(f"s_hat and t_hat batch mismatch: {s.shape} vs {t.shape}")
        x = torch.cat([s, t], dim=-1)
        if self.fourier is not None:
            x = self.fourier(x)
        return x

    def forward(
        self,
        s_hat: torch.Tensor,
        t_hat: torch.Tensor,
        pollutant: str | int | None = None,
    ) -> torch.Tensor:
        """Return ``C_b (B, 1)`` ≥ 0."""
        x = self._prep(s_hat, t_hat)
        h = self.trunk(x)
        if self.per_pollutant_heads:
            assert self.heads is not None and self.pollutant_list is not None
            if pollutant is None:
                raise ValueError("pollutant required when per_pollutant_heads=True")
            key = self.pollutant_list[int(pollutant)] if isinstance(pollutant, int) else str(pollutant)
            if key not in self.heads and key in self._key:
                key = self._key[key]
            if key not in self.heads:
                raise KeyError(f"unknown pollutant '{pollutant}'; heads: {self.pollutant_list}")
            out = self.heads[key](h)
        else:
            assert self.head_shared is not None
            out = self.head_shared(h)
        return self.softplus(out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
