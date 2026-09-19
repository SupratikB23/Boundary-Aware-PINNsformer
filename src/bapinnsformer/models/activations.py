"""Activations (PRD §4.3 `models/activations.py`, E3 ablation).

Provides the PINNsformer Wavelet activation plus SIREN/Sine and common
alternatives for the E3 ablation (``wavelet`` vs ``tanh``/``gelu``/...).

Wavelet choice
--------------
* The Morlet-style form ``sin(omega·x)·exp(-x²/2)`` is the localized
  wavelet used in PINNsformer-family code.
* The PINNsformer ``sin``/``cos`` Fourier mix additionally lives in
  :mod:`bapinnsformer.models.embeddings` (Fourier features); the config
  switch for both is a single string (see :func:`get_activation`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "Wavelet",
    "Sine",
    "SIREN",
    "Siren",
    "SirenAct",
    "TanhAct",
    "GeluAct",
    "get_activation",
]


class Wavelet(nn.Module):
    """Morlet-style wavelet: ``sin(omega·x)·exp(-x²/2)`` (E3 default)."""

    def __init__(self, omega: float = 1.0) -> None:
        super().__init__()
        self.omega = float(omega)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x) * torch.exp(-0.5 * x * x)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"omega={self.omega}"


class Sine(nn.Module):
    """SIREN-style sine activation: ``sin(w0·x)`` (Sitzmann et al. 2020)."""

    def __init__(self, w0: float = 30.0, omega_0: float | None = None,
                 omega: float | None = None) -> None:
        super().__init__()
        if omega_0 is not None:
            w0 = omega_0
        if omega is not None:
            w0 = omega
        self.w0 = float(w0)
        self.omega_0 = self.w0  # alias for older call sites

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"w0={self.w0}"


# Paper-name / case-variant aliases.
SIREN = Sine
Siren = Sine


class SirenAct(Sine):
    """Alias keeping the older ``SirenAct`` import path working."""


class TanhAct(nn.Module):
    """``tanh`` as a named module (older call sites)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class GeluAct(nn.Module):
    """``gelu`` as a named module (older call sites)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as _F

        return _F.gelu(x)


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "wavelet": Wavelet,
    "sine": Sine,
    "siren": Sine,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "identity": nn.Identity,
}


def get_activation(name: str, **kwargs) -> nn.Module:
    """Return an activation module by string name (case-insensitive).

    Extra kwargs are forwarded to the module constructor (``omega`` for
    :class:`Wavelet`, ``w0``/``omega_0`` for :class:`Sine`); stateless
    activations ignore extras. ``"tanh_act"``/``"gelu_act"``/``"siren_act"``
    spellings resolve to the legacy aliases.
    """
    key = str(name).strip().lower()
    aliases = {"tanh_act": "tanh", "gelu_act": "gelu", "siren_act": "siren"}
    key = aliases.get(key, key)
    if key not in _ACTIVATIONS:
        raise KeyError(
            f"unknown activation '{name}'; expected one of {sorted(_ACTIVATIONS)}"
        )
    cls = _ACTIVATIONS[key]
    if kwargs and cls in (Sine, Wavelet):
        try:
            return cls(**kwargs)  # type: ignore[call-arg]
        except TypeError:
            pass
    try:
        return cls(**kwargs) if kwargs else cls()  # type: ignore[call-arg]
    except TypeError:
        return cls()  # stateless activations ignore extras
