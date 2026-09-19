"""Learnable physical scalars ``K`` / ``lam`` (PRD §4.3 `models/params.py`).

* Positive by log-parameterization: ``K = exp(logK)``, ``lam = exp(loglam)``.
* Optional ``K`` as a linear function of boundary-layer height / stability.
* PM ordering helper enforces ``lam_PM1 < lam_PM2.5 < lam_PM10`` (PRD §7.2).

Symbols: ``K`` eddy diffusivity, ``lam`` deposition+removal rate.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["PhysParams", "KFromBLH", "enforce_ordering", "ordered_lams_from_raw"]


class PhysParams(nn.Module):
    """Log-parameterized ``(K, lam)`` pair.

    Parameters
    ----------
    K_init, lam_init:
        Positive initial values in physical units.
    learn_K, learn_lam:
        If False the corresponding scalar is a fixed buffer.
    K_min, K_max, lam_min, lam_max:
        Optional clamps applied on read (defaults None = unclamped).
    """

    def __init__(
        self,
        K_init: float = 100.0,
        lam_init: float = 1e-5,
        learn_K: bool = True,
        learn_lam: bool = True,
        K_min: float | None = None,
        K_max: float | None = None,
        lam_min: float | None = None,
        lam_max: float | None = None,
    ) -> None:
        super().__init__()
        if K_init <= 0 or lam_init <= 0:
            raise ValueError("K_init and lam_init must be positive")
        self._K_init = float(K_init)
        self._lam_init = float(lam_init)
        logK = torch.tensor(math.log(float(K_init)), dtype=torch.get_default_dtype())
        loglam = torch.tensor(math.log(float(lam_init)), dtype=torch.get_default_dtype())
        if learn_K:
            self.logK = nn.Parameter(logK)
        else:
            self.register_buffer("logK", logK)
        if learn_lam:
            self.loglam = nn.Parameter(loglam)
        else:
            self.register_buffer("loglam", loglam)
        self.K_min, self.K_max = K_min, K_max
        self._K_max, self._lam_max = K_max, lam_max
        self.lam_min, self.lam_max = lam_min, lam_max

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(K, lam)`` positive scalars (0-d tensors, grad-carrying)."""
        K = torch.exp(self.logK)
        lam = torch.exp(self.loglam)
        if self.K_min is not None:
            K = torch.clamp(K, min=float(self.K_min))
        if self._K_max is not None:
            K = torch.clamp(K, max=float(self._K_max))
        if self.lam_min is not None:
            lam = torch.clamp(lam, min=float(self.lam_min))
        if self.lam_max is not None:
            lam = torch.clamp(lam, max=float(self.lam_max))
        return K.reshape(()), lam.reshape(())

    @property
    def K(self) -> torch.Tensor:
        """Current ``K`` (convenience accessor)."""
        return self.forward()[0]

    @property
    def lam(self) -> torch.Tensor:
        """Current ``lam`` (convenience accessor)."""
        return self.forward()[1]

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        K, lam = self.forward()
        return f"K={float(K):.4g}, lam={float(lam):.4g}"


class KFromBLH(nn.Module):
    """Optional diffusivity model ``K(BLH, stability) = softplus(a·blh + b)``.

    A thin linear alternative to the scalar :class:`PhysParams` log-param,
    for the winter-inversion sensitivity path (PRD risk register): higher
    boundary-layer height → stronger mixing.
    """

    def __init__(self, a_init: float = 0.0, b_init: float = 4.6) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a_init)))
        self.b = nn.Parameter(torch.tensor(float(b_init)))
        self.softplus = nn.Softplus()

    def forward(self, blh: torch.Tensor) -> torch.Tensor:
        from ..utils.dtype import as_float_tensor

        blh = as_float_tensor(blh)
        return self.softplus(self.a * blh + self.b)


def ordered_lams_from_raw(
    r_pm1: torch.Tensor, r_pm25: torch.Tensor, r_pm10: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map unconstrained raw scores to ordered ``lam_PM1 < lam_PM2.5 < lam_PM10``.

    Construction: ``l1 = softplus(r1)``, ``l2 = l1 + softplus(r2)``,
    ``l3 = l2 + softplus(r3)`` — strictly increasing by positive increments.
    """
    sp = nn.Softplus()
    l1 = sp(torch.as_tensor(r_pm1))
    l2 = l1 + sp(torch.as_tensor(r_pm25))
    l3 = l2 + sp(torch.as_tensor(r_pm10))
    return l1, l2, l3


def enforce_ordering(lams: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Project a ``{PM1, PM2.5, PM10}`` lam dict onto the ordered feasible set.

    Uses cumulative-maximum projection so the returned values satisfy
    ``PM1 <= PM2.5 <= PM10`` while staying close to the inputs. Keys other
    than the three PM fractions pass through untouched.
    """
    out = dict(lams)
    keys = ["PM1", "PM2.5", "PM10"]
    if all(k in out for k in keys):
        vals = [torch.as_tensor(out[k]) for k in keys]
        ordered = [vals[0]]
        for v in vals[1:]:
            ordered.append(torch.maximum(v, ordered[-1]))
        for k, v in zip(keys, ordered):
            out[k] = v
    return out
