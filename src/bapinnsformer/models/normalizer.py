"""Normalizer: single source of truth for units (PRD §§2.1, 4.3).

All networks see normalized ``(x_hat, y_hat, t_hat) in [-1, 1]^3`` and
``s_hat in [-1, 1]``. Physical <-> normalized maps are owned here and
serialized with every checkpoint. **No module may hardcode scaling.**

Conventions
-----------
* Space/time/arc-length use an affine map to ``[-1, 1]`` derived purely
  from bounds (``fit_from_bounds`` preferred, ``fit_from_data`` fallback).
* Concentration ``C`` uses the affine map when ``C_bounds`` are known,
  a ``(c_shift, c_scale)`` shift/scale when constructed that way, and is
  otherwise the identity (no hidden scaling).
* All methods accept ``torch.Tensor``, ``numpy.ndarray`` or Python scalars.
  Torch tensors stay tensors (autograd-safe); numpy/list inputs return numpy.

Backward-compatible constructor aliases (``x_range``/``c_scale``/``c_shift``)
are accepted so older configs keep working.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a pinned dep
    np = None  # type: ignore[assignment]

import torch

__all__ = ["Normalizer"]


def _is_torch(a: Any) -> bool:
    return isinstance(a, torch.Tensor)


def _affine_fwd(a: Any, lo: float, hi: float) -> Any:
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return (a - mid) / half


def _affine_inv(b: Any, lo: float, hi: float) -> Any:
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return b * half + mid


class Normalizer:
    """Affine normalizer for ``x, y, t, C, s``."""

    FIELDS = ("x", "y", "t", "C", "s")

    def __init__(
        self,
        x_bounds: tuple[float, float] | None = None,
        y_bounds: tuple[float, float] | None = None,
        t_bounds: tuple[float, float] | None = None,
        C_bounds: tuple[float, float] | None = None,
        s_bounds: tuple[float, float] | None = None,
        # --- backward-compatible aliases ---
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        t_range: tuple[float, float] | None = None,
        c_scale: float | None = None,
        c_shift: float = 0.0,
    ) -> None:
        x_bounds = x_bounds if x_bounds is not None else x_range
        y_bounds = y_bounds if y_bounds is not None else y_range
        t_bounds = t_bounds if t_bounds is not None else t_range
        for name, b in (("x", x_bounds), ("y", y_bounds), ("t", t_bounds)):
            if b is None or not (len(b) == 2 and float(b[1]) > float(b[0])):
                raise ValueError(f"'{name}' bounds must be (lo, hi) with hi > lo, got {b}")
        self.x_bounds = (float(x_bounds[0]), float(x_bounds[1]))  # type: ignore[index]
        self.y_bounds = (float(y_bounds[0]), float(y_bounds[1]))  # type: ignore[index]
        self.t_bounds = (float(t_bounds[0]), float(t_bounds[1]))  # type: ignore[index]
        # compat aliases
        self.x_range = self.x_bounds
        self.y_range = self.y_bounds
        self.t_range = self.t_bounds

        if C_bounds is not None:
            if not (len(C_bounds) == 2 and float(C_bounds[1]) > float(C_bounds[0])):
                raise ValueError(f"'C' bounds must be (lo, hi) with hi > lo, got {C_bounds}")
            self.C_bounds: tuple[float, float] | None = (float(C_bounds[0]), float(C_bounds[1]))
            self.c_scale = 0.5 * (self.C_bounds[1] - self.C_bounds[0])
            self.c_shift = 0.5 * (self.C_bounds[1] + self.C_bounds[0])
        elif c_scale is not None:
            if not float(c_scale) > 0:
                raise ValueError("c_scale must be positive")
            self.C_bounds = None
            self.c_scale = float(c_scale)
            self.c_shift = float(c_shift)
        else:
            self.C_bounds = None
            self.c_scale = 1.0
            self.c_shift = 0.0

        if s_bounds is not None:
            if not (len(s_bounds) == 2 and float(s_bounds[1]) > float(s_bounds[0])):
                raise ValueError(f"'s' bounds must be (lo, hi) with hi > lo, got {s_bounds}")
            self.s_bounds: tuple[float, float] | None = (float(s_bounds[0]), float(s_bounds[1]))
        else:
            self.s_bounds = None

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def fit_from_bounds(
        cls,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        t_bounds: tuple[float, float],
        C_bounds: tuple[float, float] | None = None,
        s_bounds: tuple[float, float] | None = None,
    ) -> "Normalizer":
        """Build directly from known domain bounds (preferred path)."""
        return cls(x_bounds, y_bounds, t_bounds, C_bounds, s_bounds)

    @classmethod
    def fit_from_data(
        cls,
        x: Any,
        y: Any,
        t: Any,
        C: Any | None = None,
        s: Any | None = None,
        pad_frac: float = 0.0,
    ) -> "Normalizer":
        """Infer bounds from data arrays (min/max + optional padding)."""
        def _minmax(v: Any) -> tuple[float, float]:
            if _is_torch(v):
                tv = v.detach().float().reshape(-1)
                lo, hi = float(tv.min().item()), float(tv.max().item())
            elif np is not None and isinstance(v, np.ndarray):
                lo, hi = float(v.min()), float(v.max())
            else:
                flat = torch.as_tensor(list(v) if isinstance(v, (list, tuple)) else v,
                                       dtype=torch.float32).reshape(-1)
                lo, hi = float(flat.min().item()), float(flat.max().item())
            if not lo < hi:
                lo, hi = lo - 0.5, hi + 0.5
            if pad_frac > 0:
                span = hi - lo
                lo -= pad_frac * span
                hi += pad_frac * span
            return (lo, hi)

        return cls(_minmax(x), _minmax(y), _minmax(t),
                   _minmax(C) if C is not None else None,
                   _minmax(s) if s is not None else None)

    # ------------------------------------------------------------------
    # per-field forward / inverse
    # ------------------------------------------------------------------
    def normalize_x(self, x: Any) -> Any:
        return _affine_fwd(x, *self.x_bounds)

    def denormalize_x(self, x_hat: Any) -> Any:
        return _affine_inv(x_hat, *self.x_bounds)

    def normalize_y(self, y: Any) -> Any:
        return _affine_fwd(y, *self.y_bounds)

    def denormalize_y(self, y_hat: Any) -> Any:
        return _affine_inv(y_hat, *self.y_bounds)

    def normalize_t(self, t: Any) -> Any:
        return _affine_fwd(t, *self.t_bounds)

    def denormalize_t(self, t_hat: Any) -> Any:
        return _affine_inv(t_hat, *self.t_bounds)

    def normalize_C(self, C: Any) -> Any:
        """Physical ``C`` -> normalized (affine, shift/scale, or identity)."""
        if self.C_bounds is not None:
            return _affine_fwd(C, *self.C_bounds)
        if self.c_scale != 1.0 or self.c_shift != 0.0:
            return (C - self.c_shift) / self.c_scale
        return C

    def denormalize_C(self, C_hat: Any) -> Any:
        if self.C_bounds is not None:
            return _affine_inv(C_hat, *self.C_bounds)
        if self.c_scale != 1.0 or self.c_shift != 0.0:
            return C_hat * self.c_scale + self.c_shift
        return C_hat

    # lowercase aliases (older call sites)
    def normalize_c(self, c: Any) -> Any:
        return self.normalize_C(c)

    def denormalize_c(self, ch: Any) -> Any:
        return self.denormalize_C(ch)

    def normalize_s(self, s: Any) -> Any:
        if self.s_bounds is None:
            raise ValueError("s_bounds not set; cannot normalize arc-length s")
        return _affine_fwd(s, *self.s_bounds)

    def denormalize_s(self, s_hat: Any) -> Any:
        if self.s_bounds is None:
            raise ValueError("s_bounds not set; cannot denormalize arc-length s")
        return _affine_inv(s_hat, *self.s_bounds)

    # -- generic dispatch / tuple forms ----------------------------------
    _FIELD_NORM = {"x": "normalize_x", "y": "normalize_y", "t": "normalize_t",
                   "C": "normalize_C", "c": "normalize_c", "s": "normalize_s"}
    _FIELD_DENORM = {"x": "denormalize_x", "y": "denormalize_y", "t": "denormalize_t",
                     "C": "denormalize_C", "c": "denormalize_c", "s": "denormalize_s"}

    def normalize(self, *args: Any) -> Any:
        """``normalize(field, values)`` or ``normalize(x, y, t)`` tuple form."""
        if len(args) == 2 and isinstance(args[0], str):
            fn = getattr(self, self._FIELD_NORM[args[0]])
            return fn(args[1])
        if len(args) == 3:
            x, y, t = args
            return self.normalize_x(x), self.normalize_y(y), self.normalize_t(t)
        raise TypeError("normalize(field, values) or normalize(x, y, t)")

    def denormalize(self, *args: Any) -> Any:
        """``denormalize(field, values)`` or ``denormalize(xh, yh, th)`` tuple form."""
        if len(args) == 2 and isinstance(args[0], str):
            fn = getattr(self, self._FIELD_DENORM[args[0]])
            return fn(args[1])
        if len(args) == 3:
            xh, yh, th = args
            return self.denormalize_x(xh), self.denormalize_y(yh), self.denormalize_t(th)
        raise TypeError("denormalize(field, values) or denormalize(xh, yh, th)")

    def encode(self, x: Any, y: Any, t: Any) -> tuple[Any, Any, Any]:
        """Physical ``(x, y, t)`` -> normalized ``(x_hat, y_hat, t_hat)``."""
        return self.normalize_x(x), self.normalize_y(y), self.normalize_t(t)

    def decode(self, x_hat: Any, y_hat: Any, t_hat: Any) -> tuple[Any, Any, Any]:
        """Normalized ``(x_hat, y_hat, t_hat)`` -> physical ``(x, y, t)``."""
        return self.denormalize_x(x_hat), self.denormalize_y(y_hat), self.denormalize_t(t_hat)

    def forward(self, x: Any, y: Any, t: Any) -> tuple[Any, Any, Any]:
        """Alias of :meth:`encode` (physical -> ``[-1, 1]``)."""
        return self.encode(x, y, t)

    def inverse(self, x_hat: Any, y_hat: Any, t_hat: Any) -> tuple[Any, Any, Any]:
        """Alias of :meth:`decode` (``[-1, 1]`` -> physical)."""
        return self.decode(x_hat, y_hat, t_hat)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON-serializable state (bounds only; no hidden scaling)."""
        return {
            "x_bounds": list(self.x_bounds),
            "y_bounds": list(self.y_bounds),
            "t_bounds": list(self.t_bounds),
            "C_bounds": list(self.C_bounds) if self.C_bounds is not None else None,
            "s_bounds": list(self.s_bounds) if self.s_bounds is not None else None,
            "c_scale": self.c_scale,
            "c_shift": self.c_shift,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Normalizer":
        def _t(v: Any) -> tuple[float, float] | None:
            if v is None:
                return None
            return (float(v[0]), float(v[1]))

        # Prefer explicit C_bounds; fall back to shift/scale form.
        if d.get("C_bounds") is not None:
            return cls(
                x_bounds=_t(d["x_bounds"]),  # type: ignore[arg-type]
                y_bounds=_t(d["y_bounds"]),  # type: ignore[arg-type]
                t_bounds=_t(d["t_bounds"]),  # type: ignore[arg-type]
                C_bounds=_t(d.get("C_bounds")),
                s_bounds=_t(d.get("s_bounds")),
            )
        xr = d.get("x_bounds", d.get("x_range"))
        yr = d.get("y_bounds", d.get("y_range"))
        tr = d.get("t_bounds", d.get("t_range"))
        return cls(
            x_bounds=_t(xr),  # type: ignore[arg-type]
            y_bounds=_t(yr),  # type: ignore[arg-type]
            t_bounds=_t(tr),  # type: ignore[arg-type]
            s_bounds=_t(d.get("s_bounds")),
            c_scale=float(d.get("c_scale", 1.0)),
            c_shift=float(d.get("c_shift", 0.0)),
        )

    def state_dict(self) -> dict[str, Any]:
        """Alias of :meth:`to_dict` (older call sites)."""
        d = self.to_dict()
        d["x_range"] = d["x_bounds"]
        d["y_range"] = d["y_bounds"]
        d["t_range"] = d["t_bounds"]
        return d

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Normalizer":
        """Alias of :meth:`from_dict` (older call sites)."""
        return cls.from_dict(state)

    def save(self, path: str | Path) -> Path:
        """Write JSON to ``path`` (creates parent dirs)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        """Read JSON written by :meth:`save`."""
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Normalizer(x={self.x_bounds}, y={self.y_bounds}, t={self.t_bounds}, "
            f"C={self.C_bounds}, s={self.s_bounds})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Normalizer):
            return NotImplemented
        return self.to_dict() == other.to_dict()
