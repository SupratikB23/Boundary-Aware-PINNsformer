"""Dtype helpers: preserve floating precision across the codebase.

PINN practitioners often run float64. Every tensor-constructing call site
must preserve the incoming float dtype instead of hardcoding float32;
otherwise ``F.linear`` raises ``mat1 and mat2 must have the same dtype``.
"""

from __future__ import annotations

from typing import Any


def infer_float_dtype(*values: Any):
    """First floating tensor dtype found, else the torch default dtype."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None
    for v in values:
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            return v.dtype
    return torch.get_default_dtype()


def as_float_tensor(x: Any, like: Any = None, device=None):
    """``as_tensor`` preserving float dtype (non-floats → float32).

    When ``like`` (tensor or sequence of tensors) is given, match the
    first floating dtype found there.
    """
    import torch

    if like is not None:
        seq = like if isinstance(like, (list, tuple)) else (like,)
        dt = infer_float_dtype(*seq)
    else:
        dt = infer_float_dtype(x)
    t = torch.as_tensor(x, dtype=dt) if not isinstance(x, torch.Tensor) else x.to(dtype=dt)
    return t.to(device=device) if device is not None else t


__all__ = ["infer_float_dtype", "as_float_tensor"]
