"""Losses (PRD §§2.5, 4.3 `train/losses.py`).

``L_total = w_data·L_data + w_pde·L_pde + w_bc·L_bc + w_reg·L_reg``

* ``L_data`` — **masked** MSE at stations with valid observations.
  Mask, never impute (missing points contribute exactly zero value and
  zero gradient).
* ``L_pde`` — MSE of the PDE residual at interior collocation points.
* ``L_bc`` — inflow Dirichlet + outflow zero-gradient residuals.
* ``L_reg`` — L1/TV on ``S`` + temporal smoothness on ``C_b``.

Each function returns a scalar tensor (grad-carrying when inputs require
grad) so the GradNorm balancer can take per-term norms. All are CPU-safe.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "masked_mse",
    "masked_mae",
    "pde_loss",
    "bc_loss",
    "source_l1",
    "l1_source",
    "source_tv",
    "tv_source",
    "cb_temporal_smooth",
    "smoothness_time",
    "positivity_penalty",
    "total_loss",
]


def _zero_like(ref: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((), dtype=ref.dtype if ref.is_floating_point() else torch.float32,
                    device=ref.device)
    if ref.requires_grad:
        # keep the graph alive so balancer grads are well-defined (zero)
        z = z + (ref * 0.0).sum() * 0.0
    return z


def _float_tensor(x: torch.Tensor, like: torch.Tensor | None = None) -> torch.Tensor:
    """``as_tensor`` preserving float dtype (float64 PINN runs stay float64).

    Non-floating inputs become float32. When ``like`` is given, match its
    floating dtype instead.
    """
    t = torch.as_tensor(x)
    if like is not None and torch.is_tensor(like) and like.is_floating_point():
        return t.to(dtype=like.dtype, device=like.device) if (t.dtype != like.dtype or t.device != like.device) else t
    if not t.is_floating_point():
        t = t.to(dtype=torch.float32)
    return t


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked MSE over valid observations only.

    ``mask`` is bool (True = valid) or float weights, broadcastable to
    ``pred``. ``None`` reduces to plain MSE. All-masked returns a grad-safe
    zero (missingness contributes exactly zero gradient). Computed in the
    input dtype so mixed-precision loss trajectories stay consistent.
    """
    pred = torch.as_tensor(pred)
    if not pred.is_floating_point():
        pred = pred.to(dtype=torch.float32)
    target = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
    if mask is None:
        return F.mse_loss(pred, target)
    m = torch.as_tensor(mask, device=pred.device).to(dtype=pred.dtype)
    m, _, _ = torch.broadcast_tensors(m, pred, target)
    denom = m.sum()
    if float(denom.item()) == 0.0:
        return _zero_like(pred)
    diff = (pred - target) * m
    return (diff * diff).sum() / denom


def masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked MAE over valid observations only (``mask=None`` = plain MAE)."""
    pred = torch.as_tensor(pred)
    if not pred.is_floating_point():
        pred = pred.to(dtype=torch.float32)
    target = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
    if mask is None:
        return F.l1_loss(pred, target)
    m = torch.as_tensor(mask, device=pred.device).to(dtype=pred.dtype)
    m, _, _ = torch.broadcast_tensors(m, pred, target)
    denom = m.sum()
    if float(denom.item()) == 0.0:
        return _zero_like(pred)
    diff = ((pred - target).abs() * m)
    return diff.sum() / denom


def pde_loss(residual: torch.Tensor) -> torch.Tensor:
    """MSE of the PDE residual ``r = dC/dt + div(uC) - div(K grad C) - S + lam C``."""
    r = _float_tensor(residual).reshape(-1)
    if r.numel() == 0:
        return _zero_like(r)
    return torch.mean(r * r)


def bc_loss(
    C_pred: torch.Tensor,
    Cb_pred: torch.Tensor,
    dCdn: torch.Tensor | None,
    inflow_mask: torch.Tensor | None,
    outflow_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Wind-switched boundary loss.

    * Inflow (``u·n < 0``): Dirichlet ``mean((C - C_b)^2)``.
    * Outflow (``u·n >= 0``): natural ``mean((dC/dn)^2)``.

    Empty sides contribute zero. A mask of ``None`` means that side is
    empty (not "all points"): passing ``None`` for **both** sides raises
    ``ValueError`` because ``0.5*(dirichlet+neumann)`` over all points is
    unphysical double-counting — callers must pass explicit wind-switched
    masks (see :mod:`bapinnsformer.physics.boundary`).
    ``dCdn`` may be ``None`` only when there are no outflow points.
    """
    if inflow_mask is None and outflow_mask is None:
        raise ValueError(
            "bc_loss: both masks are None — pass explicit inflow/outflow masks "
            "(None now means 'empty side', not 'all points')"
        )
    C_pred = _float_tensor(C_pred).reshape(-1)
    Cb_pred = _float_tensor(Cb_pred).reshape(-1)
    total = _zero_like(C_pred)
    n_terms = 0

    def _mask(m, n, *, empty: bool):
        if m is None:
            assert empty  # one side is always explicit here
            return torch.zeros(n, dtype=torch.bool, device=C_pred.device)
        mm = torch.as_tensor(m, device=C_pred.device).reshape(-1)
        return mm.to(dtype=torch.bool) if mm.dtype == torch.bool else mm > 0.5

    inflow = _mask(inflow_mask, C_pred.numel(), empty=True)
    if bool(inflow.any().item()):
        total = total + torch.mean((C_pred[inflow] - Cb_pred[inflow]) ** 2)
        n_terms += 1
    outflow = _mask(outflow_mask, C_pred.numel(), empty=True)
    if bool(outflow.any().item()):
        if dCdn is None:
            raise ValueError("dCdn required when outflow points are present")
        g = _float_tensor(dCdn, like=C_pred).reshape(-1)
        total = total + torch.mean(g[outflow] ** 2)
        n_terms += 1
    if n_terms == 2:
        total = total * 0.5
    return total


def source_l1(S: torch.Tensor) -> torch.Tensor:
    """L1 sparsity prior on the interior source field ``S``."""
    S = _float_tensor(S).reshape(-1)
    if S.numel() == 0:
        return _zero_like(S)
    return torch.mean(torch.abs(S))


def l1_source(S: torch.Tensor) -> torch.Tensor:
    """Alias of :func:`source_l1` (older call sites)."""
    return source_l1(S)


def source_tv(S: torch.Tensor) -> torch.Tensor:
    """Total-variation (first-difference) penalty on ``S`` along dim 0."""
    S = _float_tensor(S).reshape(-1)
    if S.numel() < 2:
        return _zero_like(S)
    return torch.mean(torch.abs(S[1:] - S[:-1]))


def tv_source(S_xy: torch.Tensor) -> torch.Tensor:
    """Isotropic 2-D TV for field snapshots; 1-D inputs use :func:`source_tv`."""
    s = _float_tensor(S_xy)
    if s.dim() < 2:
        return source_tv(s)
    dx = s[1:, :] - s[:-1, :]
    dy = s[:, 1:] - s[:, :-1]
    return dx.abs().mean() + dy.abs().mean()


def cb_temporal_smooth(Cb_time: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Temporal smoothness on ``C_b`` (first/second differences, MSE)."""
    Cb = _float_tensor(Cb_time).reshape(-1)
    if Cb.numel() < (int(order) + 1):
        return _zero_like(Cb)
    d = Cb
    for _ in range(int(order)):
        d = d[1:] - d[:-1]
    return torch.mean(d * d)


def smoothness_time(series: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Alias of :func:`cb_temporal_smooth` (older call sites)."""
    return cb_temporal_smooth(series, order=order)


def positivity_penalty(x: torch.Tensor) -> torch.Tensor:
    """``mean(relu(-x)^2)`` — safety net when a head is linear."""
    x = _float_tensor(x)
    neg = F.relu(-x)
    return torch.mean(neg * neg)


def total_loss(terms: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    """Weighted sum of named loss terms (missing weight ⇒ 1.0)."""
    total = None
    for name, value in terms.items():
        v = value if isinstance(value, torch.Tensor) else torch.as_tensor(
            value, dtype=torch.float32)
        w = float(weights.get(name, 1.0))
        total = w * v if total is None else total + w * v
    if total is None:
        raise ValueError("no loss terms provided")
    return total
