"""Autodiff operators + advection–diffusion–deposition residual.

PRD §4.3 `physics/operators.py`: gradient/divergence/Laplacian helpers
over network outputs; the residual of
``dC/dt + div(uC) = div(K grad C) + S - lam*C`` assembled from them.

Symbols: ``C`` concentration, ``u, v`` known wind, ``S`` source,
``K`` diffusivity, ``lam`` deposition rate.

Dual API (merged for cross-plane compat):
- Field API (X-based): ``grad_scalar(y, X)``, ``divergence_2d(Fx, Fy,
  X)``, ``laplacian(y, X)``, ``pde_residual_fields(C, X, U, S, K,
  lam)`` — used by model-plane tests.
- Callable API: ``gradient(C_fn, x, y, t)``,
  ``pde_residual(C_fn, x, y, t, u, v, K, lam, S_fn)`` — required task
  contract. ``divergence_2d``/``laplacian`` dispatch on first-arg type.
Torch is imported lazily inside functions so module import is GPU-free.
"""

from __future__ import annotations

from collections.abc import Callable


def _torch():
    try:
        import torch as _t
    except Exception as exc:
        raise ImportError("physics.operators requires torch") from exc
    return _t


def grad_scalar(y, X, create_graph: bool = True):
    """Gradient of scalar field ``y`` (``(N,)``) w.r.t. ``X`` (``(N, d)``)."""
    torch = _torch()
    if not X.requires_grad:
        raise ValueError("X must have requires_grad=True")
    ones = torch.ones_like(y.reshape(-1))
    g = torch.autograd.grad(y.reshape(-1), X, grad_outputs=ones,
                            create_graph=create_graph, retain_graph=True)[0]
    return g


def jacobian_diag(y, X):
    """Per-output-component gradients for ``y: (N, m)`` → ``(N, m, d)``."""
    cols = []
    for j in range(y.shape[1]):
        cols.append(grad_scalar(y[:, j], X, create_graph=True).unsqueeze(1))
    torch = _torch()
    return torch.cat(cols, dim=1)


def _as_leaf(x):
    torch = _torch()
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)
    return x.detach().clone().requires_grad_(True)


def gradient(
    C_fn: Callable,
    x,
    y,
    t,
    create_graph: bool = True,
):
    """First-order space–time gradient ``(dC/dx, dC/dy, dC/dt)`` via autograd.

    ``C_fn`` maps ``(x, y, t)`` → ``C`` elementwise. Inputs are cloned
    to leaves so caller tensors are never mutated.
    """
    torch = _torch()
    xl, yl, tl = _as_leaf(x), _as_leaf(y), _as_leaf(t)
    C = C_fn(xl, yl, tl)
    if not isinstance(C, torch.Tensor):
        C = torch.as_tensor(C, dtype=xl.dtype, device=xl.device)
    C = C.reshape(xl.shape)
    ones = torch.ones_like(C)
    return torch.autograd.grad(C, (xl, yl, tl), grad_outputs=ones,
                               create_graph=create_graph, retain_graph=True, allow_unused=False)


def divergence_2d(*args, **kwargs):
    """``div(F)`` — dispatches on call pattern.

    - Field API: ``divergence_2d(Fx, Fy, X)`` → ``dFx/dx + dFy/dy``.
    - Callable API: ``divergence_2d(Fx_fn, Fy_fn, x, y, t)`` with
      ``Fx_fn(x, y, t)`` callables.
    """
    torch = _torch()
    if len(args) == 3 and not callable(args[0]):
        Fx, Fy, X = args
        gx = grad_scalar(Fx, X)
        gy = grad_scalar(Fy, X)
        return gx[:, 0] + gy[:, 1]
    if len(args) == 5 and callable(args[0]):
        Fx_fn, Fy_fn, x, y, t = args
        xl, yl, tl = _as_leaf(x), _as_leaf(y), _as_leaf(t)
        Fx = Fx_fn(xl, yl, tl).reshape(xl.shape)
        Fy = Fy_fn(xl, yl, tl).reshape(xl.shape)
        ones = torch.ones_like(Fx)
        dFx = torch.autograd.grad(Fx, xl, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        dFy = torch.autograd.grad(Fy, yl, grad_outputs=torch.ones_like(Fy), create_graph=True, retain_graph=True)[0]
        return dFx + dFy
    raise TypeError("divergence_2d: use (Fx, Fy, X) or (Fx_fn, Fy_fn, x, y, t)")


def _second_grad(first: object, leaf: object) -> object:
    """Safe 2nd-order grad: zeros when first-order grad is constant."""
    torch = _torch()
    if not isinstance(first, torch.Tensor):
        return torch.zeros_like(leaf)
    if not (getattr(first, "requires_grad", False) or getattr(first, "grad_fn", None) is not None):
        return torch.zeros_like(first)
    g = torch.autograd.grad(first, leaf, grad_outputs=torch.ones_like(first),
                            create_graph=False, retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(first) if g is None else g


def laplacian(*args, **kwargs):
    """Laplacian — dispatches on call pattern.

    - Field API: ``laplacian(y, X, n_spatial=2)``.
    - Callable API: ``laplacian(C_fn, x, y, t)`` → ``Cxx + Cyy``.
    """
    torch = _torch()
    n_spatial = kwargs.get("n_spatial", 2)
    if len(args) >= 1 and callable(args[0]):
        C_fn, x, y, t = args[0], args[1], args[2], args[3]
        xl, yl, tl = _as_leaf(x), _as_leaf(y), _as_leaf(t)
        C = C_fn(xl, yl, tl)
        if not isinstance(C, torch.Tensor):
            C = torch.as_tensor(C, dtype=xl.dtype, device=xl.device)
        C = C.reshape(xl.shape)
        ones = torch.ones_like(C)
        Cx, Cy, _ = torch.autograd.grad(C, (xl, yl, tl), grad_outputs=ones, create_graph=True, retain_graph=True)
        Cxx = _second_grad(Cx, xl)
        Cyy = _second_grad(Cy, yl)
        return Cxx + Cyy
    y, X = args[0], args[1]
    if len(args) >= 3:
        n_spatial = int(args[2])
    g = grad_scalar(y, X, create_graph=True)
    torch_ = torch
    lap = torch_.zeros_like(g[:, 0])
    for i in range(n_spatial):
        gi = g[:, i]
        ones = torch_.ones_like(gi)
        d2 = torch_.autograd.grad(gi, X, grad_outputs=ones, create_graph=True, retain_graph=True, allow_unused=True)[0]
        d2 = torch_.zeros_like(gi) if d2 is None else d2[:, i]
        lap = lap + d2
    return lap


def advection_diffusion_deposition_residual(dCdt, dCdx, dCdy, lap_C, C, u, v, S, K, lam, div_u=0.0):
    """Assemble ``dC/dt + u·∇C + C·∇·u − K·ΔC − S + λC`` (broadcastable)."""
    return dCdt + u * dCdx + v * dCdy + C * div_u - K * lap_C - S + lam * C


def pde_residual_fields(C, X, U, S, K, lam, div_u=0.0):
    """Residual from sampled fields at collocation points.

    Args:
        C: ``(N,)`` concentration (function of ``X`` via autograd graph).
        X: ``(N, 3)`` ``(x, y, t)`` with ``requires_grad=True``.
        U: ``(N, 2)`` wind ``(u, v)`` (known data: no grad).
        S: ``(N,)`` source values. ``K``, ``lam``: scalars.
    """
    g = grad_scalar(C, X, create_graph=True)
    dCdx, dCdy, dCdt = g[:, 0], g[:, 1], g[:, 2]
    lap = laplacian(C, X, n_spatial=2)
    u, v = U[:, 0], U[:, 1]
    return advection_diffusion_deposition_residual(
        dCdt, dCdx, dCdy, lap, C.reshape(-1), u, v, S.reshape(-1), K, lam, div_u=div_u)


def pde_residual(C_fn, x, y, t, u, v, K, lam, S_fn=None):
    """PDE residual ``r = dC/dt + div(uC) − K·lap(C) − S + lam·C``.

    Wind ``(u, v)`` is locally constant per collocation point
    (ERA5-interpolated), so ``div(uC) = u·Cx + v·Cy``. ``K``/``lam``
    accept scalars or broadcastable tensors. ``S_fn=None`` → zero
    source. Returns elementwise residuals with the broadcast shape.
    """
    torch = _torch()
    xl, yl, tl = _as_leaf(x), _as_leaf(y), _as_leaf(t)
    C = C_fn(xl, yl, tl)
    if not isinstance(C, torch.Tensor):
        C = torch.as_tensor(C, dtype=xl.dtype, device=xl.device)
    C = C.reshape(xl.shape)
    ones = torch.ones_like(C)
    Cx, Cy, Ct = torch.autograd.grad(C, (xl, yl, tl), grad_outputs=ones, create_graph=True, retain_graph=True)
    Cxx = _second_grad(Cx, xl)
    Cyy = _second_grad(Cy, yl)

    def _b(val):
        return val if isinstance(val, torch.Tensor) else torch.as_tensor(float(val), dtype=C.dtype, device=C.device)

    uu, vv, Kk, ll = _b(u), _b(v), _b(K), _b(lam)
    S = S_fn(xl, yl, tl).reshape(xl.shape) if S_fn is not None else torch.zeros_like(C)
    if not isinstance(S, torch.Tensor):
        S = torch.as_tensor(S, dtype=C.dtype, device=C.device)
    return Ct + uu * Cx + vv * Cy - Kk * (Cxx + Cyy) - S + ll * C


# Backwards-compatible alias: `grad` for the field API.
grad = grad_scalar

__all__ = [
    "grad_scalar",
    "grad",
    "gradient",
    "jacobian_diag",
    "divergence_2d",
    "laplacian",
    "advection_diffusion_deposition_residual",
    "pde_residual_fields",
    "pde_residual",
]
