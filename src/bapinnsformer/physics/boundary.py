"""Wind-switched boundary conditions with hysteresis.

PRD §2.4 + §4.3 `physics/boundary.py`: at each boundary collocation
point ``(s, t)`` with outward normal ``n̂(s)`` and wind ``u``,

- ``u·n̂ < 0`` (inflow)  ⇒ Dirichlet residual ``C − C_b``,
- ``u·n̂ ≥ 0`` (outflow) ⇒ natural residual ``dC/dn``.

The switch is wind-determined; a small hysteresis band ``|u·n̂| < eps``
suppresses gradient chatter for near-tangential flow. Works on numpy
arrays and torch tensors (branch-free masks so autograd flows).
"""

from __future__ import annotations


def _is_torch(x) -> bool:
    try:
        import torch

        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


def normal_flux(wind_u, wind_v, normals):
    """Outward normal wind component ``q = u·n̂`` (vectorized over points)."""
    if _is_torch(wind_u) or _is_torch(wind_v) or _is_torch(normals):
        import torch

        wu = wind_u if isinstance(wind_u, torch.Tensor) else torch.as_tensor(wind_u, dtype=torch.float64)
        wv = wind_v if isinstance(wind_v, torch.Tensor) else torch.as_tensor(wind_v, dtype=torch.float64)
        n = normals if isinstance(normals, torch.Tensor) else torch.as_tensor(normals, dtype=torch.float64)
        import numpy as _np

        if not isinstance(normals, torch.Tensor):
            n = torch.as_tensor(_np.asarray(normals, dtype=float), dtype=torch.float64)
        return wu.reshape(-1) * n[:, 0] + wv.reshape(-1) * n[:, 1]
    import numpy as np

    wu = np.asarray(wind_u, dtype=float).ravel()
    wv = np.asarray(wind_v, dtype=float).ravel()
    n = np.asarray(normals, dtype=float)
    return wu * n[:, 0] + wv * n[:, 1]


def partition(u_dot_n, hyst: float = 0.0):
    """Return ``(inflow_mask, outflow_mask)`` boolean arrays/tensors."""
    h = abs(float(hyst))
    try:
        import torch

        if isinstance(u_dot_n, torch.Tensor):
            return (u_dot_n < -h), (u_dot_n > h)
    except ImportError:
        pass
    try:
        import numpy as np

        u = np.asarray(u_dot_n)
        return (u < -h), (u > h)
    except ImportError:
        u = list(u_dot_n)
        return ([v < -h for v in u], [v > h for v in u])


def partition_boundary(s_pts, t, wind_u, wind_v, normals, hysteresis: float = 0.05):
    """Partition boundary points into inflow / outflow / buffer masks.

    Args:
        s_pts, t: arc-length and time coords (returned untouched).
        wind_u, wind_v: collocated wind components.
        normals: ``(N, 2)`` outward normals.
        hysteresis: half-width ``eps`` (m/s) of the dead band around
            ``q = 0``; ``q < −eps`` inflow, ``q > +eps`` outflow.

    Returns:
        ``{"s", "t", "q", "inflow", "outflow", "buffer"}`` with boolean
        masks (torch or numpy matching the input flavor).
    """
    q = normal_flux(wind_u, wind_v, normals)
    eps = float(hysteresis)
    try:
        import torch

        if isinstance(q, torch.Tensor):
            inflow = q < -eps
            outflow = q > eps
            return {"s": s_pts, "t": t, "q": q, "inflow": inflow, "outflow": outflow, "buffer": ~(inflow | outflow)}
    except ImportError:
        pass
    import numpy as np

    q = np.asarray(q)
    inflow = q < -eps
    outflow = q > eps
    return {"s": s_pts, "t": t, "q": q, "inflow": inflow, "outflow": outflow, "buffer": ~(inflow | outflow)}


def dirichlet_residual(C_pred, Cb_pred=None, inflow_mask=None):
    """Inflow residual — dual signature.

    - Task API: ``dirichlet_residual(C_pred, Cb_pred)`` → ``C − C_b``.
    - Legacy API: ``dirichlet_residual(C, Cb, inflow_mask)`` → masked.
    """
    if Cb_pred is None:
        raise TypeError("dirichlet_residual: need (C, Cb) with optional mask")
    if inflow_mask is None:
        # Task contract: elementwise C − C_b.
        if _is_torch(C_pred) or _is_torch(Cb_pred):
            import torch

            Ct = C_pred if isinstance(C_pred, torch.Tensor) else torch.as_tensor(C_pred, dtype=torch.float64)
            Bt = Cb_pred if isinstance(Cb_pred, torch.Tensor) else torch.as_tensor(Cb_pred, dtype=Ct.dtype, device=Ct.device)
            return Ct.reshape(-1) - Bt.reshape(-1)
        import numpy as np

        return np.asarray(C_pred, dtype=float).ravel() - np.asarray(Cb_pred, dtype=float).ravel()
    # Legacy masked path.
    try:
        import torch

        if isinstance(C_pred, torch.Tensor) or isinstance(Cb_pred, torch.Tensor):
            C_t = C_pred if isinstance(C_pred, torch.Tensor) else torch.as_tensor(C_pred, dtype=Cb_pred.dtype)
            Cb_t = Cb_pred if isinstance(Cb_pred, torch.Tensor) else torch.as_tensor(Cb_pred, dtype=C_t.dtype)
            m = inflow_mask if isinstance(inflow_mask, torch.Tensor) else torch.as_tensor(list(inflow_mask), dtype=torch.bool)
            return (C_t - Cb_t) * m.to(C_t.dtype)
    except ImportError:
        pass
    import numpy as np

    m = np.asarray(list(inflow_mask), dtype=float)
    return (np.asarray(C_pred, dtype=float) - np.asarray(Cb_pred, dtype=float)) * m


def neumann_residual(dCdn, outflow_mask=None):
    """Outflow natural residual — dual signature.

    - Task API: ``neumann_residual(dCdn)`` → ``dC/dn`` (target zero).
    - Legacy API: ``neumann_residual(dCdn, outflow_mask)`` → masked.
    """
    if outflow_mask is None:
        if _is_torch(dCdn):
            return dCdn.reshape(-1)
        import numpy as np

        return np.asarray(dCdn, dtype=float).ravel()
    try:
        import torch

        if isinstance(dCdn, torch.Tensor):
            m = outflow_mask if isinstance(outflow_mask, torch.Tensor) else torch.as_tensor(list(outflow_mask), dtype=torch.bool)
            return dCdn * m.to(dCdn.dtype)
    except ImportError:
        pass
    import numpy as np

    m = np.asarray(list(outflow_mask), dtype=float)
    return np.asarray(dCdn, dtype=float) * m


def normal_gradient(C_fn, x, y, t, normals_xy):
    """Directional derivative ``dC/dn = ∇C·n̂`` at boundary points."""
    from .operators import gradient as _grad

    Cx, Cy, _ = _grad(C_fn, x, y, t, create_graph=True)
    if _is_torch(Cx) or _is_torch(normals_xy):
        import torch

        n = normals_xy if isinstance(normals_xy, torch.Tensor) else torch.as_tensor(normals_xy, dtype=Cx.dtype, device=Cx.device)
        return Cx.reshape(-1) * n[:, 0] + Cy.reshape(-1) * n[:, 1]
    import numpy as np

    n = np.asarray(normals_xy, dtype=float)
    import numpy as _np

    return _np.asarray(Cx).ravel() * n[:, 0] + _np.asarray(Cy).ravel() * n[:, 1]


def boundary_loss(C, Cb, dCdn, u_dot_n, hyst: float = 0.0):
    """Mean-square wind-switched boundary loss (scalar).

    Returns a grad-carrying 0-d tensor when any input is a torch tensor
    (training path, compatible with :func:`train.losses.bc_loss`), else
    a plain float (numpy/analysis path).
    """
    inflow, outflow = partition(u_dot_n, hyst)
    rd = dirichlet_residual(C, Cb, inflow)
    rn = neumann_residual(dCdn, outflow)
    try:
        import torch

        if isinstance(rd, torch.Tensor):
            loss = rd.pow(2).mean() + rn.pow(2).mean()
            return loss if loss.requires_grad else loss.detach()
    except ImportError:
        pass
    import numpy as np

    return float(np.mean(np.asarray(rd) ** 2) + np.mean(np.asarray(rn) ** 2))


__all__ = [
    "normal_flux",
    "partition",
    "partition_boundary",
    "dirichlet_residual",
    "neumann_residual",
    "normal_gradient",
    "boundary_loss",
]
