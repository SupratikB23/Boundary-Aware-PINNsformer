"""Domain mass-balance diagnostic.

PRD §4.3 `physics/mass_balance.py`: integrated inflow, outflow, source,
deposition and storage change over the rectangular domain. Used as a
physical-consistency metric and as a paper figure (closure residual
should be ≈ 0 for a converged, conservative solution). Numpy-only.
"""

from __future__ import annotations

import numpy as np


def mass_balance(
    C: np.ndarray,
    C_prev: np.ndarray | None,
    u: np.ndarray,
    v: np.ndarray,
    S: np.ndarray | None,
    lam: float,
    dx: float,
    dy: float,
    dt: float,
) -> dict[str, float]:
    """Compute one-step mass budget terms (arbitrary mass units).

    Control-volume accounting on a ``(Ny, Nx)`` grid:

    - ``storage``: ``Σ(C − C_prev)·dx·dy / dt`` (rate; 0 if no prev).
    - ``inflow`` / ``outflow``: boundary flux ``Σ −/+ (u·n̂)·C·dl``
      decomposed by sign of the outward normal component, positive
      magnitudes (rate).
    - ``source``: ``ΣS·dx·dy`` (rate).
    - ``deposition``: ``ΣλC·dx·dy`` (rate, positive = removal).
    - ``closure``: ``storage − (inflow − outflow + source −
      deposition)`` (≈ 0 when closed; diffusion folds into closure).

    Returns dict with ``mass, storage, inflow, outflow, net_advective,
    source, deposition, closure``.
    """
    C = np.asarray(C, dtype=float)
    U = np.asarray(u, dtype=float)
    V = np.asarray(v, dtype=float)
    if C.ndim != 2 or U.shape != C.shape or V.shape != C.shape:
        raise ValueError("mass_balance: C/u/v must share (Ny, Nx) shape")
    if dx <= 0 or dy <= 0 or dt <= 0:
        raise ValueError("mass_balance: dx, dy, dt must be positive")
    if lam < 0:
        raise ValueError("mass_balance: lam must be non-negative")
    cell = dx * dy
    mass = float((C * cell).sum())
    storage = float(((C - np.asarray(C_prev, dtype=float)) * cell).sum() / dt) if C_prev is not None else 0.0
    south_q = -V[0, :] * C[0, :] * dx
    north_q = V[-1, :] * C[-1, :] * dx
    west_q = -U[:, 0] * C[:, 0] * dy
    east_q = U[:, -1] * C[:, -1] * dy
    q_all = np.concatenate([south_q, north_q, west_q, east_q])
    inflow = float(-q_all[q_all < 0].sum())
    outflow = float(q_all[q_all > 0].sum())
    Sarr = np.zeros_like(C) if S is None else np.asarray(S, dtype=float)
    source = float((Sarr * cell).sum())
    deposition = float((lam * C * cell).sum())
    net_advective = inflow - outflow
    closure = storage - (net_advective + source - deposition)
    return {
        "mass": mass,
        "storage": storage,
        "inflow": inflow,
        "outflow": outflow,
        "net_advective": float(net_advective),
        "source": source,
        "deposition": deposition,
        "closure": float(closure),
    }


def budget(
    C_now: np.ndarray,
    C_prev: np.ndarray,
    Q_in: float,
    Q_out: float,
    Q_S: float,
    Q_dep: float,
    dx: float,
    dy: float,
    dt: float,
) -> dict[str, float]:
    """Assemble one-step mass budget and closure residual (legacy API).

    Thin wrapper around ``eval.attribution.mass_closure`` with a local
    fallback so training diagnostics never hard-depend on `eval/`.
    """
    storage = float((np.asarray(C_now, dtype=float) - np.asarray(C_prev, dtype=float)).sum() * dx * dy)
    try:
        from ..eval.attribution import mass_closure

        return mass_closure(
            M_in=float(Q_in) * dt,
            M_S=float(Q_S) * dt,
            M_out=float(Q_out) * dt,
            M_dep=float(Q_dep) * dt,
            dM_storage=storage,
        )
    except Exception:
        inflow = float(Q_in) * dt + float(Q_S) * dt
        outflow = float(Q_out) * dt + float(Q_dep) * dt + storage
        resid = inflow - outflow
        scale = abs(inflow) + abs(outflow)
        return {
            "residual": resid,
            "rel_residual": (resid / scale) if scale > 0 else float("nan"),
            "M_in": float(Q_in) * dt,
            "M_S": float(Q_S) * dt,
            "M_out": float(Q_out) * dt,
            "M_dep": float(Q_dep) * dt,
            "dM_storage": storage,
        }


__all__ = ["mass_balance", "budget"]
