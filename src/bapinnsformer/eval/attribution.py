"""Transboundary attribution accounting (PRD §4.3 `eval/attribution.py`).

Headline-number module: converts the recovered boundary inflow field
``C_b(s, t)`` into the emission-inventory-free **transboundary
contribution share** per season/pollutant, with sector decomposition
(NW/N/E/S/W). Every rule is explicit so the number is auditable.

Accounting (discrete, per season window)
---------------------------------------
- Boundary nodes ``i`` with arc-length weights ``ds[i]`` (metres),
  outward unit normals ``n_hat[i, :]`` and wind ``u[i, t, :]``.
- Inflow mask: ``u·n̂ < 0`` (PRD §2.4; hysteresis optional).
- Advective inflow rate at time ``t``::

      Q_in(t) = Σ_{i ∈ inflow} (−u·n̂)[i,t] · C_b[i,t] · ds[i]   [mass/time]

  (Concentration × normal inflow speed × face width.)
- Interior source rate:: ``Q_S(t) = Σ_cells S · dx · dy``.
- Time-integrated masses ``M_in = Σ_t Q_in Δt``, ``M_S = Σ_t Q_S Δt``.
- **Transboundary share** ``= M_in / (M_in + M_S)``.

Deposition/loss (``λ``) and storage change are *diagnostics*, not part
of the share denominator: the share answers "of the particulate mass
entering/created in the domain, what fraction crossed the lateral
boundary?" — see :func:`mass_closure` for the full balance check.

Sector partition (compass bearing of segment midpoint rel. centre)
-----------------------------------------------------------------
Bearing ``b = (90 − atan2(dy, dx)) mod 360`` (clockwise from north)::

    NW : [292.5, 337.5)   (45° octant, the crop-burning inflow sector)
    N  : [337.5, 360) ∪ [0, 45)   (67.5°)
    E  : [45, 135)        (90°)
    S  : [135, 225)       (90°)
    W  : [225, 292.5)     (67.5°)

The five sectors partition 360° exactly, so sector inflow masses sum
to the total inflow mass (asserted in tests).
"""

from __future__ import annotations

import math

SECTORS = ("NW", "N", "E", "S", "W")

# (sector, lo_inclusive_deg, hi_exclusive_deg); N wraps around 360.
_SECTOR_EDGES = {
    "NW": [(292.5, 337.5)],
    "N": [(337.5, 360.0), (0.0, 45.0)],
    "E": [(45.0, 135.0)],
    "S": [(135.0, 225.0)],
    "W": [(225.0, 292.5)],
}


def bearing_deg(dx: float, dy: float) -> float:
    """Compass bearing (deg clockwise from north) of vector (dx, dy)."""
    return (90.0 - math.degrees(math.atan2(dy, dx))) % 360.0


def sector_of_bearing(b: float) -> str:
    """Map a bearing in [0, 360) to one of NW/N/E/S/W."""
    b = float(b) % 360.0
    for sector, spans in _SECTOR_EDGES.items():
        for lo, hi in spans:
            if lo <= b < hi:
                return sector
    raise AssertionError(f"bearing {b} fell through sector partition")


def sector_of_segment(dx: float, dy: float) -> str:
    """Sector of a boundary segment midpoint relative to domain centre."""
    return sector_of_bearing(bearing_deg(dx, dy))


def _as_float_list(a, name: str) -> list[float]:
    try:
        import torch

        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().tolist()
    except ImportError:
        pass
    try:
        import numpy as _np

        if isinstance(a, _np.ndarray):
            return [float(v) for v in a.ravel().tolist()]
    except ImportError:
        pass
    if isinstance(a, (int, float)):
        raise ValueError(f"{name} must be array-like")
    return [float(v) for v in a]


def inflow_flux(Cb_st, u_dot_n_st, ds_s) -> list[float]:
    """Per-timestep advective inflow rate ``Q_in(t)`` (mass/time).

    Args:
        Cb_st: ``(n_s, n_t)`` boundary concentration.
        u_dot_n_st: ``(n_s, n_t)`` normal wind (negative = inflow).
        ds_s: ``(n_s,)`` arc-length weights.
    """
    import numpy as np

    Cb = np.asarray(Cb_st, dtype=float)
    udn = np.asarray(u_dot_n_st, dtype=float)
    ds = np.asarray(ds_s, dtype=float).reshape(-1)
    if Cb.shape != udn.shape or Cb.shape[0] != ds.shape[0]:
        raise ValueError("incompatible Cb / u_dot_n / ds shapes")
    speed_in = np.clip(-udn, 0.0, None)
    return [float(v) for v in (speed_in * np.maximum(Cb, 0.0) * ds[:, None]).sum(axis=0)]


def transboundary_share(
    Cb_st,
    u_dot_n_st,
    ds_s,
    S_xyt=None,
    dx: float = 1.0,
    dy: float = 1.0,
    dt: float = 1.0,
    hysteresis: float = 0.0,
) -> tuple[float, dict]:
    """Compute the transboundary share + full accounting dict.

    Returns:
        ``(share, accounting)`` where share ∈ [0, 1] (``nan`` when the
        denominator is zero) and ``accounting`` holds every integrated
        term (``M_in``, ``M_S``, ``Q_in(t)``, ``n_inflow`` …).
    """
    import numpy as np

    Cb = np.asarray(Cb_st, dtype=float)
    udn = np.asarray(u_dot_n_st, dtype=float)
    ds = np.asarray(ds_s, dtype=float).reshape(-1)
    if Cb.shape != udn.shape or Cb.shape[0] != ds.shape[0]:
        raise ValueError("incompatible Cb / u_dot_n / ds shapes")
    inflow = udn < -float(hysteresis)
    speed_in = np.clip(-udn, 0.0, None) * inflow
    Q_in = (speed_in * np.maximum(Cb, 0.0) * ds[:, None]).sum(axis=0)
    M_in = float(Q_in.sum() * dt)
    if S_xyt is None:
        M_S, Q_S = 0.0, [0.0] * Cb.shape[1]
    else:
        S = np.maximum(np.asarray(S_xyt, dtype=float), 0.0)
        Q_S_arr = S.reshape(S.shape[0] * S.shape[1], S.shape[2]).sum(axis=0) * dx * dy
        Q_S = [float(v) for v in Q_S_arr]
        M_S = float(Q_S_arr.sum() * dt)
    denom = M_in + M_S
    share = float(M_in / denom) if denom > 0 else float("nan")
    accounting = {
        "M_in": M_in,
        "M_S": M_S,
        "M_total": denom,
        "share": share,
        "Q_in_t": [float(v) for v in Q_in],
        "Q_S_t": Q_S,
        "frac_inflow_faces": float(inflow.mean()) if inflow.size else 0.0,
        "n_s": int(Cb.shape[0]),
        "n_t": int(Cb.shape[1]),
        "dx": float(dx),
        "dy": float(dy),
        "dt": float(dt),
        "hysteresis": float(hysteresis),
        "rule": "share = M_in/(M_in+M_S); M_in = Σ_t Σ_{u·n<0} (−u·n)·C_b·ds·Δt",
    }
    return share, accounting


def decompose_by_sector(
    Cb_st,
    u_dot_n_st,
    ds_s,
    seg_dx_s,
    seg_dy_s,
    dt: float = 1.0,
    hysteresis: float = 0.0,
) -> dict[str, dict]:
    """Decompose inflow mass by NW/N/E/S/W sector.

    Returns ``{sector: {"M_in": …, "share_of_inflow": …, "n_faces": …}}``
    plus ``accounting["total"]``. Sector masses sum to the total
    (partition property).
    """
    import numpy as np

    Cb = np.asarray(Cb_st, dtype=float)
    udn = np.asarray(u_dot_n_st, dtype=float)
    ds = np.asarray(ds_s, dtype=float).reshape(-1)
    dxs = _as_float_list(seg_dx_s, "seg_dx_s")
    dys = _as_float_list(seg_dy_s, "seg_dy_s")
    n_s = Cb.shape[0]
    if not (len(dxs) == len(dys) == n_s):
        raise ValueError("segment offset vectors must match n_s")
    sectors = [sector_of_segment(dx, dy) for dx, dy in zip(dxs, dys)]
    inflow = udn < -float(hysteresis)
    speed_in = np.clip(-udn, 0.0, None) * inflow
    face_mass = (speed_in * np.maximum(Cb, 0.0) * ds[:, None]).sum(axis=1) * dt
    total = float(face_mass.sum())
    out: dict[str, dict] = {}
    for sector in SECTORS:
        idx = [i for i, s in enumerate(sectors) if s == sector]
        m = float(face_mass[idx].sum()) if idx else 0.0
        out[sector] = {
            "M_in": m,
            "share_of_inflow": (m / total) if total > 0 else float("nan"),
            "n_faces": len(idx),
        }
    out["accounting"] = {
        "M_in_total": total,
        "sectors_sum": float(sum(out[s]["M_in"] for s in SECTORS)),
        "rule": "5-sector compass partition; see module docstring",
    }
    return out


def mass_closure(
    M_in: float,
    M_S: float,
    M_out: float,
    M_dep: float,
    dM_storage: float,
) -> dict[str, float]:
    """Domain mass-balance closure diagnostic (PRD `physics/mass_balance`).

    Residual ``= (M_in + M_S) − (M_out + M_dep + ΔM)``; ``rel`` divides
    by the total input scale. A closed budget has residual ≈ 0.
    """
    inflow = float(M_in) + float(M_S)
    outflow = float(M_out) + float(M_dep) + float(dM_storage)
    resid = inflow - outflow
    scale = abs(inflow) + abs(outflow)
    return {
        "residual": resid,
        "rel_residual": (resid / scale) if scale > 0 else float("nan"),
        "M_in": float(M_in),
        "M_S": float(M_S),
        "M_out": float(M_out),
        "M_dep": float(M_dep),
        "dM_storage": float(dM_storage),
    }


def receptor_share_superposition(
    model,
    wind_u,
    wind_v,
    Cb_hourly,
    S_hourly,
    C0,
    receptor_xy,
    n_hours: int,
    spinup_h: int = 0,
    sector_of_node=None,
) -> dict:
    """**Headline attribution** (revised 2026-10-03): receptor-oriented share.

    The PDE is linear in ``(C_b, S, C0)`` for fixed wind/K/λ, so the field
    splits *exactly* into ``C = C^{bnd} + C^{src} + C^{ic}`` (superposition).
    The transboundary share at receptors (e.g. the Delhi CAAQMS stations)
    over a window is

        share = mean_{t ≥ spinup, receptors} C^{bnd} / mean C .

    This answers the policy question ("what fraction of the PM2.5 measured
    in Delhi entered across the airshed boundary?") and is the quantity
    that can be compared with DSS's "outside-NCR" categories. The older
    :func:`transboundary_share` (inflow *mass flux* through the box) also
    counts air that crosses the box without reaching Delhi, so it is kept
    as a secondary budget diagnostic only.

    Parameters
    ----------
    model:
        :class:`bapinnsformer.physics.greens.GridTransport` with the
        recovered ``K, λ``.
    wind_u, wind_v:
        ``(n_hours+1, ny, nx)`` hourly wind on the model grid.
    Cb_hourly:
        ``(n_hours+1, nb)`` recovered ``C_b`` at the model's boundary nodes
        (order = ``model.b_s``), linear in time between hours.
    S_hourly:
        ``(n_hours+1, ny, nx)`` recovered ``S`` (µg m⁻³ s⁻¹).
    C0:
        ``(ny, nx)`` initial field (e.g. the recovered ``C`` at window start).
    receptor_xy:
        ``(n_rec, 2)`` receptor coordinates (metres, SW-corner origin).
    sector_of_node:
        Optional ``(nb,)`` array of sector labels (e.g. from
        :func:`sector_of_segment`) to split ``C^{bnd}`` by entry sector.

    Returns ``{share, C_bnd, C_src, C_ic, C_total, closure_rel, sectors}``;
    ``closure_rel`` = |full run − sum of parts| / full run (must be ~1e-12:
    a larger value means the solver is not linear and the split is invalid).
    """
    import numpy as np

    from ..physics.greens import bilinear_weights

    Cbh = np.asarray(Cb_hourly, dtype=float)
    Sh = np.asarray(S_hourly, dtype=float)
    if Cbh.shape != (n_hours + 1, model.nb):
        raise ValueError(f"Cb_hourly must be (n_hours+1, nb={model.nb}), got {Cbh.shape}")
    if Sh.shape != (n_hours + 1, model.ny, model.nx):
        raise ValueError(f"S_hourly must be (n_hours+1, ny, nx), got {Sh.shape}")

    labels = None
    if sector_of_node is not None:
        labels = np.asarray(sector_of_node)
        sectors = [s for s in SECTORS if (labels == s).any()]
    else:
        sectors = []
    # members: 0 bnd, 1 src, 2 ic, 3 full, 4.. per-sector boundary
    m = 4 + len(sectors)
    masks = np.zeros((m, model.nb))
    masks[0] = 1.0
    masks[3] = 1.0
    for k, sec in enumerate(sectors):
        masks[4 + k] = (labels == sec).astype(float)
    s_on = np.zeros(m)
    s_on[[1, 3]] = 1.0
    c0 = np.zeros((m, model.ny, model.nx))
    c0[2] = C0
    c0[3] = C0

    def interp(arr, t):
        h = min(int(t // 3600.0), n_hours - 1)
        a = (t - h * 3600.0) / 3600.0
        return (1 - a) * arr[h] + a * arr[h + 1]

    def Cb_fn(t):
        return masks * interp(Cbh, t)[None, :]

    def S_fn(t):
        return s_on[:, None, None] * interp(Sh, t)[None]

    idx, w, _ = bilinear_weights(receptor_xy[:, 0], receptor_xy[:, 1],
                                 model.nx, model.ny, model.dx, model.dy)
    res = model.run(wind_u, wind_v, c0, Cb_fn, S_fn, n_hours, obs_idx=idx, obs_w=w)
    obs = res["obs"][spinup_h:]  # (H, m, n_rec)
    means = obs.mean(axis=(0, 2))
    parts = means[0] + means[1] + means[2]
    full = means[3]
    out = {
        "C_bnd": float(means[0]),
        "C_src": float(means[1]),
        "C_ic": float(means[2]),
        "C_total": float(full),
        "share": float(means[0] / full) if full > 0 else float("nan"),
        "share_src": float(means[1] / full) if full > 0 else float("nan"),
        "share_ic": float(means[2] / full) if full > 0 else float("nan"),
        "closure_rel": float(abs(full - parts) / max(abs(full), 1e-300)),
        "spinup_h": int(spinup_h),
        "rule": "share = mean C_bnd / mean C at receptors (exact superposition, advective PDE)",
        "sectors": {sec: float(means[4 + k] / full) if full > 0 else float("nan")
                    for k, sec in enumerate(sectors)},
    }
    return out


__all__ = [
    "receptor_share_superposition",
    "SECTORS",
    "bearing_deg",
    "sector_of_bearing",
    "sector_of_segment",
    "inflow_flux",
    "transboundary_share",
    "decompose_by_sector",
    "mass_closure",
]
