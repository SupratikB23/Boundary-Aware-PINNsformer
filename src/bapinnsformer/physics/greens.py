"""Linear transport operator + Green's functions (E1 oracle, attribution).

The governing equation used throughout the project (advective form,
RUN_PLAN §3 / PRD §2.5)::

    dC/dt + u·∇C = K ΔC + S − λ C        in Ω = [0, Lx] × [0, Ly]
    C = C_b                               on inflow  (u·n < 0)
    ∂C/∂n = 0                             on outflow (u·n ≥ 0)

is **linear** in the unknowns ``(C_b, S, C0)`` once the wind ``u`` and the
scalars ``K, λ`` are fixed. Hence every observation is a linear functional

    y = A_b c_b + A_s s + A_0 c_0 + ε                                  (1)

of finite-dimensional coefficient vectors that parameterize ``C_b``
(arc-length segment × time window), ``S`` (spatial patch × time window)
and ``C0`` (spatial patch). The columns of ``A_*`` are Green's functions,
computed here by running the discretized forward model once per basis
function, batched. Equation (1) is the backbone of

* the **identifiability analysis** (``eval/identifiability.py``): the
  geometry of ``range(A_b)`` vs ``range([A_s A_0])`` decides whether
  transboundary inflow can be separated from local emissions at all;
* the **Bayesian linear oracle** (best achievable estimator under the
  linear-Gaussian model) that E1 compares the PINNsformer against;
* the **receptor-oriented superposition attribution**
  (``eval/attribution.py``): by linearity
  ``C = C^{bnd} + C^{src} + C^{ic}`` exactly, so the transboundary share at
  receptors is ``mean C^{bnd} / mean C``.

Discretization: explicit Euler, first-order upwind advection (monotone),
second-order central diffusion, nodes on a uniform ``(ny, nx)`` grid with
``C[..., j, i]`` at ``(x_i, y_j)``. No positivity clipping (that would
break linearity). Boundary nodes are ordered **counter-clockwise from the
SW corner** exactly like ``data/domain.py`` (south → east → north → west),
so arc-length ``s`` here equals the project-wide ``s``.

Pure numpy; CPU only. Batched over a leading "member" axis ``m``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

__all__ = [
    "GridTransport",
    "GreensSystem",
    "build_greens",
    "bilinear_weights",
    "segment_of_s",
]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def segment_of_s(s: np.ndarray, perimeter: float, n_seg: int) -> np.ndarray:
    """Index of the equal-length arc-length segment containing ``s``."""
    s = np.mod(np.asarray(s, dtype=float), perimeter)
    idx = np.floor(s / (perimeter / n_seg)).astype(int)
    return np.clip(idx, 0, n_seg - 1)


def bilinear_weights(
    x: np.ndarray, y: np.ndarray, nx: int, ny: int, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat node indices ``(n_pts, 4)`` and weights ``(n_pts, 4)`` for
    bilinear sampling of a ``(ny, nx)`` grid at points ``(x, y)``.

    Points are clamped into the grid. Returns ``(flat_idx, w, inside)``
    where ``inside`` marks points that were inside the rectangle.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    Lx, Ly = dx * (nx - 1), dy * (ny - 1)
    inside = (x >= 0) & (x <= Lx) & (y >= 0) & (y <= Ly)
    xc = np.clip(x, 0.0, Lx)
    yc = np.clip(y, 0.0, Ly)
    fi = xc / dx
    fj = yc / dy
    i0 = np.clip(np.floor(fi).astype(int), 0, nx - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, ny - 2)
    wx = fi - i0
    wy = fj - j0
    idx = np.stack(
        [j0 * nx + i0, j0 * nx + i0 + 1, (j0 + 1) * nx + i0, (j0 + 1) * nx + i0 + 1], axis=1
    )
    w = np.stack([(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy], axis=1)
    return idx, w, inside


# ----------------------------------------------------------------------
# forward model
# ----------------------------------------------------------------------
class GridTransport:
    """Batched explicit FD solver of the linear advection–diffusion–loss PDE.

    Parameters
    ----------
    nx, ny:
        Grid nodes (≥ 3 each). Spacing ``dx = Lx/(nx-1)``.
    Lx, Ly:
        Domain extents in metres (origin at the SW corner).
    K, lam:
        Eddy diffusivity (m² s⁻¹) and first-order loss rate (s⁻¹).
    cfl:
        Safety factor (< 1) on the combined advective+diffusive limit.
    """

    def __init__(
        self, nx: int, ny: int, Lx: float, Ly: float, K: float, lam: float, cfl: float = 0.45
    ) -> None:
        if nx < 3 or ny < 3:
            raise ValueError("GridTransport: need nx, ny >= 3")
        if Lx <= 0 or Ly <= 0:
            raise ValueError("GridTransport: Lx, Ly must be positive")
        if K < 0 or lam < 0:
            raise ValueError("GridTransport: K, lam must be non-negative")
        if not 0 < cfl < 1:
            raise ValueError("GridTransport: cfl must be in (0, 1)")
        self.nx, self.ny = int(nx), int(ny)
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)
        self.K, self.lam, self.cfl = float(K), float(lam), float(cfl)
        self.perimeter = 2.0 * (self.Lx + self.Ly)
        self._build_boundary()

    # -- boundary bookkeeping ------------------------------------------
    def _build_boundary(self) -> None:
        nx, ny, dx, dy, Lx, Ly = self.nx, self.ny, self.dx, self.dy, self.Lx, self.Ly
        bi, bj, nxs, nys, ss = [], [], [], [], []
        for i in range(nx):  # south, SW→SE (includes both corners)
            bi.append(i); bj.append(0); nxs.append(0.0); nys.append(-1.0); ss.append(i * dx)
        for j in range(1, ny):  # east, SE→NE (includes NE corner)
            bi.append(nx - 1); bj.append(j); nxs.append(1.0); nys.append(0.0); ss.append(Lx + j * dy)
        for i in range(nx - 2, -1, -1):  # north, NE→NW (includes NW corner)
            bi.append(i); bj.append(ny - 1); nxs.append(0.0); nys.append(1.0)
            ss.append(Lx + Ly + (nx - 1 - i) * dx)
        for j in range(ny - 2, 0, -1):  # west, NW→SW (corners excluded)
            bi.append(0); bj.append(j); nxs.append(-1.0); nys.append(0.0)
            ss.append(2 * Lx + Ly + (ny - 1 - j) * dy)
        self.b_i = np.asarray(bi, dtype=int)
        self.b_j = np.asarray(bj, dtype=int)
        self.b_nx = np.asarray(nxs, dtype=float)
        self.b_ny = np.asarray(nys, dtype=float)
        self.b_s = np.asarray(ss, dtype=float)
        # inward neighbour for the zero-gradient (outflow) condition
        self.b_in_i = np.clip(self.b_i - self.b_nx.astype(int), 0, nx - 1)
        self.b_in_j = np.clip(self.b_j - self.b_ny.astype(int), 0, ny - 1)
        self.nb = self.b_i.size
        assert self.nb == 2 * (nx - 1) + 2 * (ny - 1)

    @property
    def x(self) -> np.ndarray:
        return np.linspace(0.0, self.Lx, self.nx)

    @property
    def y(self) -> np.ndarray:
        return np.linspace(0.0, self.Ly, self.ny)

    def boundary_xy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.b_i * self.dx, self.b_j * self.dy

    def u_dot_n(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Normal wind at boundary nodes ``(nb,)`` (negative = inflow)."""
        return u[self.b_j, self.b_i] * self.b_nx + v[self.b_j, self.b_i] * self.b_ny

    # -- time step -------------------------------------------------------
    def stable_dt(self, umax: float, vmax: float) -> float:
        rate = (
            abs(umax) / self.dx
            + abs(vmax) / self.dy
            + 2.0 * self.K * (1.0 / self.dx**2 + 1.0 / self.dy**2)
        )
        return math.inf if rate <= 0 else self.cfl / rate

    def step(
        self,
        C: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        S: np.ndarray | None,
        Cb: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Advance ``C (m, ny, nx)`` by ``dt``. ``Cb (m, nb)`` Dirichlet data."""
        dx, dy = self.dx, self.dy
        up, um = np.maximum(u, 0.0), np.minimum(u, 0.0)
        vp, vm = np.maximum(v, 0.0), np.minimum(v, 0.0)
        adv = np.zeros_like(C)
        adv[..., :, 1:-1] += (
            up[:, 1:-1] * (C[..., :, 1:-1] - C[..., :, :-2])
            + um[:, 1:-1] * (C[..., :, 2:] - C[..., :, 1:-1])
        ) / dx
        adv[..., 1:-1, :] += (
            vp[1:-1, :] * (C[..., 1:-1, :] - C[..., :-2, :])
            + vm[1:-1, :] * (C[..., 2:, :] - C[..., 1:-1, :])
        ) / dy
        lap = np.zeros_like(C)
        if self.K > 0:
            lap[..., 1:-1, 1:-1] = (
                (C[..., 1:-1, 2:] - 2 * C[..., 1:-1, 1:-1] + C[..., 1:-1, :-2]) / dx**2
                + (C[..., 2:, 1:-1] - 2 * C[..., 1:-1, 1:-1] + C[..., :-2, 1:-1]) / dy**2
            )
        rhs = -adv + self.K * lap - self.lam * C
        if S is not None:
            rhs = rhs + S
        Cn = C + dt * rhs
        inflow = self.u_dot_n(u, v) < 0.0
        bvals = np.where(inflow[None, :], Cb, Cn[..., self.b_in_j, self.b_in_i])
        Cn[..., self.b_j, self.b_i] = bvals
        return Cn

    # -- driver ------------------------------------------------------------
    def run(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        C0: np.ndarray,
        Cb_fn: Callable[[float], np.ndarray],
        S_fn: Callable[[float], np.ndarray | None] | None,
        n_hours: int,
        obs_idx: np.ndarray | None = None,
        obs_w: np.ndarray | None = None,
        record_fields: bool = False,
    ) -> dict[str, np.ndarray]:
        """Integrate ``n_hours`` hours; record at the end of every hour.

        ``wind_u/v``: ``(n_hours + 1, ny, nx)`` hourly snapshots at
        ``t = 0, 3600, …``; linear in time between snapshots.
        ``Cb_fn(t) -> (m, nb)``, ``S_fn(t) -> (m, ny, nx) | None``.
        ``obs_idx/obs_w``: from :func:`bilinear_weights`.
        Returns ``{"obs": (n_hours, m, n_obs)}`` (+ ``"fields"`` if asked).
        """
        U = np.asarray(wind_u, dtype=float)
        V = np.asarray(wind_v, dtype=float)
        if U.shape != V.shape or U.ndim != 3 or U.shape[1:] != (self.ny, self.nx):
            raise ValueError(f"wind must be (n_hours+1, {self.ny}, {self.nx}); got {U.shape}")
        if U.shape[0] < n_hours + 1:
            raise ValueError("wind sequence shorter than n_hours + 1")
        C = np.array(C0, dtype=float, copy=True)
        if C.ndim == 2:
            C = C[None]
        m = C.shape[0]
        dt_max = self.stable_dt(float(np.abs(U).max()), float(np.abs(V).max()))
        n_sub = max(1, int(math.ceil(3600.0 / dt_max)))
        dt = 3600.0 / n_sub
        obs = None
        if obs_idx is not None:
            obs = np.empty((n_hours, m, obs_idx.shape[0]))
        fields = np.empty((n_hours, m, self.ny, self.nx)) if record_fields else None
        t = 0.0
        for h in range(n_hours):
            for k in range(n_sub):
                a = (k + 0.5) / n_sub
                u = (1 - a) * U[h] + a * U[h + 1]
                v = (1 - a) * V[h] + a * V[h + 1]
                tm = t + 0.5 * dt
                Cb = np.broadcast_to(np.asarray(Cb_fn(tm), dtype=float), (m, self.nb))
                S = None if S_fn is None else S_fn(tm)
                C = self.step(C, u, v, S, Cb, dt)
                t += dt
            if obs is not None:
                flat = C.reshape(m, -1)
                obs[h] = (flat[:, obs_idx] * obs_w[None]).sum(axis=-1)
            if fields is not None:
                fields[h] = C
        out: dict[str, np.ndarray] = {"dt": np.array(dt), "n_sub": np.array(n_sub)}
        if obs is not None:
            out["obs"] = obs
        if fields is not None:
            out["fields"] = fields
        return out


# ----------------------------------------------------------------------
# Green's functions
# ----------------------------------------------------------------------
@dataclass
class GreensSystem:
    """Linear system ``y = A_b c_b + A_s s + A_0 c_0`` (+ receptor operators).

    Rows of ``A_*`` are ordered ``(hour, station)`` for hours
    ``spinup_h … n_hours-1``. ``R_*`` are receptor operators: the
    time-and-receptor mean concentration contributed by each column.
    """

    A_b: np.ndarray
    A_s: np.ndarray
    A_0: np.ndarray
    R_b: np.ndarray
    R_s: np.ndarray
    R_0: np.ndarray
    b_seg: np.ndarray  # (n_b_cols,) segment index
    b_win: np.ndarray  # (n_b_cols,) time-window index
    b_inflow_frac: np.ndarray  # (n_b_cols,) fraction of window the segment is inflow
    s_patch: np.ndarray
    s_win: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def G(self) -> np.ndarray:
        return np.hstack([self.A_b, self.A_s, self.A_0])

    @property
    def R(self) -> np.ndarray:
        return np.concatenate([self.R_b, self.R_s, self.R_0])

    @property
    def sizes(self) -> tuple[int, int, int]:
        return self.A_b.shape[1], self.A_s.shape[1], self.A_0.shape[1]


def _patch_index(model: GridTransport, n_px: int, n_py: int) -> np.ndarray:
    """``(ny, nx)`` map from node to patch id (row-major patches)."""
    pi = np.minimum((np.arange(model.nx) * n_px) // model.nx, n_px - 1)
    pj = np.minimum((np.arange(model.ny) * n_py) // model.ny, n_py - 1)
    return pj[:, None] * n_px + pi[None, :]


def build_greens(
    model: GridTransport,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    n_hours: int,
    obs_xy: np.ndarray,
    receptor_xy: np.ndarray | None = None,
    n_bseg: int = 12,
    bwin_h: int = 6,
    src_patches: tuple[int, int] = (4, 4),
    swin_h: int = 24,
    ic_patches: tuple[int, int] = (2, 2),
    spinup_h: int = 0,
) -> GreensSystem:
    """Compute Green's functions for boundary, source and IC bases.

    Boundary basis column ``(seg, win)``: ``C_b = 1`` on nodes whose
    arc-length lies in segment ``seg`` during window ``win`` (hours
    ``[win·bwin_h, (win+1)·bwin_h)``), 0 elsewhere. Units: µg m⁻³.

    Source basis column ``(patch, win)``: ``S = 1/3600`` µg m⁻³ s⁻¹
    (i.e. 1 µg m⁻³ h⁻¹) on the patch during window ``win``.

    IC basis column ``patch``: ``C0 = 1`` µg m⁻³ on the patch.
    """
    obs_xy = np.asarray(obs_xy, dtype=float).reshape(-1, 2)
    if receptor_xy is None:
        receptor_xy = obs_xy
    receptor_xy = np.asarray(receptor_xy, dtype=float).reshape(-1, 2)
    if spinup_h < 0 or spinup_h >= n_hours:
        raise ValueError("spinup_h must be in [0, n_hours)")

    nbw = int(math.ceil(n_hours / bwin_h))
    nsw = int(math.ceil(n_hours / swin_h))
    npx, npy = src_patches
    nip, njp = ic_patches
    n_b = n_bseg * nbw
    n_s = npx * npy * nsw
    n_0 = nip * njp
    m = n_b + n_s + n_0

    node_seg = segment_of_s(model.b_s, model.perimeter, n_bseg)
    src_map = _patch_index(model, npx, npy)
    ic_map = _patch_index(model, nip, njp)

    b_seg = np.repeat(np.arange(n_bseg), nbw)  # column c = seg*nbw + win
    b_win = np.tile(np.arange(nbw), n_bseg)
    s_patch = np.repeat(np.arange(npx * npy), nsw)
    s_win = np.tile(np.arange(nsw), npx * npy)

    # one-hot templates
    seg_onehot = (node_seg[None, :] == b_seg[:, None]).astype(float)  # (n_b, nb)
    patch_onehot = (src_map[None, :, :] == s_patch[:, None, None]).astype(float) / 3600.0

    def Cb_fn(t: float) -> np.ndarray:
        win = min(int(t // (bwin_h * 3600.0)), nbw - 1)
        out = np.zeros((m, model.nb))
        out[:n_b] = seg_onehot * (b_win == win)[:, None]
        return out

    def S_fn(t: float) -> np.ndarray:
        win = min(int(t // (swin_h * 3600.0)), nsw - 1)
        out = np.zeros((m, model.ny, model.nx))
        out[n_b:n_b + n_s] = patch_onehot * (s_win == win)[:, None, None]
        return out

    C0 = np.zeros((m, model.ny, model.nx))
    for p in range(n_0):
        C0[n_b + n_s + p] = (ic_map == p).astype(float)

    pts = np.vstack([obs_xy, receptor_xy])
    idx, w, _ = bilinear_weights(pts[:, 0], pts[:, 1], model.nx, model.ny, model.dx, model.dy)
    res = model.run(wind_u, wind_v, C0, Cb_fn, S_fn, n_hours, obs_idx=idx, obs_w=w)
    obs = res["obs"]  # (n_hours, m, n_obs + n_rec)
    n_obs = obs_xy.shape[0]
    kept = obs[spinup_h:]  # (H, m, ·)
    A = kept[:, :, :n_obs].transpose(0, 2, 1).reshape(-1, m)  # rows (hour, station)
    Rm = kept[:, :, n_obs:].mean(axis=(0, 2))  # (m,)

    # inflow fraction per boundary column (window-mean over segment nodes)
    inflow_frac = np.zeros(n_b)
    for c in range(n_b):
        hours = range(b_win[c] * bwin_h, min((b_win[c] + 1) * bwin_h, n_hours))
        nodes = node_seg == b_seg[c]
        fr = [
            float((model.u_dot_n(wind_u[h], wind_v[h])[nodes] < 0).mean()) for h in hours
        ]
        inflow_frac[c] = float(np.mean(fr)) if fr else 0.0

    return GreensSystem(
        A_b=A[:, :n_b],
        A_s=A[:, n_b:n_b + n_s],
        A_0=A[:, n_b + n_s:],
        R_b=Rm[:n_b],
        R_s=Rm[n_b:n_b + n_s],
        R_0=Rm[n_b + n_s:],
        b_seg=b_seg,
        b_win=b_win,
        b_inflow_frac=inflow_frac,
        s_patch=s_patch,
        s_win=s_win,
        meta={
            "n_hours": int(n_hours),
            "spinup_h": int(spinup_h),
            "n_bseg": int(n_bseg),
            "bwin_h": int(bwin_h),
            "src_patches": [int(npx), int(npy)],
            "swin_h": int(swin_h),
            "ic_patches": [int(nip), int(njp)],
            "n_obs_points": int(n_obs),
            "n_receptors": int(receptor_xy.shape[0]),
            "dt": float(res["dt"]),
            "n_sub": int(res["n_sub"]),
            "grid": [model.nx, model.ny, model.Lx, model.Ly],
            "K": model.K,
            "lam": model.lam,
            "row_order": "(hour, station)",
            "units": {"c_b": "ug m-3", "s": "ug m-3 h-1", "c_0": "ug m-3"},
        },
    )
