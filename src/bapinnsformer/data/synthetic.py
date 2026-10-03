"""Finite-difference synthetic forward solver for E1 ground truth.

PRD §4.3 `data/synthetic.py`: generates concentration fields from
prescribed interior sources ``S`` and boundary inflow ``C_b`` under
real ERA5-derived wind sequences. Explicit Euler in time, first-order
upwind advection, second-order central diffusion, first-order loss.
CFL-safe timestep enforced; exposes both a functional API
(``cfl_timestep``/``run_forward``/``make_gaussian_source``/
``make_boundary_inflow``) and an object API (``SyntheticSolver`` with
wind-switched Dirichlet/Neumann faces, PRD §2.4). CPU-only, numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticConfig:
    """Grid + physics config for the FD solver (all SI-ish units)."""

    nx: int = 64
    ny: int = 64
    Lx: float = 60_000.0
    Ly: float = 60_000.0
    K: float = 25.0
    lam: float = 1e-5
    dt: float | None = None  # auto from CFL if None
    cfl: float = 0.4


def cfl_limits(u_max: float, v_max: float, K: float, dx: float, dy: float) -> dict[str, float]:
    """Advective + diffusive explicit-Euler timestep limits."""
    # No wind => no advective restriction (quiescent fields accept any dt).
    adv = math.inf
    if u_max > 0 or v_max > 0:
        adv = 1.0 / (abs(u_max) / dx + abs(v_max) / dy + 1e-300)
    dif = (dx * dx * dy * dy) / (2.0 * max(K, 0.0) * (dx * dx + dy * dy) + 1e-300) if K > 0 else math.inf
    return {"advective": float(adv), "diffusive": float(dif), "stable": float(min(adv, dif))}


def cfl_timestep(dx: float, dy: float, umax: float, K: float, cfl: float = 0.4) -> float:
    """CFL-safe explicit timestep for advection + diffusion.

    ``dt_adv = cfl * min(dx, dy) / max(|u|, eps)``,
    ``dt_dif = 0.25 * (dx²·dy²) / (K·(dx²+dy²))`` (2-D explicit
    diffusion limit); returns ``min(dt_adv, dt_dif)``.
    """
    if dx <= 0 or dy <= 0:
        raise ValueError("cfl_timestep: dx, dy must be positive")
    if K < 0:
        raise ValueError("cfl_timestep: K must be non-negative")
    umax = float(abs(umax))
    dt_adv = cfl * min(dx, dy) / max(umax, 1e-9)
    if K > 0:
        dt_dif = 0.25 * (dx * dx * dy * dy) / (K * (dx * dx + dy * dy))
        return float(min(dt_adv, dt_dif))
    return float(dt_adv)


def check_cfl(u: object, v: object, K: float, dx: float, dy: float, dt: float, cfl: float = 0.9) -> dict[str, float]:
    """Raise ``ValueError`` if ``dt`` violates the CFL condition."""
    lim = cfl_limits(
        float(np.abs(np.asarray(u)).max(initial=0.0)),
        float(np.abs(np.asarray(v)).max(initial=0.0)),
        float(K), float(dx), float(dy),
    )
    if not math.isfinite(lim["stable"]):
        return lim
    if float(dt) > float(cfl) * lim["stable"]:
        raise ValueError(
            f"CFL violated: dt={dt} > {cfl}×dt_stable={cfl * lim['stable']:.4g} "
            f"(adv {lim['advective']:.4g}, diff {lim['diffusive']:.4g})"
        )
    return lim


def make_gaussian_source(
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    centers: list[tuple[float, float]],
    amplitudes: list[float],
    sigmas: list[float],
) -> np.ndarray:
    """Static 2-D Gaussian mixture source field ``S(x, y)`` (Ny, Nx)."""
    if not (len(centers) == len(amplitudes) == len(sigmas)):
        raise ValueError("make_gaussian_source: centers/amplitudes/sigmas length mismatch")
    xs = np.linspace(0.0, Lx, nx)
    ys = np.linspace(0.0, Ly, ny)
    XX, YY = np.meshgrid(xs, ys)
    S = np.zeros((ny, nx))
    for (cx, cy), a, s in zip(centers, amplitudes, sigmas):
        if s <= 0:
            raise ValueError("make_gaussian_source: sigma must be positive")
        S += float(a) * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2.0 * s * s))
    return S


def make_boundary_inflow(
    n_s: int,
    n_t: int,
    baseline: float = 50.0,
    nw_pulse_amp: float = 150.0,
    nw_sector: tuple[float, float] = (0.5, 0.8),
    pulse_window: tuple[float, float] = (0.3, 0.6),
) -> np.ndarray:
    """Prescribed ``C_b(s, t)`` (Nt, Ns): baseline + NW-sector pulse.

    The pulse mimics the Oct–Nov north-westerly crop-burning signal so
    E1 can test recovery of a localized, transient inflow event.
    Sector/window args are fractions of ``[0, 1)`` along s and t.
    """
    if n_s < 1 or n_t < 1:
        raise ValueError("make_boundary_inflow: n_s, n_t must be >= 1")
    Cb = np.full((n_t, n_s), float(baseline))
    s0, s1 = int(nw_sector[0] * n_s), int(nw_sector[1] * n_s)
    t0, t1 = int(pulse_window[0] * n_t), int(pulse_window[1] * n_t)
    tt = np.linspace(0.0, 1.0, max(t1 - t0, 1)) if t1 > t0 else np.array([1.0])
    envelope = np.sin(np.pi * np.linspace(0.0, 1.0, tt.size)) ** 2
    for j, e in enumerate(envelope):
        Cb[t0 + j, s0:s1] += float(nw_pulse_amp) * float(e)
    return Cb


def _perimeter_vector(v: np.ndarray, nx: int, ny: int) -> np.ndarray:
    need = 2 * nx + 2 * (ny - 2)
    v = np.asarray(v, dtype=float).ravel()
    if v.size < need:
        v = np.tile(v, int(np.ceil(need / max(v.size, 1))))[:need]
    return v[:need]


def _apply_boundary(C: np.ndarray, Cb_vec: np.ndarray, u: np.ndarray | None = None,
                    v: np.ndarray | None = None) -> None:
    """Wind-switched boundary condition from a perimeter vector (in place).

    ``Cb_vec`` holds [south(nx, W→E) | east(ny-2, S→N) | north(nx, W→E) |
    west(ny-2, S→N)] (tiled if shorter, truncated if longer).

    * Inflow nodes (``u·n < 0``): Dirichlet ``C = C_b``.
    * Outflow / tangent nodes: zero normal gradient (copy the inward
      neighbour) — the same switch as the model (PRD §2.4) and as
      ``physics/greens.py``. Before the 2026-10-03 fix, Dirichlet values
      were imposed on *every* face, so E1 "truth" contained boundary data
      on outflow faces that no inverse method can (or should) recover.

    ``u``/``v`` ``None`` keeps the legacy all-Dirichlet behaviour (used
    only for initial-condition stamping).
    """
    ny, nx = C.shape
    b = _perimeter_vector(Cb_vec, nx, ny)
    south, east = b[:nx], b[nx:nx + ny - 2]
    north, west = b[nx + ny - 2:2 * nx + ny - 2], b[2 * nx + ny - 2:]
    if u is None or v is None:
        C[0, :] = south
        C[1:-1, -1] = east
        C[-1, :] = north
        C[1:-1, 0] = west
        return
    # outward normals: south (0,-1), east (1,0), north (0,1), west (-1,0)
    inn = v[0, :] > 0.0  # south inflow: wind blowing north into the domain
    C[0, :] = np.where(inn, south, C[1, :])
    inn = u[1:-1, -1] < 0.0
    C[1:-1, -1] = np.where(inn, east, C[1:-1, -2])
    inn = v[-1, :] < 0.0
    C[-1, :] = np.where(inn, north, C[-2, :])
    inn = u[1:-1, 0] > 0.0
    C[1:-1, 0] = np.where(inn, west, C[1:-1, 1])


def run_forward(
    C0: np.ndarray,
    u_seq: np.ndarray,
    v_seq: np.ndarray,
    S: np.ndarray | None,
    Cb_seq: np.ndarray,
    K: float,
    lam: float,
    dx: float,
    dy: float,
    dt: float,
    n_steps: int,
    check_cfl_flag: bool = True,
    clip_negative: bool = True,
) -> dict[str, np.ndarray]:
    """Integrate the FD forward model; return ground-truth dict.

    Args:
        C0: (Ny, Nx) initial field.
        u_seq, v_seq: (Nt, Ny, Nx) or (Ny, Nx) wind in m/s.
        S: (Ny, Nx) static source or None (zeros).
        Cb_seq: (Nt, Nb) Dirichlet boundary values per step.
        K, lam: diffusivity (m²/s) and loss rate (1/s).
        dx, dy, dt: grid spacing (m) and timestep (s).
        n_steps: number of steps to integrate.
        check_cfl_flag: raise if ``dt`` violates the CFL limit.
        clip_negative: clip C at 0 each step (set False to keep the solver
            exactly linear, e.g. for superposition checks).

    Returns:
        ``{"C": (n_steps+1, Ny, Nx), "S": ..., "Cb": ..., "dt": ...}``.
    """
    C = np.asarray(C0, dtype=float)
    ny, nx = C.shape
    if dx <= 0 or dy <= 0 or dt <= 0:
        raise ValueError("run_forward: dx, dy, dt must be positive")
    if K < 0 or lam < 0:
        raise ValueError("run_forward: K, lam must be non-negative")
    U = np.asarray(u_seq, dtype=float)
    V = np.asarray(v_seq, dtype=float)
    if U.ndim == 2:
        U = np.broadcast_to(U, (n_steps, ny, nx)).copy()
    if V.ndim == 2:
        V = np.broadcast_to(V, (n_steps, ny, nx)).copy()
    Cb = np.asarray(Cb_seq, dtype=float)
    Sarr = np.zeros_like(C) if S is None else np.asarray(S, dtype=float)
    if Sarr.shape != C.shape:
        raise ValueError("run_forward: S shape must match C0")
    if check_cfl_flag:
        check_cfl(U, V, K, dx, dy, dt, cfl=1.0)
    traj = np.empty((n_steps + 1, ny, nx))
    traj[0] = C
    cur = C.copy()
    dx2, dy2 = dx * dx, dy * dy
    for k in range(n_steps):
        kk = min(k, U.shape[0] - 1)
        uu, vv = U[kk], V[kk]
        _apply_boundary(cur, Cb[min(k, Cb.shape[0] - 1)], uu, vv)
        # First-order upwind advection in advective form (same stencil as
        # SyntheticSolver.step): positive wind takes the backward
        # difference of C, negative wind the forward difference. (The old
        # code differenced the flux u*C with the backward stencil for
        # both wind signs — non-monotone overshoots under negative
        # winds.) Boundary columns/rows reuse the one-sided difference
        # available there (rims are overwritten by _apply_boundary anyway).
        dudx = np.zeros_like(cur)
        dudx[:, 1:-1] = (
            np.maximum(uu[:, 1:-1], 0) * (cur[:, 1:-1] - cur[:, :-2])
            + np.minimum(uu[:, 1:-1], 0) * (cur[:, 2:] - cur[:, 1:-1])
        ) / dx
        dudx[:, 0] = np.minimum(uu[:, 0], 0) * (cur[:, 1] - cur[:, 0]) / dx
        dudx[:, -1] = np.maximum(uu[:, -1], 0) * (cur[:, -1] - cur[:, -2]) / dx
        dvdy = np.zeros_like(cur)
        dvdy[1:-1, :] = (
            np.maximum(vv[1:-1, :], 0) * (cur[1:-1, :] - cur[:-2, :])
            + np.minimum(vv[1:-1, :], 0) * (cur[2:, :] - cur[1:-1, :])
        ) / dy
        dvdy[0, :] = np.minimum(vv[0, :], 0) * (cur[1, :] - cur[0, :]) / dy
        dvdy[-1, :] = np.maximum(vv[-1, :], 0) * (cur[-1, :] - cur[-2, :]) / dy
        lap = np.zeros_like(cur)
        lap[1:-1, 1:-1] = (
            (cur[1:-1, 2:] - 2 * cur[1:-1, 1:-1] + cur[1:-1, :-2]) / dx2
            + (cur[2:, 1:-1] - 2 * cur[1:-1, 1:-1] + cur[:-2, 1:-1]) / dy2
        )
        nxt = cur + dt * (-dudx - dvdy + K * lap + Sarr - lam * cur)
        if clip_negative:
            nxt = np.maximum(nxt, 0.0)
        _apply_boundary(nxt, Cb[min(k + 1, Cb.shape[0] - 1)], uu, vv)
        traj[k + 1] = nxt
        cur = nxt
    return {"C": traj, "S": Sarr.copy(), "Cb": Cb.copy(), "dt": np.array(dt)}


def add_noise(
    C: np.ndarray,
    noise_std: float,
    seed: int = 0,
    relative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """E1 noise model: Gaussian obs noise; returns (noisy, all-valid mask)."""
    rng = np.random.default_rng(seed)
    C = np.asarray(C, dtype=float)
    scale = np.abs(C) * noise_std if relative else noise_std
    return C + rng.normal(0.0, 1.0, size=C.shape) * scale, np.ones_like(C, dtype=bool)


class SyntheticSolver:
    """Explicit 2-D advection–diffusion–deposition stepper (numpy).

    Object API around the same discretization as :func:`run_forward`,
    with per-face wind-switched Dirichlet (inflow) / zero-gradient
    (outflow) boundaries (PRD §2.4). Grid convention here is
    ``(nx, ny)`` (x-major) to match the solver tests.
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float, dt: float,
                 K: float = 0.0, lam: float = 0.0, cfl: float = 0.9) -> None:
        if nx < 3 or ny < 3:
            raise ValueError("need nx, ny ≥ 3")
        self.nx, self.ny = int(nx), int(ny)
        self.dx, self.dy, self.dt = float(dx), float(dy), float(dt)
        self.K, self.lam, self.cfl = float(K), float(lam), float(cfl)

    def step(self, C: np.ndarray, u: object, v: object, S: object = 0.0, Cb: dict | None = None) -> np.ndarray:
        """Advance one step. ``u``/``v`` scalars or ``(nx, ny)`` arrays.

        ``Cb`` dict with keys west/east/south/north (scalar or face
        vector) applied as Dirichlet values on inflow faces; outflow
        faces get zero-gradient (copied interior value).
        """
        C = np.asarray(C, dtype=float)
        if C.shape != (self.nx, self.ny):
            raise ValueError(f"C shape {C.shape} != {(self.nx, self.ny)}")
        U = np.broadcast_to(np.asarray(u, dtype=float), C.shape).copy()
        V = np.broadcast_to(np.asarray(v, dtype=float), C.shape).copy()
        check_cfl(U, V, self.K, self.dx, self.dy, self.dt, self.cfl)
        Src = np.broadcast_to(np.asarray(S, dtype=float), C.shape)
        Cn = C.copy()
        i, j = slice(1, -1), slice(1, -1)
        adv = (
            np.maximum(U[i, j], 0) * (C[i, j] - C[:-2, j]) / self.dx
            + np.minimum(U[i, j], 0) * (C[2:, j] - C[i, j]) / self.dx
            + np.maximum(V[i, j], 0) * (C[i, j] - C[i, :-2]) / self.dy
            + np.minimum(V[i, j], 0) * (C[i, 2:] - C[i, j]) / self.dy
        )
        dif = 0.0
        if self.K > 0:
            dif = self.K * (
                (C[2:, j] - 2 * C[i, j] + C[:-2, j]) / self.dx**2
                + (C[i, 2:] - 2 * C[i, j] + C[i, :-2]) / self.dy**2
            )
        Cn[i, j] = C[i, j] + self.dt * (-adv + dif + Src[i, j] - self.lam * C[i, j])
        Cb = Cb or {}
        faces = {
            "west": (U[0, :] < 0, (0, slice(None)), 1),
            "east": (U[-1, :] > 0, (-1, slice(None)), 1),
            "south": (V[:, 0] < 0, (slice(None), 0), 1),
            "north": (V[:, -1] > 0, (slice(None), -1), 1),
        }
        for key, (out_mask, idx, _) in faces.items():
            out_mask = np.asarray(out_mask)
            if key in Cb and Cb[key] is not None:
                val = np.broadcast_to(np.asarray(Cb[key], dtype=float), out_mask.shape)
                in_mask = ~out_mask
                face = np.empty(out_mask.shape)
                face[in_mask] = val[in_mask]
                if key == "west":
                    face[out_mask] = Cn[1, :][out_mask]
                    Cn[0, :] = face
                elif key == "east":
                    face[out_mask] = Cn[-2, :][out_mask]
                    Cn[-1, :] = face
                elif key == "south":
                    face[out_mask] = Cn[:, 1][out_mask]
                    Cn[:, 0] = face
                else:
                    face[out_mask] = Cn[:, -2][out_mask]
                    Cn[:, -1] = face
            else:
                if key == "west":
                    Cn[0, :] = Cn[1, :]
                elif key == "east":
                    Cn[-1, :] = Cn[-2, :]
                elif key == "south":
                    Cn[:, 0] = Cn[:, 1]
                else:
                    Cn[:, -1] = Cn[:, -2]
        return np.maximum(Cn, 0.0)

    def total_mass(self, C: np.ndarray) -> float:
        return float(np.asarray(C, dtype=float).sum() * self.dx * self.dy)

    def run(self, C0: np.ndarray, u: object, v: object, S: object = 0.0, Cb: dict | None = None, n_steps: int = 1) -> np.ndarray:
        C = np.asarray(C0, dtype=float).copy()
        for _ in range(int(n_steps)):
            C = self.step(C, u, v, S=S, Cb=Cb)
        return C


__all__ = [
    "SyntheticConfig",
    "cfl_timestep",
    "cfl_limits",
    "check_cfl",
    "make_gaussian_source",
    "make_boundary_inflow",
    "run_forward",
    "add_noise",
    "SyntheticSolver",
]
