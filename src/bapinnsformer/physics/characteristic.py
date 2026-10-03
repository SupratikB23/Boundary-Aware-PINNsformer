"""Characteristic / Feynman–Kac residual with exit-time boundary coupling.

Contribution C1 of the revised method (RUN_PLAN §1). For the advective form

    ∂C/∂t + u·∇C = K ΔC + S − λ C,        C = C_b on inflow boundary,

the Feynman–Kac representation over a look-back ``τ = k·Δt`` reads

    C(x, t) = E[ e^{−λ τ_k} · V_k  +  ∫_{t−τ_k}^{t} e^{−λ(t−r)} S(X_r, r) dr ]   (FK)

where ``X`` solves the backward SDE ``dX = −u dt + √(2K) dW`` started at
``(x, t)``, ``τ_k = min(kΔt, t − t_exit)`` and

* ``V_k = C(X_{t−kΔt}, t − kΔt)`` if the path is still inside the domain;
* ``V_k = C_b(s_exit, t_exit)`` if the path left through an **inflow**
  boundary point first (Dirichlet ⇔ absorption: this is exactly where the
  unknown boundary inflow enters the equation);
* ``V_k = C(X_exit, t_exit)`` if it touched an **outflow** point (zero
  normal gradient ⇔ the field's own boundary value; no new information).

The ``advection_jitter`` pseudo-sequence generator produces precisely the
Euler–Maruyama samples ``X`` of this SDE (step std ``√(2KΔt)``), and the
deterministic ``advection_backward`` generator is its ``K → 0`` limit (pure
characteristics, valid at high Péclet number). The residual

    r_k = [ C(x,t) − e^{−λτ_k} V_k − Σ_trap e^{−λ(t−r)} S ] / τ_k           (2)

therefore (i) re-uses the very tokens the transformer attends over, (ii)
needs **no second derivatives** (no Laplacian through autograd), and (iii)
couples interior observations *directly* to ``C_b`` at the exit point,
which is the mechanism that makes boundary inflow recoverable at all.
Dividing by ``τ_k`` puts ``r_k`` in the same units as the PDE residual.

Accuracy notes (state these in the paper): exit time/point are first-order
(the first token flagged out-of-domain, clamped onto the rectangle); the
source integral is a trapezoid over tokens; with deterministic tokens the
diffusion term is neglected (error O(K Δ C · τ)). Use ``substeps ≥ 2``.

Arc-length convention: counter-clockwise from the SW corner (south → east →
north → west), identical to ``data/domain.py`` and ``physics/greens.py``.
"""

from __future__ import annotations

from typing import Callable

import torch

__all__ = ["xy_to_arclength", "outward_normal", "characteristic_residual"]


def xy_to_arclength(x: torch.Tensor, y: torch.Tensor, x_bounds, y_bounds) -> torch.Tensor:
    """Arc-length of points on (or clamped onto) the rectangle boundary.

    Each point is assigned to its **nearest** edge; ``s`` is measured
    CCW from the SW corner. Points strictly inside are projected onto the
    nearest edge as well (callers pass already-clamped exit points).
    """
    x0, x1 = float(x_bounds[0]), float(x_bounds[1])
    y0, y1 = float(y_bounds[0]), float(y_bounds[1])
    Lx, Ly = x1 - x0, y1 - y0
    xc = x.clamp(x0, x1) - x0
    yc = y.clamp(y0, y1) - y0
    d = torch.stack([yc, Lx - xc, Ly - yc, xc], dim=-1)  # south, east, north, west
    edge = d.argmin(dim=-1)
    s = torch.where(edge == 0, xc, torch.zeros_like(xc))
    s = torch.where(edge == 1, Lx + yc, s)
    s = torch.where(edge == 2, Lx + Ly + (Lx - xc), s)
    s = torch.where(edge == 3, 2 * Lx + Ly + (Ly - yc), s)
    return s


def outward_normal(x: torch.Tensor, y: torch.Tensor, x_bounds, y_bounds) -> tuple[torch.Tensor, torch.Tensor]:
    """Outward unit normal of the nearest rectangle edge."""
    x0, x1 = float(x_bounds[0]), float(x_bounds[1])
    y0, y1 = float(y_bounds[0]), float(y_bounds[1])
    xc = x.clamp(x0, x1) - x0
    yc = y.clamp(y0, y1) - y0
    d = torch.stack([yc, (x1 - x0) - xc, (y1 - y0) - yc, xc], dim=-1)
    edge = d.argmin(dim=-1)
    nx = torch.zeros_like(xc)
    ny = torch.zeros_like(yc)
    ny = torch.where(edge == 0, -torch.ones_like(ny), ny)
    nx = torch.where(edge == 1, torch.ones_like(nx), nx)
    ny = torch.where(edge == 2, torch.ones_like(ny), ny)
    nx = torch.where(edge == 3, -torch.ones_like(nx), nx)
    return nx, ny


def characteristic_residual(
    tokens: torch.Tensor,
    flags: torch.Tensor,
    C_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    Cb_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    S_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    lam: torch.Tensor | float,
    wind_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    x_bounds,
    y_bounds,
    n_mc: int = 1,
) -> dict[str, torch.Tensor]:
    """Residual (2) for every look-back ``k = 1 … L−1``.

    Parameters
    ----------
    tokens:
        ``(B·n_mc, L, 3)`` physical ``(x, y, t)`` from an advection
        generator; token 0 is the query. With ``n_mc > 1`` the batch is
        ``n_mc`` stacked SDE replicas of the same ``B`` queries
        (replica-major: rows ``r·B … r·B+B−1``).
    flags:
        ``(B·n_mc, L)`` bool, latched out-of-domain flags.
    C_fn, Cb_fn, S_fn:
        Physical-unit callables: ``C(x, y, t)``, ``C_b(s, t)``,
        ``S(x, y, t)``, each returning ``(N,)``.
    lam:
        Loss rate (s⁻¹), scalar tensor.
    wind_fn:
        ``(x, y, t) -> (u, v)``; used only to decide inflow vs outflow at
        exit points.

    Returns
    -------
    ``{"residual": (B, L−1), "coupled": (B, L−1) bool, "tau": (B, L−1)}``;
    ``coupled`` marks look-backs whose target used ``C_b`` (boundary
    information actually reached the query — the "boundary reach"
    diagnostic logged during training).
    """
    if tokens.dim() != 3 or tokens.size(-1) != 3:
        raise ValueError(f"tokens must be (N, L, 3), got {tuple(tokens.shape)}")
    N, L, _ = tokens.shape
    if L < 2:
        raise ValueError("characteristic residual needs L >= 2")
    if N % n_mc:
        raise ValueError("tokens batch is not a multiple of n_mc")
    B = N // n_mc
    lam_t = torch.as_tensor(lam, dtype=tokens.dtype, device=tokens.device)

    x, y, t = tokens[..., 0], tokens[..., 1], tokens[..., 2]
    flat = lambda a: a.reshape(-1)  # noqa: E731
    C_all = C_fn(flat(x), flat(y), flat(t)).reshape(N, L)
    S_all = S_fn(flat(x), flat(y), flat(t)).reshape(N, L)

    # first exit index per trajectory (L if never exits)
    f = flags.clone()
    f[:, 0] = False  # the query itself is interior by construction
    any_exit = f.any(dim=1)
    first = torch.where(any_exit, f.float().argmax(dim=1), torch.full_like(any_exit, L, dtype=torch.long))

    ar = torch.arange(N, device=tokens.device)
    dtp, dev = tokens.dtype, tokens.device
    x0b, x1b = float(x_bounds[0]), float(x_bounds[1])
    y0b, y1b = float(y_bounds[0]), float(y_bounds[1])

    # ---- exit point / time (straight-line refinement of the first exit) ----
    # The generator clamps the first out-of-domain token onto the edge at
    # the token time, which is first-order. We refine: from the last
    # inside token p = first-1, move backward along -u (wind at p) until
    # the first edge is reached; that gives the exit point and time.
    p_idx = (first - 1).clamp(min=0, max=L - 1)
    nxt = first.clamp(max=L - 1)
    xp, yp, tp = x[ar, p_idx], y[ar, p_idx], t[ar, p_idx]
    up, vp = wind_fn(xp, yp, tp)
    up = torch.as_tensor(up, dtype=dtp, device=dev).reshape(-1)
    vp = torch.as_tensor(vp, dtype=dtp, device=dev).reshape(-1)
    inf = torch.full_like(xp, float("inf"))
    eps = 1e-12
    t_w = torch.where(up > eps, (xp - x0b) / up.clamp_min(eps), inf)
    t_e = torch.where(up < -eps, (x1b - xp) / (-up).clamp_min(eps), inf)
    t_s = torch.where(vp > eps, (yp - y0b) / vp.clamp_min(eps), inf)
    t_n = torch.where(vp < -eps, (y1b - yp) / (-vp).clamp_min(eps), inf)
    tau_e = torch.stack([t_w, t_e, t_s, t_n], dim=-1).min(dim=-1).values.clamp_min(0.0)
    seg = (tp - t[ar, nxt]).abs()
    refined = torch.isfinite(tau_e) & (tau_e <= seg * (1 + 1e-9))
    tau_e = torch.where(refined, tau_e, seg)
    xe = torch.where(refined, xp - up * tau_e, x[ar, nxt]).clamp(x0b, x1b)
    ye = torch.where(refined, yp - vp * tau_e, y[ar, nxt]).clamp(y0b, y1b)
    te = tp - tau_e
    se = xy_to_arclength(xe, ye, x_bounds, y_bounds)
    nxe, nye = outward_normal(xe, ye, x_bounds, y_bounds)
    ue, ve = wind_fn(xe, ye, te)
    ue = torch.as_tensor(ue, dtype=dtp, device=dev).reshape(-1)
    ve = torch.as_tensor(ve, dtype=dtp, device=dev).reshape(-1)
    inflow_exit = (ue * nxe + ve * nye) < 0.0
    Cb_e = Cb_fn(se, te).reshape(-1)
    C_e = C_fn(xe, ye, te).reshape(-1)
    V_exit = torch.where(inflow_exit, Cb_e, C_e)

    # ---- source integral: cumulative trapezoid along the path ----
    t0 = t[:, :1]
    g = torch.exp(-lam_t * (t0 - t)) * S_all  # (N, L)
    seg_int = 0.5 * (g[:, :-1] + g[:, 1:]) * (t[:, :-1] - t[:, 1:]).abs()  # (N, L-1)
    cum = torch.cat([torch.zeros_like(seg_int[:, :1]), seg_int.cumsum(dim=1)], dim=1)  # (N, L)
    S_e = S_fn(xe, ye, te).reshape(-1)
    g_e = torch.exp(-lam_t * (t0[:, 0] - te)) * S_e
    I_exit = cum[ar, p_idx] + 0.5 * (g[ar, p_idx] + g_e) * tau_e
    tau_exit = (t0[:, 0] - te).clamp_min(1e-9)
    target_exit = torch.exp(-lam_t * tau_exit) * V_exit + I_exit

    res, coup, taus = [], [], []
    for k in range(1, L):
        exited = first <= k
        tau_k = (t0[:, 0] - t[:, k]).clamp_min(1e-9)
        target_k = torch.exp(-lam_t * tau_k) * C_all[:, k] + cum[:, k]
        target = torch.where(exited, target_exit, target_k)
        tau = torch.where(exited, tau_exit, tau_k)
        res.append(C_all[:, 0] - target)  # numerator; divided by tau below
        coup.append(exited & inflow_exit)
        taus.append(tau)
    Num = torch.stack(res, dim=1)  # (N, L-1)
    Cp = torch.stack(coup, dim=1)
    T = torch.stack(taus, dim=1)
    if n_mc > 1:
        # FK with a stopping time is exact in expectation: average the
        # numerators C(x,t) - target over SDE replicas *before* squaring,
        # then normalize by the mean look-back.
        Num = Num.reshape(n_mc, B, L - 1).mean(dim=0)
        Cp = Cp.reshape(n_mc, B, L - 1).any(dim=0)
        T = T.reshape(n_mc, B, L - 1).mean(dim=0)
    return {"residual": Num / T, "coupled": Cp, "tau": T}
