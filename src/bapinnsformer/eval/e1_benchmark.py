"""E1 synthetic identifiability benchmark (rewritten 2026-10-03).

The previous sweep added noise to the true field and called that a
"recovery error": no inversion was performed and station count / wind had
no effect. This module replaces it with a real, falsifiable experiment.

Per cell ``(n_stations, archetype, wind window, noise, replicate)``:

1. **Truth** is simulated on a *fine* grid (2× resolution) with smooth
   ``C_b(s, t)`` (background + north-westerly pulse), Gaussian-plume
   sources with a diurnal cycle and a smooth initial field. The inversion
   basis (coarse grid, piecewise-constant segments/windows/patches) cannot
   represent this truth exactly — no "inverse crime".
2. **Observations** = fine-grid concentration at station points, hourly,
   plus Gaussian noise (``noise × mean``) and random missingness.
3. **Oracle**: the closed-form Bayesian linear inversion on the coarse
   Green's system with *data-derived* priors (never truth-derived). With
   the true ``K, λ`` it is the best achievable linear-Gaussian estimator;
   a second, deliberately **misspecified** oracle (``K×2, λ×0.5``) shows
   robustness.
4. **Truth share** comes from exact superposition on the *fine* model
   (``eval/attribution.receptor_share_superposition``), so share errors
   include discretization and representation error.
5. Optional **PINNsformer** fit on the same data (``run_pinn_cell``).

Null tests (``kind`` = ``null_boundary`` / ``null_source``) set the
boundary inflow or the sources to zero in the truth; the inversion must
then return a share near 0 / near 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..physics.greens import GridTransport, bilinear_weights, build_greens, segment_of_s
from .attribution import receptor_share_superposition
from .identifiability import directional_entropy, identifiability_report, receptor_share

ARCHETYPES = ("clustered", "uniform", "perimeter_biased")

__all__ = [
    "ARCHETYPES",
    "E1Settings",
    "place_stations",
    "synthetic_wind_window",
    "regrid_uniform",
    "window_entropy",
    "make_truth",
    "run_cell",
    "run_pinn_cell",
]


@dataclass
class E1Settings:
    Lx: float = 150_000.0
    Ly: float = 155_000.0
    n_coarse: int = 17
    fine_factor: int = 2
    K: float = 100.0
    lam: float = 1.0e-5
    n_hours: int = 96
    spinup_h: int = 24
    n_bseg: int = 12
    bwin_h: int = 6
    src_patches: tuple[int, int] = (4, 4)
    swin_h: int = 24
    ic_patches: tuple[int, int] = (2, 2)
    missing_frac: float = 0.2
    noise_floor: float = 1.0  # µg m⁻³ representation-error floor in the oracle


# ----------------------------------------------------------------------
# geometry + wind
# ----------------------------------------------------------------------
def place_stations(n: int, archetype: str, Lx: float, Ly: float, rng: np.random.Generator) -> np.ndarray:
    """Station coordinates ``(n, 2)`` for an archetype.

    * ``clustered`` — Delhi-like: Gaussian cluster at the centre (σ = 12 % of size).
    * ``uniform`` — uniform over the interior (5 % margin).
    * ``perimeter_biased`` — half within 15 % of an edge, half uniform.
    """
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype {archetype!r}")
    m = 0.05
    if archetype == "clustered":
        pts = rng.normal([0.5 * Lx, 0.5 * Ly], [0.12 * Lx, 0.12 * Ly], (n, 2))
    elif archetype == "uniform":
        pts = rng.uniform([m * Lx, m * Ly], [(1 - m) * Lx, (1 - m) * Ly], (n, 2))
    else:
        k = n // 2
        edge = rng.integers(0, 4, k)
        u = rng.uniform(m, 1 - m, k)
        d = rng.uniform(m, 0.15, k)
        x = np.where(edge == 0, u, np.where(edge == 1, 1 - d, np.where(edge == 2, u, d))) * Lx
        y = np.where(edge == 0, d, np.where(edge == 1, u, np.where(edge == 2, 1 - d, u))) * Ly
        rest = rng.uniform([m * Lx, m * Ly], [(1 - m) * Lx, (1 - m) * Ly], (n - k, 2))
        pts = np.vstack([np.stack([x, y], 1), rest])
    return np.clip(pts, [m * Lx, m * Ly], [(1 - m) * Lx, (1 - m) * Ly])


def synthetic_wind_window(n_hours: int, level: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Domain-mean hourly ``(u, v)`` series (``n_hours+1``) with controlled rotation.

    Base direction: from the north-west (post-monsoon Delhi). ``level``:
    ``low`` ±15° wobble, ``mid`` ±70°, ``high`` steady rotation through
    360° over the window. Speed 2–5 m s⁻¹. Used only when the real
    station-wind field is not yet available; every row records which.
    """
    t = np.arange(n_hours + 1)
    base = math.radians(315.0)  # wind FROM NW
    if level == "low":
        frm = base + math.radians(15) * np.sin(2 * np.pi * t / 37.0 + rng.uniform(0, 6.28))
    elif level == "mid":
        frm = base + math.radians(70) * np.sin(2 * np.pi * t / 41.0 + rng.uniform(0, 6.28))
    elif level == "high":
        frm = base + 2 * np.pi * t / n_hours + rng.uniform(0, 6.28)
    else:
        raise ValueError("level must be low|mid|high")
    spd = 3.5 + 1.5 * np.sin(2 * np.pi * t / 24.0 + rng.uniform(0, 6.28))
    # meteorological FROM direction -> vector components
    return -spd * np.sin(frm), -spd * np.cos(frm)


def regrid_uniform(u: np.ndarray, v: np.ndarray, nx: int, ny: int, shear: float = 0.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast domain-mean series to a grid with optional linear shear (±shear fraction)."""
    gx = np.linspace(-1, 1, nx)[None, None, :]
    gy = np.linspace(-1, 1, ny)[None, :, None]
    fac = 1.0 + shear * (0.5 * gx + 0.5 * gy)
    return u[:, None, None] * fac, v[:, None, None] * fac


def window_entropy(U: np.ndarray, V: np.ndarray, n_bins: int = 8) -> float:
    """Directional entropy (nats) of the hourly domain-mean wind direction."""
    um = U.reshape(U.shape[0], -1).mean(1)
    vm = V.reshape(V.shape[0], -1).mean(1)
    return directional_entropy(np.arctan2(vm, um), n_bins=n_bins)


# ----------------------------------------------------------------------
# truth
# ----------------------------------------------------------------------
def make_truth(model: GridTransport, n_hours: int, rng: np.random.Generator, kind: str = "full") -> dict:
    """Smooth truth on ``model``'s grid: hourly ``Cb (H+1, nb)``, ``S (H+1, ny, nx)``, ``C0``."""
    if kind not in ("full", "null_boundary", "null_source"):
        raise ValueError("kind must be full|null_boundary|null_source")
    H = n_hours
    t = np.arange(H + 1, dtype=float)
    s = model.b_s / model.perimeter  # fraction of perimeter, CCW from SW
    # NW corner sits at s = (2Lx+Ly)/P
    s_nw = (2 * model.Lx + model.Ly) / model.perimeter
    ds = np.minimum(np.abs(s - s_nw), 1 - np.abs(s - s_nw))
    bg = rng.uniform(40, 80) * (1 + 0.15 * np.sin(2 * np.pi * (s[None, :] + t[:, None] / 53.0)))
    amp = rng.uniform(60, 160)
    tc, tw = rng.uniform(0.3, 0.7) * H, rng.uniform(0.1, 0.25) * H
    pulse = amp * np.exp(-(ds[None, :] / 0.08) ** 2) * np.exp(-((t[:, None] - tc) / tw) ** 2)
    Cb = bg + pulse
    X, Y = np.meshgrid(model.x, model.y)
    S = np.zeros((H + 1, model.ny, model.nx))
    n_src = int(rng.integers(2, 5))
    diurnal = 1.0 + 0.6 * np.sin(2 * np.pi * (t - 6) / 24.0)
    for _ in range(n_src):
        cx, cy = rng.uniform(0.3, 0.7) * model.Lx, rng.uniform(0.3, 0.7) * model.Ly
        sig = rng.uniform(0.04, 0.10) * min(model.Lx, model.Ly)
        a = rng.uniform(4, 15) / 3600.0  # µg m⁻³ h⁻¹ at the plume centre
        S += a * diurnal[:, None, None] * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sig**2))[None]
    C0 = rng.uniform(50, 90) * (1 + 0.2 * np.sin(X / model.Lx * 3) * np.cos(Y / model.Ly * 2))
    if kind == "null_boundary":
        Cb = np.zeros_like(Cb)
    if kind == "null_source":
        S = np.zeros_like(S)
    return {"Cb": Cb, "S": S, "C0": C0}


def _interp_hourly(arr: np.ndarray, t: float, H: int) -> np.ndarray:
    h = min(int(t // 3600.0), H - 1)
    a = (t - h * 3600.0) / 3600.0
    return (1 - a) * arr[h] + a * arr[h + 1]


def _project_truth(sys, coarse: GridTransport, fine: GridTransport, truth: dict, st: E1Settings
                   ) -> np.ndarray:
    """Truth coefficients in the coarse basis (window/segment/patch averages)."""
    H = st.n_hours
    nb, ns, n0 = sys.sizes
    nbw = math.ceil(H / st.bwin_h)
    nsw = math.ceil(H / st.swin_h)
    seg_f = segment_of_s(fine.b_s, fine.perimeter, st.n_bseg)
    cb = np.zeros(nb)
    for c in range(nb):
        g, w = sys.b_seg[c], sys.b_win[c]
        hrs = slice(w * st.bwin_h, min((w + 1) * st.bwin_h, H) + 1)
        cb[c] = truth["Cb"][hrs][:, seg_f == g].mean()
    npx, npy = st.src_patches
    pi = np.minimum((np.arange(fine.nx) * npx) // fine.nx, npx - 1)
    pj = np.minimum((np.arange(fine.ny) * npy) // fine.ny, npy - 1)
    pmap = pj[:, None] * npx + pi[None, :]
    sv = np.zeros(ns)
    for c in range(ns):
        p, w = sys.s_patch[c], sys.s_win[c]
        hrs = slice(w * st.swin_h, min((w + 1) * st.swin_h, H) + 1)
        sv[c] = truth["S"][hrs][:, pmap == p].mean() * 3600.0
    nip, njp = st.ic_patches
    ii = np.minimum((np.arange(fine.nx) * nip) // fine.nx, nip - 1)
    jj = np.minimum((np.arange(fine.ny) * njp) // fine.ny, njp - 1)
    imap = jj[:, None] * nip + ii[None, :]
    c0 = np.array([truth["C0"][imap == p].mean() for p in range(n0)])
    return np.concatenate([cb, sv, c0])


# ----------------------------------------------------------------------
# one cell
# ----------------------------------------------------------------------
def run_cell(
    st: E1Settings,
    U_coarse: np.ndarray,
    V_coarse: np.ndarray,
    U_fine: np.ndarray,
    V_fine: np.ndarray,
    stations: np.ndarray,
    receptors: np.ndarray,
    noise: float,
    rng: np.random.Generator,
    kind: str = "full",
    misspec: bool = True,
) -> dict:
    """Simulate truth, observe, invert with the oracle; return one tidy row (+ arrays)."""
    H = st.n_hours
    nf = (st.n_coarse - 1) * st.fine_factor + 1
    fine = GridTransport(nf, nf, st.Lx, st.Ly, st.K, st.lam)
    coarse = GridTransport(st.n_coarse, st.n_coarse, st.Lx, st.Ly, st.K, st.lam)
    truth = make_truth(fine, H, rng, kind=kind)

    # observations from the fine truth
    idx, w, _ = bilinear_weights(stations[:, 0], stations[:, 1], fine.nx, fine.ny, fine.dx, fine.dy)
    run = fine.run(U_fine, V_fine, truth["C0"],
                   lambda t: _interp_hourly(truth["Cb"], t, H)[None],
                   lambda t: _interp_hourly(truth["S"], t, H)[None],
                   H, obs_idx=idx, obs_w=w)
    clean = run["obs"][st.spinup_h:, 0, :].reshape(-1)  # rows (hour, station)
    scale = max(float(np.mean(np.abs(clean))), 1e-6)
    y = clean + rng.normal(0.0, noise * scale, clean.shape)
    keep = rng.uniform(size=y.shape) >= st.missing_frac

    # true receptor share by exact superposition on the fine model
    share_true = receptor_share_superposition(
        fine, U_fine, V_fine, truth["Cb"], truth["S"], truth["C0"], receptors, H,
        spinup_h=st.spinup_h)

    row: dict = {"kind": kind, "n_obs_rows": int(keep.sum()), "obs_mean": float(y[keep].mean()),
                 "share_true": share_true["share"], "share_true_ic": share_true["share_ic"]}

    def oracle(Kf: float, lf: float, tag: str):
        mdl = GridTransport(st.n_coarse, st.n_coarse, st.Lx, st.Ly, Kf, lf)
        sys = build_greens(mdl, U_coarse, V_coarse, H, stations, receptor_xy=receptors,
                           n_bseg=st.n_bseg, bwin_h=st.bwin_h, src_patches=st.src_patches,
                           swin_h=st.swin_h, ic_patches=st.ic_patches, spinup_h=st.spinup_h)
        sys.A_b, sys.A_s, sys.A_0 = sys.A_b[keep], sys.A_s[keep], sys.A_0[keep]
        nb, ns, n0 = sys.sizes
        ybar = float(y[keep].mean())
        # data-derived, weakly informative priors (never truth-derived)
        mu = np.concatenate([np.full(nb, ybar), np.full(ns, 0.05 * ybar), np.full(n0, ybar)])
        pv = np.concatenate([np.full(nb, ybar**2), np.full(ns, (0.2 * ybar) ** 2),
                             np.full(n0, (0.5 * ybar) ** 2)])
        nv = (noise * scale) ** 2 + st.noise_floor**2
        th_true = _project_truth(sys, mdl, fine, truth, st)
        rep = identifiability_report(sys, mu, pv, nv, y=y[keep], theta_true=th_true)
        # what the prior alone would say; if share_est ≈ prior_share the data did not decide
        rep["prior_share"] = receptor_share(sys, mu)
        rep["data_shift"] = abs(rep["share_est"] - rep["prior_share"])
        if kind != "full":
            # relative errors against an all-zero block are undefined
            rep.pop("rel_l2_cb_inflow" if kind == "null_boundary" else "rel_l2_s", None)
        for k, v in rep.items():
            if not k.startswith("_"):
                row[f"{tag}_{k}"] = v
        row[f"{tag}_share_err_vs_fine"] = abs(rep["share_est"] - share_true["share"])
        row[f"{tag}_covered90_vs_fine"] = bool(
            rep["share_post_q05"] <= share_true["share"] <= rep["share_post_q95"])
        return sys, rep

    oracle(st.K, st.lam, "oracle")
    if misspec:
        oracle(st.K * 2.0, st.lam * 0.5, "misspec")
    row["_truth"] = truth
    row["_y"] = y
    row["_keep"] = keep
    return row


# ----------------------------------------------------------------------
# optional PINNsformer fit on the same synthetic cell
# ----------------------------------------------------------------------
def run_pinn_cell(st: E1Settings, row: dict, U_fine: np.ndarray, V_fine: np.ndarray,
                  stations: np.ndarray, receptors: np.ndarray, variant: str = "advection_backward",
                  steps: int = 2000, w_char: float = 1.0, seed: int = 0, device: str = "cpu") -> dict:
    """Fit the PINNsformer to the cell's observations and score it like the oracle.

    The recovered ``C_b``, ``S`` and ``C0`` are pushed through the *fine*
    linear model by superposition to obtain the share — the same rule as
    the truth — so PINN, oracle and truth are compared on one definition.
    """
    import torch

    from ..data.wind_field import WindField
    from ..models.normalizer import Normalizer
    from ..train.trainer import Trainer

    torch.manual_seed(seed)
    H = st.n_hours
    nf = U_fine.shape[1]
    fine = GridTransport(nf, nf, st.Lx, st.Ly, st.K, st.lam)
    tgrid = np.arange(H + 1) * 3600.0
    wf = WindField(U_fine, V_fine, fine.x, fine.y, tgrid, device=device)
    norm = Normalizer((0.0, st.Lx), (0.0, st.Ly), (0.0, H * 3600.0), s_bounds=(0.0, fine.perimeter))
    cfg = {
        "seed": seed, "device": device,
        "pseudoseq": {"name": variant, "seq_len": 5, "dt": 3600.0, "integrator": "rk2", "substeps": 2},
        "train": {"steps": steps, "lr": 1e-3, "w_char": w_char, "w_reg": 1e-2, "balance_every": 100},
        "scales": {"mode": "auto"},
        "phys": {"K_init": 50.0, "lam_init": 2e-5},
    }
    tr = Trainer(config=cfg, normalizer=norm, wind_field=wf)

    y, keep = row["_y"], row["_keep"]
    hrs = np.arange(st.spinup_h, H) + 1  # obs recorded at hour ends
    T, Xs = np.meshgrid(hrs * 3600.0, np.arange(stations.shape[0]), indexing="ij")
    xs = stations[Xs.reshape(-1), 0][keep]
    ys = stations[Xs.reshape(-1), 1][keep]
    ts = T.reshape(-1)[keep]
    obs = y[keep]
    rng = np.random.default_rng(seed)

    def batch(i):
        B = 512
        j = rng.integers(0, obs.size, B)
        sb = rng.uniform(0, fine.perimeter, 256)
        # rectangle edge points from arc-length (SW-corner origin)
        Lx, Ly = st.Lx, st.Ly
        xb = np.where(sb < Lx, sb, np.where(sb < Lx + Ly, Lx, np.where(sb < 2 * Lx + Ly, Lx - (sb - Lx - Ly), 0.0)))
        yb = np.where(sb < Lx, 0.0, np.where(sb < Lx + Ly, sb - Lx, np.where(sb < 2 * Lx + Ly, Ly, Ly - (sb - 2 * Lx - Ly))))
        nxv = np.where(sb < Lx, 0.0, np.where(sb < Lx + Ly, 1.0, np.where(sb < 2 * Lx + Ly, 0.0, -1.0)))
        nyv = np.where(sb < Lx, -1.0, np.where(sb < Lx + Ly, 0.0, np.where(sb < 2 * Lx + Ly, 1.0, 0.0)))
        f = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
        return {
            "x_data": f(xs[j]), "y_data": f(ys[j]), "t_data": f(ts[j]), "C_obs": f(obs[j]),
            "mask": torch.ones(B),
            "x_pde": f(rng.uniform(0, Lx, 512)), "y_pde": f(rng.uniform(0, Ly, 512)),
            "t_pde": f(rng.uniform(0, H * 3600.0, 512)),
            "x_bc": f(xb), "y_bc": f(yb), "s_bc": f(sb), "t_bc": f(rng.uniform(0, H * 3600.0, 256)),
            "nx": f(nxv), "ny": f(nyv),
        }

    hist = tr.fit(batch, steps=steps)
    tr.eval_mode()
    with torch.no_grad():
        Kf, lf = (float(v) for v in tr.phys.forward())
    est_model = GridTransport(nf, nf, st.Lx, st.Ly, Kf, lf)
    Cb_h = np.stack([tr.predict_Cb(torch.as_tensor(fine.b_s), torch.full((fine.nb,), h * 3600.0)
                                   ).detach().cpu().numpy() for h in range(H + 1)])
    Xg, Yg = np.meshgrid(fine.x, fine.y)
    S_h = np.stack([tr.predict_S(torch.as_tensor(Xg.ravel()), torch.as_tensor(Yg.ravel()),
                                 torch.full((Xg.size,), h * 3600.0)).detach().cpu().numpy()
                    .reshape(fine.ny, fine.nx) for h in range(H + 1)])
    C0 = tr.predict_C(torch.as_tensor(Xg.ravel()), torch.as_tensor(Yg.ravel()),
                      torch.zeros(Xg.size)).detach().cpu().numpy().reshape(fine.ny, fine.nx)
    sh = receptor_share_superposition(est_model, U_fine, V_fine, Cb_h, S_h, C0, receptors, H,
                                      spinup_h=st.spinup_h)
    truth = row["_truth"]
    u_dn = np.stack([fine.u_dot_n(U_fine[h], V_fine[h]) for h in range(H + 1)])
    inflow = u_dn < 0
    err_cb = float(np.linalg.norm((Cb_h - truth["Cb"])[inflow]) /
                   max(np.linalg.norm(truth["Cb"][inflow]), 1e-12))
    last = hist[-1] if hist else {}
    return {
        "pinn_variant": variant,
        "pinn_steps": steps,
        "pinn_w_char": w_char,
        "pinn_share_est": sh["share"],
        "pinn_share_err_vs_fine": abs(sh["share"] - row["share_true"]),
        "pinn_rel_l2_cb_inflow_nodes": err_cb,
        "pinn_K": Kf,
        "pinn_lam": lf,
        "pinn_final_total": float(last.get("total", float("nan"))),
        "pinn_boundary_reach": float(last.get("boundary_reach", float("nan"))),
    }
