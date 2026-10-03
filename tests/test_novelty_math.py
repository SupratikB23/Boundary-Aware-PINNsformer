"""Verification tests for the revised mathematical core (RUN_PLAN §1, §4).

Every contribution claimed in the paper has a test here that would fail if
the mathematics were wrong:

* C1  characteristic / Feynman–Kac residual — exactly zero on exact
      solutions, including paths that exit through the inflow boundary.
* C2  Green's functions + identifiability theory — linearity,
      superposition, wind-switched boundaries, principal angles, Bayesian
      posterior, confounding coefficient.
* C3  receptor superposition attribution — exact closure.
* Trainer — dimensionless scaling and a manufactured-solution check of the
      autodiff PDE residual through the normalizer chain rule.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from bapinnsformer.physics.greens import GridTransport, build_greens, segment_of_s


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _rotating_wind(model, n_hours, speed=4.0, turn_deg_per_h=20.0):
    th = np.radians(30.0 + turn_deg_per_h * np.arange(n_hours + 1))
    U = np.stack([np.full((model.ny, model.nx), speed * np.cos(a)) for a in th])
    V = np.stack([np.full((model.ny, model.nx), speed * np.sin(a)) for a in th])
    return U, V


@pytest.fixture
def small_model():
    return GridTransport(nx=9, ny=9, Lx=40_000.0, Ly=40_000.0, K=50.0, lam=2e-5)


# ----------------------------------------------------------------------
# C2: forward operator
# ----------------------------------------------------------------------
def test_uniform_state_is_steady(small_model):
    """C ≡ c with C_b ≡ c, S = 0, λ = 0 must stay c (consistency of BCs + stencil)."""
    m = GridTransport(9, 9, 40_000.0, 40_000.0, K=50.0, lam=0.0)
    U, V = _rotating_wind(m, 3)
    out = m.run(U, V, np.full((m.ny, m.nx), 70.0), lambda t: np.full((1, m.nb), 70.0), None, 3,
                record_fields=True)
    assert np.allclose(out["fields"], 70.0, atol=1e-9)


def test_superposition_exact(small_model):
    m = small_model
    U, V = _rotating_wind(m, 5)
    rng = np.random.default_rng(0)
    cb = rng.uniform(10, 90, m.nb)
    S = rng.uniform(0, 1e-3, (m.ny, m.nx))
    C0 = rng.uniform(20, 60, (m.ny, m.nx))
    run = lambda c0, b, s: m.run(U, V, c0, lambda t: b[None], (lambda t: s[None]) if s is not None else None,
                                 5, record_fields=True)["fields"]
    full = run(C0, cb, S)
    parts = run(np.zeros_like(C0), cb, None) + run(np.zeros_like(C0), np.zeros(m.nb), S) \
        + run(C0, np.zeros(m.nb), None)
    assert np.allclose(full, parts, rtol=1e-10, atol=1e-9)


def test_greens_matrix_reproduces_direct_run(small_model):
    """y = A_b c_b + A_s s + A_0 c_0 equals a direct forward run (linearity of the basis)."""
    m = small_model
    H, bw, sw = 6, 3, 3
    U, V = _rotating_wind(m, H)
    obs = np.array([[10e3, 12e3], [20e3, 20e3], [33e3, 8e3]])
    sys = build_greens(m, U, V, H, obs, n_bseg=4, bwin_h=bw, src_patches=(2, 2), swin_h=sw,
                       ic_patches=(2, 2))
    rng = np.random.default_rng(1)
    theta = rng.uniform(0.5, 2.0, sys.G.shape[1])
    y_lin = sys.G @ theta

    nb_, ns_, n0_ = sys.sizes
    nbw = math.ceil(H / bw)
    nsw = math.ceil(H / sw)
    node_seg = segment_of_s(m.b_s, m.perimeter, 4)
    pi = np.minimum((np.arange(m.nx) * 2) // m.nx, 1)
    pj = np.minimum((np.arange(m.ny) * 2) // m.ny, 1)
    patch = pj[:, None] * 2 + pi[None, :]

    def Cb_fn(t):
        w = min(int(t // (bw * 3600)), nbw - 1)
        coef = theta[:nb_].reshape(4, nbw)[:, w]
        return coef[node_seg][None]

    def S_fn(t):
        w = min(int(t // (sw * 3600)), nsw - 1)
        coef = theta[nb_:nb_ + ns_].reshape(4, nsw)[:, w]
        return (coef[patch] / 3600.0)[None]

    C0 = theta[nb_ + ns_:][patch]
    from bapinnsformer.physics.greens import bilinear_weights

    idx, w, _ = bilinear_weights(obs[:, 0], obs[:, 1], m.nx, m.ny, m.dx, m.dy)
    direct = m.run(U, V, C0, Cb_fn, S_fn, H, obs_idx=idx, obs_w=w)["obs"][:, 0, :].reshape(-1)
    assert np.allclose(y_lin, direct, rtol=1e-9, atol=1e-9)


def test_outflow_boundary_has_no_influence():
    """Constant eastward wind: east-face boundary values never reach the interior."""
    m = GridTransport(11, 11, 50_000.0, 50_000.0, K=0.0, lam=0.0)
    H = 4
    U = np.full((H + 1, m.ny, m.nx), 5.0)
    V = np.zeros_like(U)
    obs = np.array([[25e3, 25e3], [40e3, 20e3]])
    sys = build_greens(m, U, V, H, obs, n_bseg=8, bwin_h=H, src_patches=(1, 1), swin_h=H,
                       ic_patches=(1, 1))
    # segments 2,3 cover the east edge (s in [Lx, Lx+Ly)) for a square domain with 8 segments
    east_cols = np.isin(sys.b_seg, [2, 3])
    west_cols = np.isin(sys.b_seg, [6, 7])
    assert np.abs(sys.A_b[:, east_cols]).max() == 0.0
    assert np.abs(sys.A_b[:, west_cols]).max() > 0.1
    assert np.all(sys.b_inflow_frac[east_cols] == 0.0)
    # the NW corner node belongs to the north edge (normal +y), so segment 6 is 4/5 inflow
    assert np.all(sys.b_inflow_frac[west_cols] >= 0.8)


# ----------------------------------------------------------------------
# C2: identifiability theory
# ----------------------------------------------------------------------
def test_principal_angles_limits():
    from bapinnsformer.eval.identifiability import principal_angles

    rng = np.random.default_rng(0)
    A = rng.standard_normal((20, 3))
    assert np.allclose(principal_angles(A, A @ rng.standard_normal((3, 3))), 0.0, atol=1e-6)
    Q, _ = np.linalg.qr(rng.standard_normal((20, 6)))
    assert np.allclose(principal_angles(Q[:, :3], Q[:, 3:]), np.pi / 2, atol=1e-8)


def test_bayes_inversion_recovers_truth():
    from bapinnsformer.eval.identifiability import bayes_linear_inversion

    rng = np.random.default_rng(2)
    G = rng.standard_normal((200, 10))
    th = rng.uniform(1, 3, 10)
    m, S = bayes_linear_inversion(G, G @ th, np.zeros(10), np.full(10, 1e6), 1e-10)
    assert np.allclose(m, th, atol=1e-5)
    assert np.all(np.linalg.eigvalsh(S) > 0)


def test_confounding_detects_identical_footprints():
    """If source and boundary footprints are identical, their contributions are confounded (ρ≈−1)."""
    from bapinnsformer.eval.identifiability import bayes_linear_inversion, confounding_coefficient
    from bapinnsformer.physics.greens import GreensSystem

    rng = np.random.default_rng(3)
    a = rng.uniform(0.5, 1.5, (50, 1))
    sys = GreensSystem(A_b=a, A_s=a.copy(), A_0=np.zeros((50, 0)), R_b=np.ones(1), R_s=np.ones(1),
                       R_0=np.zeros(0), b_seg=np.zeros(1, int), b_win=np.zeros(1, int),
                       b_inflow_frac=np.ones(1), s_patch=np.zeros(1, int), s_win=np.zeros(1, int))
    _, S = bayes_linear_inversion(sys.G, np.zeros(50), np.zeros(2), np.ones(2), 1e-4)
    assert confounding_coefficient(sys, S) < -0.99


def test_identifiability_report_on_real_greens(small_model):
    from bapinnsformer.eval.identifiability import identifiability_report

    m = small_model
    H = 8
    U, V = _rotating_wind(m, H, turn_deg_per_h=40.0)
    obs = np.array([[x, y] for x in (10e3, 20e3, 30e3) for y in (10e3, 20e3, 30e3)])
    sys = build_greens(m, U, V, H, obs, n_bseg=4, bwin_h=4, src_patches=(2, 2), swin_h=8,
                       ic_patches=(1, 1), spinup_h=2)
    n = sys.G.shape[1]
    rng = np.random.default_rng(4)
    th = rng.uniform(20, 60, n)
    th[sys.sizes[0]:sys.sizes[0] + sys.sizes[1]] = rng.uniform(0, 5, sys.sizes[1])
    y = sys.G @ th + rng.normal(0, 0.5, sys.G.shape[0])
    rep = identifiability_report(sys, np.full(n, 30.0), np.full(n, 400.0), 0.25, y=y, theta_true=th)
    for k in ("min_angle_deg", "confounding_rho", "boundary_observability", "share_est",
              "share_post_q05", "share_post_q95"):
        assert np.isfinite(rep[k]), k
    assert 0.0 <= rep["share_post_q05"] <= rep["share_post_q95"] <= 1.0
    assert -1.0 <= rep["confounding_rho"] <= 1.0


# ----------------------------------------------------------------------
# C3: superposition attribution
# ----------------------------------------------------------------------
def test_receptor_share_closure_and_limits(small_model):
    from bapinnsformer.eval.attribution import receptor_share_superposition

    m = small_model
    H = 6
    U, V = _rotating_wind(m, H)
    rec = np.array([[20e3, 20e3], [15e3, 25e3]])
    Cb = np.full((H + 1, m.nb), 50.0)
    S = np.full((H + 1, m.ny, m.nx), 5e-4)
    C0 = np.full((m.ny, m.nx), 30.0)
    out = receptor_share_superposition(m, U, V, Cb, S, C0, rec, H, spinup_h=2)
    assert out["closure_rel"] < 1e-10
    assert out["share"] + out["share_src"] + out["share_ic"] == pytest.approx(1.0, abs=1e-9)
    z = receptor_share_superposition(m, U, V, np.zeros_like(Cb), S, C0, rec, H)
    assert z["share"] == pytest.approx(0.0, abs=1e-12)
    one = receptor_share_superposition(m, U, V, Cb, np.zeros_like(S), np.zeros_like(C0), rec, H)
    assert one["share"] == pytest.approx(1.0, abs=1e-12)


# ----------------------------------------------------------------------
# C1: characteristic / Feynman–Kac residual
# ----------------------------------------------------------------------
U0, V0, LAM = 4.0, -1.5, 3e-5
XB, YB = (0.0, 60_000.0), (0.0, 50_000.0)


def _exact(x, y, t):
    """Exact solution of C_t + u·∇C = −λC (K = 0, S = 0)."""
    xi = x - U0 * t
    eta = y - V0 * t
    return torch.exp(-LAM * t) * (80.0 + 20.0 * torch.sin(xi / 9000.0) + 10.0 * torch.cos(eta / 7000.0))


def _exact_cb(s, t):
    from bapinnsformer.physics.characteristic import xy_to_arclength  # noqa: F401

    Lx, Ly = XB[1], YB[1]
    x = torch.where(s < Lx, s, torch.where(s < Lx + Ly, torch.full_like(s, Lx),
                    torch.where(s < 2 * Lx + Ly, Lx - (s - Lx - Ly), torch.zeros_like(s))))
    y = torch.where(s < Lx, torch.zeros_like(s), torch.where(s < Lx + Ly, s - Lx,
                    torch.where(s < 2 * Lx + Ly, torch.full_like(s, Ly), Ly - (s - 2 * Lx - Ly))))
    return _exact(x, y, t)


def test_characteristic_residual_zero_on_exact_solution_with_exits():
    from bapinnsformer.models.pseudoseq import AdvectionBackwardGenerator
    from bapinnsformer.physics.characteristic import characteristic_residual

    torch.set_default_dtype(torch.float64)
    try:
        gen = AdvectionBackwardGenerator(seq_len=6, dt=3600.0, x_bounds=XB, y_bounds=YB,
                                         integrator="rk4", substeps=2)
        x = torch.tensor([5_000.0, 30_000.0, 55_000.0, 12_000.0])
        y = torch.tensor([45_000.0, 25_000.0, 5_000.0, 40_000.0])
        t = torch.full((4,), 50_000.0)
        tok, fl = gen.generate(x, y, t, (U0, V0))
        assert fl.any(), "test geometry must include boundary exits"
        out = characteristic_residual(
            tok, fl, C_fn=_exact, Cb_fn=_exact_cb,
            S_fn=lambda a, b, c: torch.zeros_like(a), lam=LAM,
            wind_fn=lambda a, b, c: (torch.full_like(a, U0), torch.full_like(b, V0)),
            x_bounds=XB, y_bounds=YB,
        )
        assert out["coupled"].any()
        assert out["residual"].abs().max().item() < 1e-9
        # a wrong boundary function must be detected
        bad = characteristic_residual(
            tok, fl, C_fn=_exact, Cb_fn=lambda s, tt: _exact_cb(s, tt) + 10.0,
            S_fn=lambda a, b, c: torch.zeros_like(a), lam=LAM,
            wind_fn=lambda a, b, c: (torch.full_like(a, U0), torch.full_like(b, V0)),
            x_bounds=XB, y_bounds=YB,
        )
        assert bad["residual"][out["coupled"]].abs().min().item() > 1e-4
    finally:
        torch.set_default_dtype(torch.float32)


def test_characteristic_residual_with_constant_source():
    """C = c∞ + (C0 − c∞) e^{−λt} with S = λ c∞ (uniform): residual 0 for interior paths."""
    from bapinnsformer.models.pseudoseq import AdvectionBackwardGenerator
    from bapinnsformer.physics.characteristic import characteristic_residual

    torch.set_default_dtype(torch.float64)
    try:
        lam, cinf, c0 = 1e-5, 60.0, 20.0  # realistic deposition time scale (~1 day)
        Cfn = lambda a, b, c: cinf + (c0 - cinf) * torch.exp(-lam * c)  # noqa: E731
        gen = AdvectionBackwardGenerator(seq_len=4, dt=1800.0, x_bounds=(0, 1e6), y_bounds=(0, 1e6))
        x = torch.tensor([5e5, 4e5])
        y = torch.tensor([5e5, 6e5])
        t = torch.tensor([20_000.0, 30_000.0])
        tok, fl = gen.generate(x, y, t, (2.0, 1.0))
        assert not fl.any()
        out = characteristic_residual(
            tok, fl, C_fn=Cfn, Cb_fn=lambda s, tt: torch.zeros_like(s),
            S_fn=lambda a, b, c: torch.full_like(a, lam * cinf), lam=lam,
            wind_fn=lambda a, b, c: (torch.full_like(a, 2.0), torch.full_like(b, 1.0)),
            x_bounds=(0, 1e6), y_bounds=(0, 1e6),
        )
        # trapezoid on an exponential: error O((λΔt)²)·λc∞ only (~1e-8 here)
        assert out["residual"].abs().max().item() < 1e-7
    finally:
        torch.set_default_dtype(torch.float32)


# ----------------------------------------------------------------------
# Trainer: scaling + manufactured PDE residual
# ----------------------------------------------------------------------
class _AnalyticC(torch.nn.Module):
    """Returns Ĉ = C_exact(query)/C_ref from normalized tokens (reads token 0)."""

    def __init__(self, norm, cref, u, v, lam):
        super().__init__()
        self.norm, self.cref, self.u, self.v, self.lam = norm, cref, u, v, lam
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, tok):
        x = self.norm.denormalize_x(tok[:, 0, 0])
        y = self.norm.denormalize_y(tok[:, 0, 1])
        t = self.norm.denormalize_t(tok[:, 0, 2])
        # linear profile: zero Laplacian; exact for C_t + u·∇C = −λC
        C = torch.exp(-self.lam * t) * (100.0 + 1e-3 * (x - self.u * t) + 2e-3 * (y - self.v * t))
        return (C / self.cref + 0.0 * self.dummy).unsqueeze(-1)


class _ZeroS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, y, t):
        return torch.zeros_like(torch.as_tensor(x)).reshape(-1, 1) + 0.0 * self.dummy


def test_trainer_pde_residual_manufactured_solution():
    from bapinnsformer.models.normalizer import Normalizer
    from bapinnsformer.models.params import PhysParams
    from bapinnsformer.train.trainer import Trainer

    torch.set_default_dtype(torch.float64)
    try:
        u, v, lam = 3.0, -2.0, 2e-5
        norm = Normalizer((0.0, 80_000.0), (0.0, 60_000.0), (0.0, 86_400.0))
        cfg = {"scales": {"mode": "fixed", "C_ref": 75.0, "T_ref": 3600.0, "L_ref": 1e4},
               "pseudoseq": {"name": "uniform_forward", "seq_len": 3, "dt": 3600.0}}
        tr = Trainer(config=cfg, normalizer=norm, wind_field=(u, v),
                     C_net=_AnalyticC(norm, 75.0, u, v, lam), S_net=_ZeroS(),
                     phys_params=PhysParams(K_init=150.0, lam_init=lam, learn_K=False, learn_lam=False))
        x = torch.tensor([10_000.0, 40_000.0, 70_000.0])
        y = torch.tensor([5_000.0, 30_000.0, 55_000.0])
        t = torch.tensor([3_600.0, 40_000.0, 80_000.0])
        r = tr.pde_residual(x, y, t)
        assert r.abs().max().item() < 1e-9
    finally:
        torch.set_default_dtype(torch.float32)


def test_trainer_auto_scale_balances_terms():
    """With auto scaling the data loss is O(1) for realistic µg/m³ values, not O(10⁴)."""
    from bapinnsformer.train.trainer import Trainer

    tr = Trainer(config={"scales": {"mode": "auto"}})
    B = 16
    obs = torch.rand(B) * 200 + 100
    batch = {"x_data": torch.rand(B), "y_data": torch.rand(B), "t_data": torch.rand(B),
             "C_obs": obs, "mask": torch.ones(B),
             "x_pde": torch.rand(B), "y_pde": torch.rand(B), "t_pde": torch.rand(B)}
    losses = tr.compute_losses(batch)
    assert tr.C_ref == pytest.approx(float(obs.median()), rel=1e-6)
    assert float(losses["data"]) < 10.0
    assert np.isfinite(float(losses["pde"]))


def test_trainer_characteristic_loss_runs_and_reports_reach():
    from bapinnsformer.models.normalizer import Normalizer
    from bapinnsformer.train.trainer import Trainer

    norm = Normalizer((0.0, 50_000.0), (0.0, 50_000.0), (0.0, 86_400.0), s_bounds=(0.0, 200_000.0))
    cfg = {"train": {"w_char": 1.0}, "scales": {"mode": "auto"},
           "pseudoseq": {"name": "advection_backward", "seq_len": 4, "dt": 3600.0}}
    tr = Trainer(config=cfg, normalizer=norm, wind_field=(5.0, 0.0))
    B = 12
    batch = {"x_data": torch.rand(B) * 5e4, "y_data": torch.rand(B) * 5e4,
             "t_data": torch.rand(B) * 8e4 + 4e3, "C_obs": torch.rand(B) * 50 + 80,
             "mask": torch.ones(B),
             "x_pde": torch.rand(B) * 5e4, "y_pde": torch.rand(B) * 5e4,
             "t_pde": torch.rand(B) * 8e4 + 4e3}
    info = tr.train_step(batch)
    assert "char" in info and np.isfinite(info["char"])
    assert 0.0 <= info["boundary_reach"] <= 1.0


def test_char_loss_requires_advection_generator():
    from bapinnsformer.train.trainer import Trainer

    with pytest.raises(ValueError):
        Trainer(config={"train": {"w_char": 1.0},
                        "pseudoseq": {"name": "uniform_forward", "seq_len": 3, "dt": 3600.0}})


def test_checkpoint_roundtrip_weights_only(tmp_path):
    """Checkpoints must load with torch.load(weights_only=True) (no pickle RCE)."""
    from bapinnsformer.models.normalizer import Normalizer
    from bapinnsformer.train.checkpoint import load_checkpoint, save_checkpoint
    from bapinnsformer.train.trainer import Trainer

    tr = Trainer()
    p = save_checkpoint(tmp_path / "c.pt", tr.C_net, tr.Cb_net, tr.S_net, tr.phys,
                        Normalizer((0, 1), (0, 1), (0, 1)), {"a": 1, "b": [1.0, 2]})
    payload = load_checkpoint(p, tr.C_net, tr.Cb_net, tr.S_net, tr.phys)
    assert payload["config"]["a"] == 1


def test_e1_null_boundary_identifiable_with_perimeter_stations():
    """E1 sanity: dense perimeter network + rotating wind recovers a zero boundary share."""
    from bapinnsformer.eval.e1_benchmark import (
        E1Settings, place_stations, regrid_uniform, run_cell, synthetic_wind_window,
    )

    st = E1Settings(n_coarse=9, n_hours=48, spinup_h=12)
    rng = np.random.default_rng(0)
    u, v = synthetic_wind_window(st.n_hours, "high", rng)
    Uc, Vc = regrid_uniform(u, v, 9, 9, 0.2)
    Uf, Vf = regrid_uniform(u, v, 17, 17, 0.2)
    S = place_stations(40, "perimeter_biased", st.Lx, st.Ly, rng)
    R = place_stations(5, "clustered", st.Lx, st.Ly, rng)
    r = run_cell(st, Uc, Vc, Uf, Vf, S, R, 0.05, np.random.default_rng(1),
                 kind="null_boundary", misspec=False)
    assert r["share_true"] == 0.0
    assert r["oracle_share_est"] < 0.10
    assert r["oracle_prior_share"] > 0.5  # so passing is the data's doing, not the prior's


def test_qc_keeps_multi_hour_episode_but_flags_isolated_glitch():
    import pandas as pd

    from bapinnsformer.data.qc import detect_spikes

    base = 80 + 5 * np.sin(np.arange(72) / 3.0)
    x = base.copy()
    x[20] = 900.0                 # single-hour glitch
    x[45:51] = [300, 420, 500, 480, 390, 310]  # 6-h smoke episode
    sp = detect_spikes(pd.Series(x))
    assert sp.iloc[20]
    assert not sp.iloc[45:51].any()
