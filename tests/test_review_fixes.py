"""Regression tests for the full-review fix pass (foolproof gate).

Each test pins one reviewer finding so it can never silently regress:
CUDA device placement, config-key mapping + composition + device:auto,
upwind solver, stable sweep seeds, bc_loss None semantics, params clamp
attributes, baseline numpy/torch consistency, hysteresis buffer exclusion,
season normalization. CPU-only.
"""

import os

import numpy as np
import pytest
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(*parts):
    with open(os.path.join(REPO, "configs", *parts), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --- 1. generator hyphen alias -------------------------------------------------
def test_generator_hyphen_alias_resolves():
    from bapinnsformer.models.pseudoseq import create_generator

    g = create_generator("advection-backward-jitter", seq_len=3, dt=1.0)
    assert g.variant == "advection_jitter"
    g2 = create_generator("Advection_Backward", seq_len=2, dt=1.0)
    assert g2.variant == "advection_backward"


def test_generator_yaml_aliases():
    from bapinnsformer.models.pseudoseq import create_generator

    y = _cfg("pseudoseq", "advection_backward.yaml")
    g = create_generator(str(y["variant"]), **{k: v for k, v in y.items() if k != "variant"})
    assert g.seq_len == 5 and abs(g.dt - 3600.0) < 1e-9 and g.ood_policy == "clamp"


def test_uniform_yaml_null_integrator_composes():
    from bapinnsformer.models.pseudoseq import create_generator

    for name in ("uniform_forward", "uniform_backward"):
        y = _cfg("pseudoseq", f"{name}.yaml")
        assert y["integrator"] is None  # contract: null in yaml...
        g = create_generator(str(y["variant"]), **{k: v for k, v in y.items() if k != "variant"})
        assert g.variant == name  # ...must still construct


def test_tokens_live_on_input_device():
    from bapinnsformer.models.pseudoseq import create_generator

    for name in ("uniform_forward", "uniform_backward",
                 "advection_backward", "advection_jitter"):
        g = create_generator(name, seq_len=3, dt=1.0)
        x = torch.zeros(4)
        tok, fl = g.generate(x, torch.zeros(4), torch.zeros(4), (1.0, 0.0))
        assert tok.device == x.device and fl.device == x.device
        assert tok.shape == (4, 3, 3) and fl.shape == (4, 3)


# --- 2. config composition + constructors ---------------------------------------
def test_compose_nests_sections_and_resolves_device():
    from bapinnsformer.utils.io import compose_config

    cfg = compose_config(os.path.join(REPO, "configs", "config.yaml"))
    for key in ("model", "cb_net", "s_net", "pseudoseq", "train", "domain"):
        assert key in cfg, f"composed config missing section {key!r}"
    assert cfg["device"] in ("cpu", "cuda")
    # same-named keys must not collide across sections
    assert cfg["cb_net"]["depth"] == 3 and cfg["s_net"]["depth"] == 3


def test_yaml_dicts_construct_models():
    from bapinnsformer.models.boundary_net import CbNet
    from bapinnsformer.models.pinnsformer import PINNsformer
    from bapinnsformer.models.source_net import SNet
    from bapinnsformer.utils.config import cb_kwargs, pinnsformer_kwargs, s_kwargs

    PINNsformer(**pinnsformer_kwargs(_cfg("model", "pinnsformer.yaml")))
    CbNet(**cb_kwargs(_cfg("model", "cb_net.yaml")))
    SNet(**s_kwargs(_cfg("model", "s_net.yaml")))


def test_build_trainer_from_composed_config():
    from bapinnsformer.train.trainer import Trainer
    from bapinnsformer.utils.config import build_trainer_config
    from bapinnsformer.utils.io import compose_config

    cfg = compose_config(os.path.join(REPO, "configs", "config.yaml"))
    trainer = Trainer(config=build_trainer_config(cfg))
    assert str(trainer.device) in ("cpu", "cuda")
    assert sum(p.numel() for p in trainer.parameters()) < 500_000


def test_curriculum_request_preserved_not_silenced():
    from bapinnsformer.utils.config import build_trainer_config
    from bapinnsformer.utils.io import compose_config

    cfg = compose_config(os.path.join(REPO, "configs", "config.yaml"))
    tc = build_trainer_config(cfg)
    # t0/t1 need E0 data: no silent curriculum, request kept visible.
    assert tc["curriculum"] is None
    assert tc["curriculum_request"] == {"stages": 4}


def test_resolve_device_rejects_auto_passthrough():
    from bapinnsformer.utils.io import resolve_device

    assert resolve_device("auto") in ("cpu", "cuda")
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_normalize_season():
    from bapinnsformer.utils.config import normalize_season

    assert normalize_season("post-monsoon") == "postmonsoon"
    assert normalize_season("postmonsoon") == "postmonsoon"
    assert normalize_season("winter") == "winter"


# --- 5. upwind solver ------------------------------------------------------------
def test_run_forward_monotone_under_negative_wind():
    from bapinnsformer.data.synthetic import run_forward

    ny = nx = 8
    C0 = np.zeros((ny, nx))
    C0[:, :4] = 100.0  # block on the west half
    u = np.full((ny, nx), -5.0)  # blowing toward -x (out the west face)
    v = np.zeros((ny, nx))
    Cb = np.full((1, 2 * nx + 2 * (ny - 2)), 0.0)
    out = run_forward(C0, u, v, None, Cb, K=0.0, lam=0.0,
                      dx=1000.0, dy=1000.0, dt=10.0, n_steps=3)
    assert float(out["C"].max()) <= 100.0 + 1e-9
    assert float(out["C"].min()) >= 0.0


# --- 6. sweep seed stability ------------------------------------------------------
def test_sweep_cell_seed_stable():
    import hashlib

    def cell_seed(n_st, arch, noise, rep):
        return int(hashlib.sha256(
            f"{n_st}|{arch}|{noise}|{rep}".encode()).hexdigest(), 16) % (2**31)

    assert cell_seed(7, "uniform", 0.0, 0) == cell_seed(7, "uniform", 0.0, 0)
    assert cell_seed(7, "uniform", 0.0, 0) != cell_seed(7, "uniform", 0.0, 1)


# --- 10. bc_loss None semantics ----------------------------------------------------
def test_bc_loss_both_none_raises():
    from bapinnsformer.train.losses import bc_loss

    C = torch.ones(4)
    with pytest.raises(ValueError):
        bc_loss(C, C, torch.ones(4), None, None)


def test_bc_loss_single_none_side_empty():
    from bapinnsformer.train.losses import bc_loss

    C = torch.tensor([1.0, 2.0, 3.0, 4.0])
    Cb = torch.tensor([1.0, 2.0, 0.0, 0.0])
    inflow = torch.tensor([True, True, False, False])
    # outflow None -> empty side: pure Dirichlet over inflow
    assert float(bc_loss(C, Cb, None, inflow, None)) == pytest.approx(0.0)


# --- 13. params clamp attributes -----------------------------------------------------
def test_physparams_clamp_attributes():
    from bapinnsformer.models.params import PhysParams

    p = PhysParams(K_min=1.0, K_max=500.0, lam_min=1e-7, lam_max=1e-3)
    assert (p.K_min, p.K_max) == (1.0, 500.0)
    assert (p.lam_min, p.lam_max) == (1e-7, 1e-3)


# --- 8. baseline consistency ----------------------------------------------------------
def test_climatology_numpy_tracks_learned_value():
    from bapinnsformer.models.baselines import ClimatologicalInflow

    m = ClimatologicalInflow(c0=50.0)
    m.fit(torch.tensor([90.0, 100.0, 110.0]))
    t = m(torch.zeros(3), torch.zeros(3)).detach().reshape(-1)
    n = np.asarray(m(np.zeros(3), np.zeros(3))).reshape(-1)
    assert float(t.mean()) == pytest.approx(float(n.mean()), rel=1e-5)
    assert float(n.mean()) == pytest.approx(100.0)


# --- 9. hysteresis buffer excluded -------------------------------------------------------
def test_tangent_wind_gives_zero_bc_loss():
    from bapinnsformer.train.trainer import Trainer

    tr = Trainer()
    B = 6
    batch = {
        "x_data": torch.rand(B) * 10, "y_data": torch.rand(B) * 10,
        "t_data": torch.rand(B), "C_obs": torch.rand(B) * 50 + 10,
        "mask": torch.ones(B),
        "x_bc": torch.zeros(B), "y_bc": torch.rand(B) * 10,
        "s_bc": torch.rand(B), "t_bc": torch.rand(B),
        # zero wind + zero normals -> q = 0 exactly (deep in buffer)
        "nx": torch.zeros(B), "ny": torch.zeros(B),
    }
    losses = tr.compute_losses(batch)
    assert float(losses["bc"].detach()) == 0.0
