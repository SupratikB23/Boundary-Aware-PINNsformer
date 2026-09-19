"""Eval-plane + utils contracts (metrics, attribution, identifiability, io)."""

import math

import numpy as np
import pytest

from bapinnsformer.eval import attribution as A
from bapinnsformer.eval import compare_external as CE
from bapinnsformer.eval import identifiability as I
from bapinnsformer.eval import profiling as P
from bapinnsformer.eval.metrics import (
    identifiability_margin,
    mae,
    r2,
    rel_l2,
    rmse,
    spearman_with_p,
)
from bapinnsformer.utils.io import validate_results_record
from bapinnsformer.utils.provenance import collect_provenance, config_hash


def test_regression_metrics_exact_and_noisy():
    true = [1.0, 2.0, 3.0, 4.0]
    assert rel_l2(true, true) == 0.0
    assert rmse(true, true) == 0.0 and mae(true, true) == 0.0
    assert r2(true, true) == pytest.approx(1.0)
    est = [1.1, 1.9, 3.2, 3.8]
    assert rmse(est, true) == pytest.approx(math.sqrt(0.025))
    assert r2([2.5, 2.5, 2.5, 2.5], true) == pytest.approx(0.0)
    # NaN pairs are masked (never imputed): valid pairs still score ...
    assert rmse([1.0, float("nan")], [1.0, 99.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):  # ... but all-NaN input is an error.
        rmse([float("nan")], [float("nan")])


def test_spearman_perfect_and_null():
    rho, p = spearman_with_p([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert rho == pytest.approx(1.0) and p <= 0.05
    rho, _ = spearman_with_p([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
    assert rho == pytest.approx(-1.0)
    rng = np.random.default_rng(0)
    rho, _ = spearman_with_p(rng.normal(size=200), rng.normal(size=200))
    assert abs(rho) < 0.25


def test_identifiability_margin_sign():
    assert identifiability_margin(0.1, 0.2) > 0
    assert identifiability_margin(0.1, 0.3, threshold=0.25) < 0


def test_share_accounting_and_sector_partition():
    n_s, n_t = 36, 10
    rng = np.random.default_rng(1)
    Cb = rng.uniform(20, 100, (n_s, n_t))
    udn = np.full((n_s, n_t), -2.0)  # uniform inflow
    ds = np.full(n_s, 500.0)
    share, acc = A.transboundary_share(Cb, udn, ds, S_xyt=np.full((4, 4, n_t), 1e-3),
                                       dx=1000.0, dy=1000.0, dt=3600.0)
    assert 0.0 < share < 1.0
    assert acc["M_in"] > 0 and acc["M_S"] > 0
    # Sectors partition the boundary: masses sum to the total.
    theta = np.linspace(0, 2 * math.pi, n_s, endpoint=False)
    dec = A.decompose_by_sector(Cb, udn, ds, np.cos(theta), np.sin(theta), dt=3600.0)
    total = dec["accounting"]["M_in_total"]
    assert dec["accounting"]["sectors_sum"] == pytest.approx(total, rel=1e-9)
    assert abs(sum(dec[s]["share_of_inflow"] for s in A.SECTORS) - 1.0) < 1e-9
    # Sector labels cover the compass exactly once.
    assert {A.sector_of_bearing(b) for b in range(0, 360, 5)} == set(A.SECTORS)


def test_divergence_table_and_latex():
    rows = CE.divergence_table({"Oct-Nov PM2.5": 0.42}, {"DSS": {"Oct-Nov PM2.5": 0.40}})
    assert rows[0]["agreement"] == "agree"
    rows = CE.divergence_table({"Oct-Nov PM2.5": 0.9}, {"DSS": {"Oct-Nov PM2.5": 0.1}})
    assert rows[0]["agreement"] == "divergent"
    assert "\\begin{tabular}" in CE.to_latex(rows)


def test_identifiability_surface_direction():
    rng = np.random.default_rng(2)
    cells = []
    for n in (7, 15, 25, 40):
        for rep in range(3):
            ent = float(rng.uniform(0.5, 2.0))
            noise = float(rng.uniform(0.0, 0.2))
            err = 2.0 / math.sqrt(n) * math.exp(-0.3 * ent) * (1 + noise)
            cells.append({"n_stations": n, "config": "uniform",
                          "entropy": ent, "noise": noise, "err": err})
    surf = I.fit_surface(cells)
    assert surf["r2"] > 0.8
    assert surf["interpretation"]["more_stations_help"]
    assert surf["interpretation"]["entropy_helps"]
    assert I.predict_surface(surf, 25, "uniform", 1.5, 0.05) > 0
    assert math.isfinite(I.stations_for_threshold(surf, "uniform", 1.5, 0.05))


def test_profiling_and_provenance_record():
    with P.Timer() as t:
        s = sum(range(1000))
    assert t.elapsed >= 0 and s == 499500
    assert P.count_parameters(torch_linear())["trainable"] > 0
    cfg = {"seed": 0, "model": {"L": 5}}
    assert len(config_hash(cfg)) == 64
    rec = {"run_id": "r", "config": cfg,
           "provenance": collect_provenance(cfg, run_id="r"), "metrics": {"rmse": 1.0}}
    validate_results_record(rec)
    with pytest.raises(ValueError):
        validate_results_record({"run_id": "r"})


def torch_linear():
    import torch

    return torch.nn.Linear(4, 2)
