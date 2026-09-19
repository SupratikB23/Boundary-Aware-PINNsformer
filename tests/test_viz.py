"""Viz smoke tests: every figure function saves a file, never shows."""

import numpy as np

from bapinnsformer.viz import (
    ablation_plots,
    attention_plots,
    boundary_plots,
    fields,
    maps,
)


def test_domain_map_saves(tmp_path):
    out = maps.domain_map(
        (0.0, 60_000.0, 0.0, 60_000.0),
        interior_xy=[(20_000.0, 30_000.0), (30_000.0, 25_000.0)],
        perimeter_xy=[(2_000.0, 2_000.0), (58_000.0, 58_000.0)],
        wind_xyuv=[(30_000.0, 30_000.0, 3.0, 1.0)],
        save_path=str(tmp_path / "domain.png"),
    )
    assert out.endswith(".png")


def test_fields_and_boundary_plots_save(tmp_path):
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 100, 60), rng.uniform(0, 100, 60), rng.uniform(0, 50, 60)])
    fields.field_snapshot(pts, (0, 100, 0, 100), str(tmp_path / "field.png"))
    fields.truth_vs_recovered(pts, pts + 1.0, (0, 100, 0, 100), str(tmp_path / "tvr.png"))
    boundary_plots.hovmoller(rng.uniform(0, 100, (16, 12)), str(tmp_path / "hov.png"))
    boundary_plots.sector_timeseries(
        {"NW": [1, 2, 3], "N": [2, 2, 2], "E": [1, 1, 1], "S": [0, 1, 0], "W": [1, 0, 1]},
        str(tmp_path / "sec.png"),
    )
    boundary_plots.inflow_vs_frp([1, 2, 3, 4], [0, 5, 2, 8], save_path=str(tmp_path / "frp.png"))


def test_ablation_and_attention_plots_save(tmp_path):
    rows = [
        {"variant": "uniform_forward", "L": 3, "rmse": 9.0, "wall_s": 1.0, "peak_mb": 100.0},
        {"variant": "uniform_forward", "L": 5, "rmse": 8.0, "wall_s": 1.5, "peak_mb": 120.0},
        {"variant": "advection_backward", "L": 3, "rmse": 7.0, "wall_s": 1.2, "peak_mb": 110.0},
        {"variant": "advection_backward", "L": 5, "rmse": 6.0, "wall_s": 1.8, "peak_mb": 130.0},
    ]
    ablation_plots.accuracy_vs_length(rows, str(tmp_path / "avl.png"))
    ablation_plots.accuracy_vs_cost(rows, str(tmp_path / "avc.png"))
    attention_plots.attention_over_tokens(
        [0.5, 0.2, 0.15, 0.1, 0.05], [False, True, False, False, True],
        str(tmp_path / "att.png"),
    )
