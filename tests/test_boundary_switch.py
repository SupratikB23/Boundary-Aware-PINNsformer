"""Inflow/outflow partition matches wind sign (PRD §4.5 test_boundary_switch.py)."""

import numpy as np
import pytest
import torch

from bapinnsformer.physics.boundary import (
    boundary_loss,
    dirichlet_residual,
    neumann_residual,
    normal_flux,
    partition,
    partition_boundary,
)

Q = [-2.0, -0.5, 0.0, 0.3, 1.5]


def test_partition_matches_sign_no_hysteresis():
    inflow, outflow = partition(Q, hyst=0.0)
    assert list(map(bool, np.asarray(inflow))) == [True, True, False, False, False]
    assert list(map(bool, np.asarray(outflow))) == [False, False, False, True, True]


def test_hysteresis_dead_band():
    inflow, outflow = partition(Q, hyst=1.0)
    assert list(map(bool, np.asarray(inflow))) == [True, False, False, False, False]
    assert list(map(bool, np.asarray(outflow))) == [False, False, False, False, True]


def test_partition_torch_path():
    q = torch.tensor(Q)
    inflow, outflow = partition(q, hyst=0.0)
    assert isinstance(inflow, torch.Tensor) and inflow.dtype == torch.bool
    assert inflow.tolist() == [True, True, False, False, False]
    assert outflow.tolist() == [False, False, False, True, True]


def test_normal_flux_and_full_partition():
    normals = [(1.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0)]
    q = normal_flux([3.0, -3.0, 3.0, -3.0], [0.0, 0.0, 0.0, 0.0], normals)
    np.testing.assert_allclose(np.asarray(q), [3.0, -3.0, -3.0, 3.0])
    out = partition_boundary(None, None, [3.0, -3.0, 3.0, -3.0],
                             [0.0, 0.0, 0.0, 0.0], normals, hysteresis=0.05)
    assert list(map(bool, out["inflow"])) == [False, True, True, False]
    assert list(map(bool, out["outflow"])) == [True, False, False, True]
    assert not np.asarray(out["buffer"]).any()


def test_dirichlet_only_on_inflow():
    C = [10.0, 20.0, 30.0]
    Cb = [12.0, 18.0, 33.0]
    inflow, _ = partition([-1.0, -1.0, 1.0])
    r = np.asarray(dirichlet_residual(C, Cb, inflow))
    np.testing.assert_allclose(r, [-2.0, 2.0, 0.0])


def test_neumann_only_on_outflow():
    _, outflow = partition([-1.0, 0.5, 0.0], hyst=0.0)
    r = np.asarray(neumann_residual([7.0, 8.0, 9.0], outflow))
    np.testing.assert_allclose(r, [0.0, 8.0, 0.0])


def test_boundary_loss_zero_when_satisfied():
    C = Cb = [50.0, 60.0]
    inflow, outflow = partition([-1.0, 1.0])
    dcdn = [0.0, 0.0]
    assert boundary_loss(C, Cb, dcdn, [-1.0, 1.0]) == pytest.approx(0.0)
