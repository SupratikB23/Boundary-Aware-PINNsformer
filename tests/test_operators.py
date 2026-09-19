"""Autodiff derivatives vs analytic derivatives (PRD §4.5 test_operators.py)."""

import numpy as np
import pytest
import torch

from bapinnsformer.physics.operators import (
    divergence_2d,
    grad_scalar,
    laplacian,
)

torch.set_default_dtype(torch.float64)


def _X(n: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 3, generator=g, dtype=torch.float64) * 2 - 1).requires_grad_(True)


def test_grad_linear_is_exact():
    X = _X()
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] - 1.5 * X[:, 2] + 5.0
    g = grad_scalar(y, X)
    np.testing.assert_allclose(
        g.detach().numpy(),
        np.tile([2.0, 3.0, -1.5], (X.shape[0], 1)),
        rtol=1e-12, atol=1e-12,
    )


def test_grad_trig_is_exact():
    X = _X()
    y = torch.sin(X[:, 0]) * torch.cos(X[:, 1]) * torch.exp(-X[:, 2])
    g = grad_scalar(y, X)
    x, yy, t = (a.detach().numpy() for a in (X[:, 0], X[:, 1], X[:, 2]))
    expect = np.stack([
        np.cos(x) * np.cos(yy) * np.exp(-t),
        -np.sin(x) * np.sin(yy) * np.exp(-t),
        -np.sin(x) * np.cos(yy) * np.exp(-t),
    ], axis=1)
    np.testing.assert_allclose(g.detach().numpy(), expect, rtol=1e-10, atol=1e-10)


def test_divergence_quadratic():
    X = _X()
    Fx, Fy = X[:, 0] ** 2, X[:, 1] ** 2
    div = divergence_2d(Fx, Fy, X)
    expect = 2 * X[:, 0] + 2 * X[:, 1]
    np.testing.assert_allclose(div.detach().numpy(), expect.detach().numpy(), rtol=1e-12, atol=1e-12)


def test_laplacian_quadratic_is_constant():
    X = _X()
    y = X[:, 0] ** 2 + 2.0 * X[:, 1] ** 2 + 3.0 * X[:, 2] ** 2
    lap = laplacian(y, X, n_spatial=2)  # dxx + dyy = 2 + 4
    np.testing.assert_allclose(
        lap.detach().numpy(), np.full(X.shape[0], 6.0), rtol=1e-10, atol=1e-10
    )


def test_requires_grad_enforced():
    X = torch.zeros(4, 3, dtype=torch.float64)  # requires_grad False
    with pytest.raises(ValueError):
        grad_scalar(torch.zeros(4, dtype=torch.float64), X)
