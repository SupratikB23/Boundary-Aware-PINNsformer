"""Method of manufactured solutions (PRD §4.5 test_pde_residual.py).

Non-negotiable: for an analytic ``C`` satisfying the PDE with its
analytic source ``S``, the assembled residual must be ≈ 0.
PDE: dC/dt + u·∇C (+ C·div u, div-free here) = K·ΔC + S − lam·C.
"""

import math

import numpy as np
import torch

from bapinnsformer.physics.operators import pde_residual_fields

torch.set_default_dtype(torch.float64)

A_WIND = 2.0
B_WIND = -0.5
K_DIFF = 25.0
LAM = 1e-5


def _manufactured(n: int = 32, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g, dtype=torch.float64) * 0.8 + 0.1
    t = torch.rand(n, 1, generator=g, dtype=torch.float64)
    X = torch.cat([xy, t], dim=1).requires_grad_(True)
    x, y, tt = X[:, 0], X[:, 1], X[:, 2]
    C = torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.exp(-tt)
    # Analytic derivatives of the manufactured solution.
    Cx = math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y) * torch.exp(-tt)
    Cy = math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y) * torch.exp(-tt)
    Ct = -C
    Lap = -2.0 * math.pi**2 * C
    S = Ct + A_WIND * Cx + B_WIND * Cy - K_DIFF * Lap + LAM * C
    U = torch.stack([
        torch.full_like(x, A_WIND), torch.full_like(x, B_WIND),
    ], dim=1)
    return C, X, U, S


def test_manufactured_residual_is_zero():
    C, X, U, S = _manufactured()
    r = pde_residual_fields(C, X, U, S, K_DIFF, LAM)
    assert r.shape == (X.shape[0],)
    np.testing.assert_allclose(
        r.detach().numpy(), np.zeros(X.shape[0]), rtol=0, atol=1e-9
    )


def test_wrong_source_gives_nonzero_residual():
    C, X, U, S = _manufactured()
    r = pde_residual_fields(C, X, U, torch.zeros_like(S), K_DIFF, LAM)
    assert float(torch.abs(r).max()) > 1e-3
