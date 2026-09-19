"""Baselines (PRD §§4.3, 7.1 `models/baselines.py`).

1. ``ZeroInflow`` — implicit treatment in existing physics-informed AQ work.
2. ``ClimatologicalInflow`` — seasonal/hourly mean boundary assumption.
3. ``MLPPINN`` — vanilla MLP-PINN with the identical inverse signature
   (isolates the transformer's contribution). Also exposed as the
   ``uniform_psf`` PINNsformer control via :mod:`pseudoseq`.
4. ``TrajectoryRegression`` — back-trajectory + (ridge) regression
   statistical attribution; pure-advection offline implementation so tests
   run without HYSPLIT. A full trajectory engine plugs in at the
   experiment-driver level.

All baselines expose the boundary-inflow call ``(s_hat, t_hat) -> C_b``
(torch, CPU-safe) plus numpy-friendly ``boundary_value`` aliases, so the
trainer/evaluator can swap them by config string. No baseline imports
fire data of any kind (fire observations never enter training).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .activations import get_activation

__all__ = [
    "ZeroInflow",
    "ZeroInflowBaseline",
    "ClimatologicalInflow",
    "ClimatologicalInflowBaseline",
    "MLPPINN",
    "MlpPinnBaseline",
    "UniformPsfBaseline",
    "TrajectoryRegression",
    "TrajectoryBaseline",
    "BASELINES",
]


def _col(v: Any) -> torch.Tensor:
    from ..utils.dtype import as_float_tensor

    return as_float_tensor(v).reshape(-1, 1)


class ZeroInflow(nn.Module):
    """Boundary assumption ``C_b(s, t) = 0`` everywhere."""

    name = "zero_inflow"

    def forward(self, s_hat: torch.Tensor, t_hat: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(_col(s_hat))

    def boundary_value(self, s: Any, t: Any):
        """Numpy-friendly alias of :meth:`forward` (older call sites)."""
        out = self.forward(_col(s), _col(t)).reshape(-1)
        try:
            import numpy as np

            if not isinstance(s, torch.Tensor) and not isinstance(t, torch.Tensor):
                return np.zeros(len(out))
        except ImportError:
            pass
        return out

    def count_parameters(self) -> int:
        return 0


ZeroInflowBaseline = ZeroInflow


class ClimatologicalInflow(nn.Module):
    """Constant (or fitted) climatological inflow ``C_b ≡ value``.

    Parameters
    ----------
    value / c0:
        Scalar climatology in physical concentration units.
    learnable:
        If True the value is an ``nn.Parameter``.
    """

    name = "climatological_inflow"

    def __init__(self, value: float = 50.0, learnable: bool = False,
                 c0: float | None = None) -> None:
        super().__init__()
        if c0 is not None:
            value = c0
        self.c0 = float(value)
        v = torch.tensor(float(value), dtype=torch.get_default_dtype())
        if learnable:
            self.value = nn.Parameter(v)
        else:
            self.register_buffer("value", v)

    def fit(self, C_obs: torch.Tensor) -> float:
        """Set the climatology to ``nanmean(C_obs)``; returns the value."""
        from ..utils.dtype import as_float_tensor

        C_obs = as_float_tensor(C_obs)
        valid = C_obs[~torch.isnan(C_obs)]
        if valid.numel() == 0:
            raise ValueError("no valid observations to fit climatology")
        mean = float(valid.mean().item())
        with torch.no_grad():
            self.value.fill_(mean)
        self.c0 = mean
        return mean

    def forward(self, s_hat: torch.Tensor, t_hat: torch.Tensor) -> torch.Tensor:
        s = _col(s_hat)
        return torch.full_like(s, float(self.value.item()))

    def boundary_value(self, s: Any, t: Any):
        # Single source of truth: the learnable `value` param (same as
        # forward). `c0` is only the init/fit snapshot for reporting.
        out = self.forward(_col(s), _col(t)).reshape(-1)
        try:
            import numpy as np

            if not isinstance(s, torch.Tensor) and not isinstance(t, torch.Tensor):
                return np.full(len(out), float(self.value.item()))
        except ImportError:
            pass
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


ClimatologicalInflowBaseline = ClimatologicalInflow


class MLPPINN(nn.Module):
    """Vanilla MLP-PINN ``C_net`` replacement with the identical call signature.

    Consumes the same ``tokens (B, L, 3)`` but reads only the query token
    (index 0) through an MLP — the E3 control isolating advection alignment
    and attention. Output is non-negative via softplus.
    """

    name = "mlp_pinn"

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        activation: str = "tanh",
        query_index: int = 0,
        width: int | None = None,  # alias for hidden_dim
        depth: int | None = None,  # alias for num_layers
    ) -> None:
        super().__init__()
        if width is not None:
            hidden_dim = width
        if depth is not None:
            num_layers = depth
        self.width = int(hidden_dim)
        self.depth = int(num_layers)
        self.activation = str(activation)
        self.query_index = int(query_index)
        layers: list[nn.Module] = []
        prev = 3
        for _ in range(int(num_layers)):
            layers += [nn.Linear(prev, hidden_dim), get_activation(activation)]
            prev = hidden_dim
        layers += [nn.Linear(hidden_dim, 1)]
        self.mlp = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def build(self) -> nn.Module:
        """Return the underlying MLP (older call sites)."""
        return self.mlp

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3 or tokens.size(-1) != 3:
            raise ValueError(f"expected tokens (B, L, 3), got {tuple(tokens.shape)}")
        q = tokens[:, self.query_index, :]
        return self.softplus(self.mlp(q))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


MlpPinnBaseline = MLPPINN


class UniformPsfBaseline:
    """PINNsformer with the original uniform forward pseudo-sequence (E3 control)."""

    name = "uniform_psf"

    def __init__(self, L: int = 5, dt: float = 1.0, **kwargs: Any) -> None:
        self.L = int(L)
        self.dt = float(dt)
        self.kwargs = kwargs

    def generate(self, x, y, t, **kwargs):
        from .pseudoseq import UniformForwardGenerator

        kw = dict(self.kwargs)
        kw.update(kwargs)
        return UniformForwardGenerator().generate(x, y, t, L=self.L, dt=self.dt, **kw)


class TrajectoryRegression(nn.Module):
    """Back-trajectory + ridge-regression attribution (offline, torch/numpy).

    Pure-advection trajectories (no HYSPLIT dependency) with a closed-form
    ridge fit, so unit tests run offline. The full engine plugs in at the
    experiment-driver level; use :meth:`fit`/:meth:`predict` for the
    statistical path and :meth:`forward` for the boundary-inflow call.
    """

    name = "trajectory"

    def __init__(self, coef: float = 0.0, L: int = 5, dt: float = 1.0,
                 l2: float = 1.0) -> None:
        super().__init__()
        self.coef = float(coef)
        self.L = int(L)
        self.dt = float(dt)
        self.l2 = float(l2)
        self.coef_: Any = None
        self._fitted = False

    # -- boundary-inflow call (config-swappable with CbNet) -----------------
    def forward(self, s_hat: torch.Tensor, t_hat: torch.Tensor) -> torch.Tensor:
        return torch.full_like(_col(s_hat), self.coef)

    # -- statistical path ----------------------------------------------------
    def back_trajectory(self, x, y, t, wind):
        from .pseudoseq import integrate_backward_rk2

        pts = [(float(x), float(y))]
        for k in range(1, self.L):
            xk, yk = integrate_backward_rk2(x, y, t, wind, self.dt, k)
            pts.append((xk, yk))
        return pts

    def fit(self, X, y):
        import numpy as np

        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float)
        A = Xa.T @ Xa + self.l2 * np.eye(Xa.shape[1])
        self.coef_ = np.linalg.solve(A, Xa.T @ ya)
        self._fitted = True
        return self

    def predict(self, X):
        import numpy as np

        if self.coef_ is None:
            raise RuntimeError("TrajectoryRegression must be fit before predict")
        return np.asarray(X, dtype=float) @ self.coef_

    def count_parameters(self) -> int:
        return 0


TrajectoryBaseline = TrajectoryRegression


BASELINES = (
    "zero_inflow",
    "climatological_inflow",
    "mlp_pinn",
    "uniform_psf",
    "trajectory",
)
