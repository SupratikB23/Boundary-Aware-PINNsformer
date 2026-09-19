"""GPU-resident wind interpolator shared by PDE + pseudo-sequences.

PRD §4.3 `data/wind_field.py`: wraps the ERA5 wind into a
differentiable, GPU-resident interpolator with a `sample(x, y, t)`
contract. This is the performance-critical path for backward
characteristic integration (every token of every collocation point).
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch required by PRD model plane
    torch = None  # type: ignore[assignment]


class WindField:
    """Bilinear-in-space, linear-in-time wind lookup.

    Grids:
        u_grid, v_grid: (Nt, Ny, Nx) arrays/tensors in m/s.
        x_coords: (Nx,) metres, strictly increasing.
        y_coords: (Ny,) metres, strictly increasing.
        t_coords: (Nt,) seconds (float, e.g. epoch seconds), increasing.

    ``device`` selects torch device (``"cpu"`` default; pass
    ``"cuda"`` on GPU builds). All state is stored as torch tensors
    when torch is available, otherwise as numpy arrays with an
    equivalent numpy sampling path.
    """

    def __init__(
        self,
        u_grid: np.ndarray,
        v_grid: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        t_coords: np.ndarray,
        device: str = "cpu",
    ) -> None:
        u = np.asarray(u_grid, dtype=np.float64)
        v = np.asarray(v_grid, dtype=np.float64)
        x = np.asarray(x_coords, dtype=np.float64)
        y = np.asarray(y_coords, dtype=np.float64)
        t = np.asarray(t_coords, dtype=np.float64)
        if u.shape != v.shape:
            raise ValueError(f"WindField: u/v shape mismatch {u.shape} vs {v.shape}")
        if u.ndim != 3:
            raise ValueError(f"WindField: expected (Nt,Ny,Nx), got {u.shape}")
        nt, ny, nx = u.shape
        if x.size != nx or y.size != ny or t.size != nt:
            raise ValueError("WindField: coord sizes inconsistent with grid shape")
        if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0) and (nt == 1 or np.all(np.diff(t) > 0))):
            raise ValueError("WindField: coords must be strictly increasing")
        self.nx, self.ny, self.nt = nx, ny, nt
        self.device = device
        if torch is not None:
            dev = torch.device(device) if isinstance(device, str) else device
            self._torch = True
            self.u = torch.as_tensor(u, dtype=torch.float64).to(dev)
            self.v = torch.as_tensor(v, dtype=torch.float64).to(dev)
            self.x = torch.as_tensor(x, dtype=torch.float64).to(dev)
            self.y = torch.as_tensor(y, dtype=torch.float64).to(dev)
            self.t = torch.as_tensor(t, dtype=torch.float64).to(dev)
            self.device_t = dev
        else:  # numpy-only fallback (CPU, no autograd)
            self._torch = False
            self.u, self.v, self.x, self.y, self.t = u, v, x, y, t
            self.device_t = None

    # -- helpers -----------------------------------------------------
    def _interp_torch(self, field: "torch.Tensor", xq: "torch.Tensor", yq: "torch.Tensor", tq: "torch.Tensor") -> "torch.Tensor":
        # Spatial indices via searchsorted (vectorized).
        ix1 = torch.searchsorted(self.x, xq)
        iy1 = torch.searchsorted(self.y, yq)
        ix0 = torch.clamp(ix1 - 1, 0, self.nx - 2 if self.nx > 1 else 0)
        ix1c = torch.clamp(ix1, 0 if self.nx == 1 else 1, self.nx - 1)
        iy0 = torch.clamp(iy1 - 1, 0, self.ny - 2 if self.ny > 1 else 0)
        iy1c = torch.clamp(iy1, 0 if self.ny == 1 else 1, self.ny - 1)
        x0, x1 = self.x[ix0], self.x[ix1c]
        y0, y1 = self.y[iy0], self.y[iy1c]
        dx = torch.where(x1 > x0, x1 - x0, torch.ones_like(x0))
        dy = torch.where(y1 > y0, y1 - y0, torch.ones_like(y0))
        wx = torch.clamp((xq - x0) / dx, 0.0, 1.0)
        wy = torch.clamp((yq - y0) / dy, 0.0, 1.0)
        # Time bracketing.
        if self.nt == 1:
            k0 = torch.zeros_like(tq, dtype=torch.long)
            k1 = torch.zeros_like(tq, dtype=torch.long)
            a = torch.zeros_like(tq)
        else:
            k1 = torch.searchsorted(self.t, tq)
            k1c = torch.clamp(k1, 1, self.nt - 1)
            k0 = k1c - 1
            t0, t1 = self.t[k0], self.t[k1c]
            denom = torch.where(t1 > t0, t1 - t0, torch.ones_like(t0))
            a = torch.clamp((tq - t0) / denom, 0.0, 1.0)
            k0, k1 = k0, k1c
        # Gather 8 corners: field[k, j, i].
        def g(k: "torch.Tensor", j: "torch.Tensor", i: "torch.Tensor") -> "torch.Tensor":
            return field[k, j, i]

        f00_0 = g(k0, iy0, ix0)
        f10_0 = g(k0, iy0, ix1c)
        f01_0 = g(k0, iy1c, ix0)
        f11_0 = g(k0, iy1c, ix1c)
        f00_1 = g(k1, iy0, ix0)
        f10_1 = g(k1, iy0, ix1c)
        f01_1 = g(k1, iy1c, ix0)
        f11_1 = g(k1, iy1c, ix1c)
        s0 = (1 - wx) * (1 - wy) * f00_0 + wx * (1 - wy) * f10_0 + (1 - wx) * wy * f01_0 + wx * wy * f11_0
        s1 = (1 - wx) * (1 - wy) * f00_1 + wx * (1 - wy) * f10_1 + (1 - wx) * wy * f01_1 + wx * wy * f11_1
        return (1 - a) * s0 + a * s1

    def _interp_numpy(self, field: np.ndarray, xq: np.ndarray, yq: np.ndarray, tq: np.ndarray) -> np.ndarray:
        ix1 = np.searchsorted(self.x, xq, side="left")
        iy1 = np.searchsorted(self.y, yq, side="left")
        ix0 = np.clip(ix1 - 1, 0, max(self.nx - 2, 0))
        ix1c = np.clip(ix1, 0 if self.nx == 1 else 1, self.nx - 1)
        iy0 = np.clip(iy1 - 1, 0, max(self.ny - 2, 0))
        iy1c = np.clip(iy1, 0 if self.ny == 1 else 1, self.ny - 1)
        x0, x1 = self.x[ix0], self.x[ix1c]
        y0, y1 = self.y[iy0], self.y[iy1c]
        wx = np.clip(np.where(x1 > x0, (xq - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0), 0, 1)
        wy = np.clip(np.where(y1 > y0, (yq - y0) / np.where(y1 > y0, y1 - y0, 1.0), 0.0), 0, 1)
        if self.nt == 1:
            k0 = np.zeros_like(tq, dtype=int)
            k1 = np.zeros_like(tq, dtype=int)
            a = np.zeros_like(tq)
        else:
            k1 = np.searchsorted(self.t, tq, side="left")
            k1c = np.clip(k1, 1, self.nt - 1)
            k0 = k1c - 1
            k1 = k1c
            t0, t1 = self.t[k0], self.t[k1]
            a = np.clip(np.where(t1 > t0, (tq - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0), 0, 1)
        s0 = ((1 - wx) * (1 - wy) * field[k0, iy0, ix0] + wx * (1 - wy) * field[k0, iy0, ix1c]
              + (1 - wx) * wy * field[k0, iy1c, ix0] + wx * wy * field[k0, iy1c, ix1c])
        s1 = ((1 - wx) * (1 - wy) * field[k1, iy0, ix0] + wx * (1 - wy) * field[k1, iy0, ix1c]
              + (1 - wx) * wy * field[k1, iy1c, ix0] + wx * wy * field[k1, iy1c, ix1c])
        return (1 - a) * s0 + a * s1

    # -- public contract ---------------------------------------------
    def sample(self, x: Any, y: Any, t: Any) -> tuple[Any, Any]:
        """Sample ``(u, v)`` at query points ``(x, y, t)``.

        Accepts torch tensors (returned as tensors on the field device,
        differentiable w.r.t. field values) or numpy arrays/scalars
        (returned as ndarrays). Shapes broadcast against each other.
        Out-of-domain queries are clamped to the grid edge (documented
        policy; pseudo-sequence code flags boundary exits separately).
        """
        if self._torch and torch is not None and isinstance(x, torch.Tensor):
            dev = self.device_t
            xq = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x, dtype=torch.float64, device=dev).reshape(-1)
            yq = torch.as_tensor(np.asarray(y) if not isinstance(y, torch.Tensor) else y, dtype=torch.float64, device=dev).reshape(-1)
            tq = torch.as_tensor(np.asarray(t) if not isinstance(t, torch.Tensor) else t, dtype=torch.float64, device=dev).reshape(-1)
            try:
                b = np.broadcast_shapes(xq.shape, yq.shape, tq.shape)
                xq = xq.expand(b).reshape(-1)
                yq = yq.expand(b).reshape(-1)
                tq = tq.expand(b).reshape(-1)
            except Exception:
                pass
            # clamp to domain (documented policy)
            xq = torch.clamp(xq, self.x.min(), self.x.max())
            yq = torch.clamp(yq, self.y.min(), self.y.max())
            tq = torch.clamp(tq, self.t.min(), self.t.max())
            return self._interp_torch(self.u, xq, yq, tq), self._interp_torch(self.v, xq, yq, tq)
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        ta = np.asarray(t, dtype=float)
        bshape = np.broadcast_shapes(xa.shape, ya.shape, ta.shape)
        xq = np.broadcast_to(xa, bshape).ravel()
        yq = np.broadcast_to(ya, bshape).ravel()
        tq = np.broadcast_to(ta, bshape).ravel()
        if self._torch and torch is not None:
            # torch-backed field, numpy query → compute in numpy via cpu copies
            u = self.u.detach().cpu().numpy()
            v = self.v.detach().cpu().numpy()
            xx = self.x.detach().cpu().numpy()
            yy = self.y.detach().cpu().numpy()
            tt = self.t.detach().cpu().numpy()
            keep = (self.u, self.v, self.x, self.y, self.t)
            self_np = (u, v, xx, yy, tt)
            # temporary swap for numpy interp
            old = (self.u, self.v, self.x, self.y, self.t)
            self.u, self.v, self.x, self.y, self.t = self_np  # type: ignore[assignment]
            try:
                ru = self._interp_numpy(u, xq, yq, tq).reshape(bshape)
                rv = self._interp_numpy(v, xq, yq, tq).reshape(bshape)
            finally:
                self.u, self.v, self.x, self.y, self.t = old
            return ru, rv
        xq = np.clip(xq, float(np.min(self.x)), float(np.max(self.x)))
        yq = np.clip(yq, float(np.min(self.y)), float(np.max(self.y)))
        tq = np.clip(tq, float(np.min(self.t)), float(np.max(self.t)))
        return self._interp_numpy(self.u, xq, yq, tq).reshape(bshape), self._interp_numpy(self.v, xq, yq, tq).reshape(bshape)

    @classmethod
    def constant(cls, u0: float, v0: float, x: np.ndarray, y: np.ndarray, t: np.ndarray, device: str = "cpu") -> "WindField":
        """Convenience: spatially/temporally uniform wind (tests, E1 toy)."""
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        ta = np.asarray(t, dtype=float)
        u = np.full((ta.size, ya.size, xa.size), float(u0))
        v = np.full((ta.size, ya.size, xa.size), float(v0))
        return cls(u, v, xa, ya, ta, device=device)
