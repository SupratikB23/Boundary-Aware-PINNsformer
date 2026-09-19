"""Pseudo-sequence generators (PRD §§2.3, 4.3 — headline methodological file).

A generator takes a query collocation point ``(x, y, t)`` (physical units)
plus a wind field and returns ``L`` tokens ``(B, L, 3)`` and boundary-hit
flags ``(B, L)`` bool.

Variants (all behind :func:`create_generator` / :func:`get_generator`)
-----------------------------------------------------------------------
* ``uniform_forward``  (PINNsformer original): ``(x, y, t + k·dt)``
* ``uniform_backward``:                          ``(x, y, t - k·dt)``
* ``advection_backward`` (**ours**):             ``(x_k, y_k, t - k·dt)``
  where ``(x_k, y_k)`` integrates ``-u`` from ``(x, y, t)`` over
  ``[t - k·dt, t]`` with RK2/RK4 on the bilinearly-interpolated wind.
* ``advection_jitter`` (**ours**): as above plus an isotropic Gaussian
  random-walk jitter of per-step scale ``sqrt(2·K·dt)`` (diffusive spread).

Token convention: index 0 is the query point; index ``k`` is ``k`` steps
into the past (or future for ``uniform_forward``).

Wind-field contract
-------------------
Any object with ``sample(x, y, t) -> (u, v)`` (torch tensors, physical
units) — see ``data/wind_field.py``. For offline/numpy call sites a
constant ``(u, v)`` tuple or a callable ``(x, y, t) -> (u, v)`` is also
accepted. Out-of-domain characteristics are clamped (default) or reflected
to the boundary and flagged — flagged tokens carry ``C_b`` information
(PRD §2.3) and feed the boundary loss + attention-attribution figure.

Return type: torch tensors when any input is a torch tensor, else numpy
arrays (same numerics) so scalar/offline call sites keep working.
"""

from __future__ import annotations

import abc
import math
from typing import Any, Literal

import torch

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]

__all__ = [
    "PseudoSequenceGenerator",
    "UniformForwardGenerator",
    "UniformBackwardGenerator",
    "AdvectionBackwardGenerator",
    "AdvectionJitterGenerator",
    "VARIANTS",
    "create_generator",
    "get_generator",
    "integrate_backward_rk2",
]

OODPolicy = Literal["clamp", "reflect"]


# ----------------------------------------------------------------------
# wind resolution + scalar RK2 helper (older offline call sites)
# ----------------------------------------------------------------------
def _wind_call(wind: Any, x: float, y: float, t: float) -> tuple[float, float]:
    """Evaluate ``wind`` (sampler object / callable / constant) at scalars."""
    if wind is None:
        return 0.0, 0.0
    if hasattr(wind, "sample"):
        u, v = wind.sample(
            torch.tensor([x]), torch.tensor([y]), torch.tensor([t])
        )
        return float(torch.as_tensor(u).reshape(-1)[0]), float(
            torch.as_tensor(v).reshape(-1)[0]
        )
    if callable(wind):
        u, v = wind(float(x), float(y), float(t))
        return float(u), float(v)
    u, v = wind
    return float(u), float(v)


def integrate_backward_rk2(
    x: float, y: float, t: float, wind: Any, dt: float, steps: int,
    substeps: int = 1,
) -> tuple[float, float]:
    """Scalar midpoint (RK2) backward integration of ``-u`` over ``steps·dt``."""
    h = float(dt) / max(1, int(substeps))
    xc, yc, tc = float(x), float(y), float(t)
    for _ in range(int(steps) * max(1, int(substeps))):
        u1, v1 = _wind_call(wind, xc, yc, tc)
        xm, ym, tm = xc - 0.5 * h * u1, yc - 0.5 * h * v1, tc - 0.5 * h
        u2, v2 = _wind_call(wind, xm, ym, tm)
        xc, yc, tc = xc - h * u2, yc - h * v2, tc - h
    return xc, yc


def _as_batch_vec(v: Any, ref: torch.Tensor) -> torch.Tensor:
    t = v if isinstance(v, torch.Tensor) else torch.as_tensor(
        v, dtype=ref.dtype, device=ref.device)
    t = t.to(dtype=ref.dtype, device=ref.device).reshape(-1)
    if t.numel() == 1 and ref.numel() > 1:
        t = t.expand(ref.numel())
    return t


def _is_scalar_input(*vs) -> bool:
    return all(isinstance(v, (int, float)) for v in vs)


def _finalize_np(tokens: torch.Tensor, flags: torch.Tensor, scalar: bool = False):
    """Numpy return path: always batched ``(B, L, 3)`` + ``(B, L)`` (PRD §2.3)."""
    return tokens.numpy(), flags.numpy()


class PseudoSequenceGenerator(abc.ABC):
    """Base class: owns ``L``, ``dt``, integrator, domain and OOD policy."""

    #: index of the query token in the returned sequence
    query_index: int = 0
    variant: str = "base"

    def __init__(
        self,
        seq_len: int = 5,
        dt: float = 1.0,
        x_bounds: tuple[float, float] | None = None,
        y_bounds: tuple[float, float] | None = None,
        integrator: str = "rk2",
        substeps: int = 1,
        ood_policy: OODPolicy = "clamp",
        L: int | None = None,  # alias for seq_len (older call sites)
        bounds: tuple[float, float, float, float] | None = None,  # (xmin,xmax,ymin,ymax)
    ) -> None:
        if L is not None:
            seq_len = L
        if bounds is not None:
            x_bounds = (float(bounds[0]), float(bounds[1]))
            y_bounds = (float(bounds[2]), float(bounds[3]))
        if int(seq_len) < 1:
            raise ValueError("seq_len L must be >= 1")
        if float(dt) <= 0:
            raise ValueError("dt must be positive")
        if integrator not in ("euler", "rk2", "rk4"):
            raise ValueError(f"integrator must be euler/rk2/rk4, got '{integrator}'")
        if int(substeps) < 1:
            raise ValueError("substeps must be >= 1")
        if ood_policy not in ("clamp", "reflect"):
            raise ValueError(f"ood_policy must be clamp/reflect, got '{ood_policy}'")
        self.seq_len = int(seq_len)
        self.dt = float(dt)
        self.L = self.seq_len  # alias
        self.x_bounds = None if x_bounds is None else (float(x_bounds[0]), float(x_bounds[1]))
        self.y_bounds = None if y_bounds is None else (float(y_bounds[0]), float(y_bounds[1]))
        self.integrator = integrator
        self.substeps = int(substeps)
        self.ood_policy: OODPolicy = ood_policy

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def generate(self, x, y, t, wind_field=None, **kwargs):
        """Return ``(tokens (B, L, 3), flags (B, L) bool)``."""

    def __call__(self, x, y, t, wind_field=None, **kwargs):
        return self.generate(x, y, t, wind_field, **kwargs)

    # ------------------------------------------------------------------
    # shared torch core helpers
    # ------------------------------------------------------------------
    def _resolve(self, kwargs: dict, wind_field: Any, wind: Any):
        L = kwargs.pop("L", kwargs.pop("seq_len", self.seq_len))
        dt = kwargs.pop("dt", self.dt)
        bounds = kwargs.pop("bounds", None)
        wf = wind_field if wind_field is not None else wind
        xb = self.x_bounds
        yb = self.y_bounds
        if bounds is not None:
            xb = (float(bounds[0]), float(bounds[1]))
            yb = (float(bounds[2]), float(bounds[3]))
        return int(L), float(dt), wf, xb, yb, kwargs

    def _prep(self, x, y, t):
        dev = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        # Preserve floating dtype (float64 PINN runs stay float64); plain
        # python scalars default to float32.
        dt = torch.float32
        for v in (x, y, t):
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                dt = v.dtype
                dev = v.device
                break

        def _c(v):
            if isinstance(v, torch.Tensor):
                return v.to(dtype=dt).reshape(-1)
            return torch.as_tensor(v, dtype=dt, device=dev).reshape(-1)

        bx, by, bt = _c(x), _c(y), _c(t)
        n = max(bx.numel(), by.numel(), bt.numel())
        if bx.numel() == 1:
            bx = bx.expand(n)
        if by.numel() == 1:
            by = by.expand(n)
        if bt.numel() == 1:
            bt = bt.expand(n)
        return bx, by, bt

    def _maybe_numpy(self, tokens, flags, want_torch: bool):
        # NOTE: retained for backward-compat only; new code uses
        # _finalize_np. Correct stacking is (B, L, 3) -> numpy as-is.
        if want_torch or _np is None:
            return tokens, flags
        return tokens.cpu().numpy(), flags.cpu().numpy()

    def _pack(self, xs, ys, ts, hits, want_torch: bool, device):
        """Compat packer for scalar-list paths (older call sites)."""
        if want_torch or _np is None:
            import torch as _t

            tok = _t.stack(
                [_t.as_tensor(xs, dtype=_t.float32), _t.as_tensor(ys, dtype=_t.float32),
                 _t.as_tensor(ts, dtype=_t.float32)], dim=1).unsqueeze(0)
            fl = _t.as_tensor(hits, dtype=_t.bool).unsqueeze(0)
            return tok, fl
        tokens = _np.stack(
            [_np.asarray(xs, float), _np.asarray(ys, float), _np.asarray(ts, float)], axis=1)
        return tokens, _np.asarray(hits, dtype=bool)

    def _apply_domain(self, px, py):
        """Apply clamp/reflect policy; return ``(px, py, hit)`` bool ``(B,)``."""
        hit = torch.zeros_like(px, dtype=torch.bool)
        if self.x_bounds is not None:
            lo, hi = self.x_bounds
            oob = (px < lo) | (px > hi)
            hit = hit | oob
            if self.ood_policy == "clamp":
                px = px.clamp(lo, hi)
            else:
                px = torch.where(px < lo, 2 * lo - px, px)
                px = torch.where(px > hi, 2 * hi - px, px)
                px = px.clamp(lo, hi)
        if self.y_bounds is not None:
            lo, hi = self.y_bounds
            oob = (py < lo) | (py > hi)
            hit = hit | oob
            if self.ood_policy == "clamp":
                py = py.clamp(lo, hi)
            else:
                py = torch.where(py < lo, 2 * lo - py, py)
                py = torch.where(py > hi, 2 * hi - py, py)
                py = py.clamp(lo, hi)
        return px, py, hit

    def _sample_wind(self, wf: Any, px, py, pt):
        if wf is None:
            return torch.zeros_like(px), torch.zeros_like(py)
        if hasattr(wf, "sample"):
            u, v = wf.sample(px, py, pt)
            return _as_batch_vec(u, px), _as_batch_vec(v, py)
        if callable(wf):
            us, vs = [], []
            for xi, yi, ti in zip(px.tolist(), py.tolist(), pt.tolist()):
                uu, vv = wf(xi, yi, ti)
                us.append(float(uu))
                vs.append(float(vv))
            return (torch.tensor(us, dtype=px.dtype, device=px.device),
                    torch.tensor(vs, dtype=py.dtype, device=py.device))
        u, v = wf  # constant (u, v) pair
        return (torch.full_like(px, float(u)), torch.full_like(py, float(v)))

    def _advect_step(self, px, py, pt, h: float, wf: Any):
        """Single backward step of size ``h`` (``h`` positive; time decreases)."""
        u1, v1 = self._sample_wind(wf, px, py, pt)
        if self.integrator == "euler":
            return px - h * u1, py - h * v1
        if self.integrator == "rk2":
            mx, my, mt = px - 0.5 * h * u1, py - 0.5 * h * v1, pt - 0.5 * h
            u2, v2 = self._sample_wind(wf, mx, my, mt)
            return px - h * u2, py - h * v2
        k1x, k1y = u1, v1
        u2, v2 = self._sample_wind(wf, px - 0.5 * h * k1x, py - 0.5 * h * k1y, pt - 0.5 * h)
        u3, v3 = self._sample_wind(wf, px - 0.5 * h * u2, py - 0.5 * h * v2, pt - 0.5 * h)
        u4, v4 = self._sample_wind(wf, px - h * u3, py - h * v3, pt - h)
        return (px - (h / 6.0) * (k1x + 2 * u2 + 2 * u3 + u4),
                py - (h / 6.0) * (k1y + 2 * v2 + 2 * v3 + v4))

    def _advect(self, px, py, pt, h, wf):
        # NOTE: no domain clamping here — the caller clamps once per token
        # and latches the boundary-hit flag. Clamping inside the substep
        # loop would erase the out-of-domain signal before flags are read.
        # Wind lookup outside the rectangle is the wind field's own
        # responsibility (bilinear ext/clamp in data/wind_field.py).
        hs = h / self.substeps
        for _ in range(self.substeps):
            px, py = self._advect_step(px, py, pt, hs, wf)
            pt = pt - hs
        return px, py, pt


# Compat alias for older subclass sites.
_Base = PseudoSequenceGenerator


# ======================================================================
class UniformForwardGenerator(PseudoSequenceGenerator):
    """PINNsformer original: ``token k = (x, y, t + k·dt)``."""

    variant = "uniform_forward"

    def generate(self, x, y, t, wind_field=None, wind=None, **kwargs):
        L, dt, _, xb, yb, _ = self._resolve(dict(kwargs), wind_field, wind)
        want_torch = isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor) or isinstance(t, torch.Tensor)
        scalar = _is_scalar_input(x, y, t)
        bx, by, bt = self._prep(x, y, t)
        B = bx.numel()
        dev = bx.device
        tokens = torch.empty(B, L, 3, dtype=bx.dtype, device=dev)
        flags = torch.zeros(B, L, dtype=torch.bool, device=dev)
        for k in range(L):
            tokens[:, k, 0], tokens[:, k, 1], tokens[:, k, 2] = bx, by, bt + k * dt
        # flag queries outside the domain (tokens share x, y)
        svx, svy = self.x_bounds, self.y_bounds
        self.x_bounds, self.y_bounds = xb, yb
        _, _, hit = self._apply_domain(bx.clone(), by.clone())
        self.x_bounds, self.y_bounds = svx, svy
        flags[:] = hit.unsqueeze(1).expand(B, L)
        if want_torch:
            return tokens, flags
        if _np is not None:
            return _finalize_np(tokens, flags, scalar)
        return tokens, flags


class UniformBackwardGenerator(PseudoSequenceGenerator):
    """Time-reversed uniform offsets: ``token k = (x, y, t - k·dt)``."""

    variant = "uniform_backward"

    def generate(self, x, y, t, wind_field=None, wind=None, **kwargs):
        L, dt, _, xb, yb, _ = self._resolve(dict(kwargs), wind_field, wind)
        want_torch = isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor) or isinstance(t, torch.Tensor)
        scalar = _is_scalar_input(x, y, t)
        bx, by, bt = self._prep(x, y, t)
        B = bx.numel()
        dev = bx.device
        tokens = torch.empty(B, L, 3, dtype=bx.dtype, device=dev)
        flags = torch.zeros(B, L, dtype=torch.bool, device=dev)
        for k in range(L):
            tokens[:, k, 0], tokens[:, k, 1], tokens[:, k, 2] = bx, by, bt - k * dt
        svx, svy = self.x_bounds, self.y_bounds
        self.x_bounds, self.y_bounds = xb, yb
        _, _, hit = self._apply_domain(bx.clone(), by.clone())
        self.x_bounds, self.y_bounds = svx, svy
        flags[:] = hit.unsqueeze(1).expand(B, L)
        if want_torch:
            return tokens.to(bx.device), flags.to(bx.device)
        if _np is not None:
            return _finalize_np(tokens, flags, scalar)
        return tokens, flags


class AdvectionBackwardGenerator(PseudoSequenceGenerator):
    """Ours: tokens on backward wind characteristics (RK2/RK4, bilinear wind).

    A token whose characteristic had to be clamped/reflected onto the
    boundary is flagged — those flagged tokens carry ``C_b`` information
    (PRD §2.3). Once a trajectory hits, that and all later tokens stay flagged.
    """

    variant = "advection_backward"

    def generate(self, x, y, t, wind_field=None, wind=None, **kwargs):
        L, dt, wf, xb, yb, rest = self._resolve(dict(kwargs), wind_field, wind)
        sub = int(rest.pop("substeps", self.substeps))
        want_torch = isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor) or isinstance(t, torch.Tensor)
        scalar = _is_scalar_input(x, y, t)
        if wf is None and L > 1:
            raise ValueError("AdvectionBackwardGenerator requires a wind field for L > 1")
        bx, by, bt = self._prep(x, y, t)
        svx, svy, svs = self.x_bounds, self.y_bounds, self.substeps
        self.x_bounds, self.y_bounds, self.substeps = xb, yb, sub
        try:
            B = bx.numel()
            dev = bx.device
            tokens = torch.empty(B, L, 3, dtype=bx.dtype, device=dev)
            flags = torch.zeros(B, L, dtype=torch.bool, device=dev)
            px, py, pt = bx.clone(), by.clone(), bt.clone()
            tokens[:, 0, 0], tokens[:, 0, 1], tokens[:, 0, 2] = px, py, pt
            _, _, qhit = self._apply_domain(px.clone(), py.clone())
            flags[:, 0] = qhit
            latched = qhit.clone()
            for k in range(1, L):
                px, py, pt = self._advect(px, py, pt, dt, wf)
                cpx, cpy, hit = self._apply_domain(px.clone(), py.clone())
                px, py = cpx, cpy
                latched = latched | hit
                tokens[:, k, 0], tokens[:, k, 1], tokens[:, k, 2] = px, py, pt
                flags[:, k] = latched
        finally:
            self.x_bounds, self.y_bounds, self.substeps = svx, svy, svs
        if want_torch:
            return tokens, flags
        if _np is not None:
            return _finalize_np(tokens, flags, scalar)
        return tokens, flags


class AdvectionJitterGenerator(AdvectionBackwardGenerator):
    """Ours + diffusive jitter: per-step ``N(0, sqrt(2·K·dt))`` random walk.

    Marginal std at token ``k`` is ``sqrt(k)·sqrt(2·K·dt)`` (random walk),
    matching diffusive spread ``sqrt(2·K·k·dt)``.
    """

    variant = "advection_jitter"

    def __init__(self, *args: Any, K: float = 100.0, seed: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if float(K) < 0:
            raise ValueError("K must be non-negative")
        self.K = float(K)
        self.seed = None if seed is None else int(seed)
        self._gen: torch.Generator | None = None
        if self.seed is not None:
            self._gen = torch.Generator().manual_seed(self.seed)

    def _randn(self, shape, device) -> torch.Tensor:
        if self._gen is None:
            return torch.randn(shape, device=device, dtype=torch.float32)
        return torch.randn(shape, generator=self._gen, device=device, dtype=torch.float32)

    def generate(self, x, y, t, wind_field=None, wind=None, **kwargs):
        L, dt, wf, xb, yb, rest = self._resolve(dict(kwargs), wind_field, wind)
        K = float(rest.pop("K", self.K))
        seed = rest.pop("seed", None)
        sub = int(rest.pop("substeps", self.substeps))
        want_torch = isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor) or isinstance(t, torch.Tensor)
        scalar = _is_scalar_input(x, y, t)
        if wf is None and L > 1:
            raise ValueError("AdvectionJitterGenerator requires a wind field for L > 1")
        gen = self._gen
        if seed is not None:
            gen = torch.Generator().manual_seed(int(seed))
        bx, by, bt = self._prep(x, y, t)
        svx, svy, svs = self.x_bounds, self.y_bounds, self.substeps
        self.x_bounds, self.y_bounds, self.substeps = xb, yb, sub
        try:
            B = bx.numel()
            dev = bx.device
            tokens = torch.empty(B, L, 3, dtype=bx.dtype, device=dev)
            flags = torch.zeros(B, L, dtype=torch.bool, device=dev)
            px, py, pt = bx.clone(), by.clone(), bt.clone()
            tokens[:, 0, 0], tokens[:, 0, 1], tokens[:, 0, 2] = px, py, pt
            _, _, qhit = self._apply_domain(px.clone(), py.clone())
            flags[:, 0] = qhit
            latched = qhit.clone()
            step_std = math.sqrt(2.0 * K * dt) if K > 0 else 0.0
            for k in range(1, L):
                px, py, pt = self._advect(px, py, pt, dt, wf)
                if step_std > 0:
                    if gen is not None:
                        inc = torch.randn(B, 2, generator=gen, dtype=px.dtype, device=dev) * step_std
                    else:
                        inc = torch.randn(B, 2, dtype=px.dtype, device=dev) * step_std
                    px = px + inc[:, 0]
                    py = py + inc[:, 1]
                cpx, cpy, hit = self._apply_domain(px.clone(), py.clone())
                px, py = cpx, cpy
                latched = latched | hit
                tokens[:, k, 0], tokens[:, k, 1], tokens[:, k, 2] = px, py, pt
                flags[:, k] = latched
        finally:
            self.x_bounds, self.y_bounds, self.substeps = svx, svy, svs
        if want_torch:
            return tokens, flags
        if _np is not None:
            return _finalize_np(tokens, flags, scalar)
        return tokens, flags


VARIANTS = (
    "uniform_forward",
    "uniform_backward",
    "advection_backward",
    "advection_jitter",
)

_GENERATORS = {
    "uniform_forward": UniformForwardGenerator,
    "uniform_backward": UniformBackwardGenerator,
    "advection_backward": AdvectionBackwardGenerator,
    "advection_jitter": AdvectionJitterGenerator,
}
_GENERATORS.update({
    "uniform-forward": UniformForwardGenerator,
    "uniform-backward": UniformBackwardGenerator,
    "advection-backward": AdvectionBackwardGenerator,
    "advection-jitter": AdvectionJitterGenerator,
    "advection-backward-jitter": AdvectionJitterGenerator,
})


def _norm_key(name: str) -> str:
    return "_".join(p for p in str(name).strip().lower().replace("-", "_").replace(" ", "_").split("_") if p)


def create_generator(name: str, **kwargs: Any) -> PseudoSequenceGenerator:
    """Factory: ``create_generator("advection_backward", seq_len=5, dt=3600.0)``.

    Names are normalized (case/separator-insensitive), so
    ``"advection-backward-jitter"``, ``"advection_backward_jitter"`` and
    ``"advection_backward"``-family aliases all resolve. YAML aliases
    (``L``, ``dt_hours``, ``out_of_domain``) are translated here.
    """
    key = _norm_key(name)
    norm_map = {_norm_key(k): v for k, v in _GENERATORS.items()}
    if key not in norm_map:
        raise KeyError(
            f"unknown pseudo-sequence generator '{name}'; expected one of {sorted(VARIANTS)}"
        )
    kwargs = _translate_yaml_aliases(dict(kwargs))
    return norm_map[key](**kwargs)


def _translate_yaml_aliases(kwargs: dict) -> dict:
    """Translate ``configs/pseudoseq/*.yaml`` keys to constructor kwargs.

    ``L`` → ``seq_len``; ``dt_hours`` → ``dt`` (hours → seconds);
    ``out_of_domain`` → ``ood_policy``. Unknown keys pass through so
    subclass kwargs (``K``, ``seed``) keep working.
    """
    if "L" in kwargs and "seq_len" not in kwargs:
        kwargs["seq_len"] = kwargs.pop("L")
    elif "L" in kwargs:
        kwargs.pop("L")
    if "dt_hours" in kwargs and "dt" not in kwargs:
        kwargs["dt"] = float(kwargs.pop("dt_hours")) * 3600.0
    elif "dt_hours" in kwargs:
        kwargs.pop("dt_hours")
    if "out_of_domain" in kwargs and "ood_policy" not in kwargs:
        kwargs["ood_policy"] = kwargs.pop("out_of_domain")
    elif "out_of_domain" in kwargs:
        kwargs.pop("out_of_domain")
    kwargs.pop("variant", None)  # config label, not a constructor arg
    kwargs.pop("jitter", None)  # descriptive rule string, not a constructor arg
    if kwargs.get("integrator") is None:
        # Uniform-offset yamls set `integrator: null` (no integration used).
        kwargs.pop("integrator", None)
    return kwargs


def get_generator(variant: str, **kwargs: Any) -> PseudoSequenceGenerator:
    """Alias of :func:`create_generator` (older call sites)."""
    return create_generator(variant, **kwargs)
