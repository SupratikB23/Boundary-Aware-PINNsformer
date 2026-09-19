"""Collocation sampling: interior / boundary / initial + curriculum + RAR.

PRD §4.3 `physics/collocation.py`: sampling of interior, boundary and
initial collocation points; curriculum-aware time windows (expanding
``[t0, t0+Δ]`` schedule); residual-weighted (RAR-style) resampling.
Numpy-only; torch conversion happens in the trainer. CPU-runnable.
"""

from __future__ import annotations

import numpy as np


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def sample_interior(n: int, *args: object, seed: int | None = 0, **kwargs: object) -> dict[str, np.ndarray]:
    """Uniform interior ``(x, y, t)`` samples — dual signature.

    - Task API: ``sample_interior(n, x_bounds, y_bounds, t_bounds)``.
    - Legacy API: ``sample_interior(n, bounds=(xmin,xmax,ymin,ymax),
      t_window=(t0,t1))``.
    """
    r = _rng(seed)
    n = int(n)
    if n < 1:
        raise ValueError("sample_interior: n must be >= 1")
    if len(args) == 3:  # task form
        (x0, x1), (y0, y1), (t0, t1) = args  # type: ignore[misc]
    elif len(args) == 2:  # legacy form
        bounds, t_window = args  # type: ignore[misc]
        x0, x1, y0, y1 = (float(v) for v in bounds)  # type: ignore[union-attr]
        t0, t1 = (float(v) for v in t_window)  # type: ignore[union-attr]
    elif "x_bounds" in kwargs:
        (x0, x1), (y0, y1), (t0, t1) = kwargs["x_bounds"], kwargs["y_bounds"], kwargs["t_bounds"]  # type: ignore[misc]
    elif "bounds" in kwargs:
        bounds, t_window = kwargs["bounds"], kwargs["t_window"]  # type: ignore[misc]
        x0, x1, y0, y1 = (float(v) for v in bounds)  # type: ignore[union-attr]
        t0, t1 = (float(v) for v in t_window)  # type: ignore[union-attr]
    else:
        raise TypeError("sample_interior: use (n, x_bounds, y_bounds, t_bounds) or (n, bounds, t_window)")
    if not (x0 < x1 and y0 < y1 and t0 <= t1):
        raise ValueError("sample_interior: invalid bounds")
    return {
        "x": r.uniform(x0, x1, n),
        "y": r.uniform(y0, y1, n),
        "t": r.uniform(t0, t1, n) if t1 > t0 else np.full(n, t0),
    }


def sample_boundary(n: int, *args: object, seed: int | None = 0, **kwargs: object) -> dict[str, np.ndarray]:
    """Boundary samples with arc-length ``s`` (CCW from SW corner).

    - Task API: ``sample_boundary(n, s_perimeter=P, t_bounds=(t0,t1))``
      → ``{s, t}`` uniform.
    - Legacy API: ``sample_boundary(n_per_edge, bounds, t_window)`` →
      ``{x, y, t, s, perimeter}`` with ``4*n`` points mapped to edges.
    """
    r = _rng(seed)
    n = int(n)
    if n < 1:
        raise ValueError("sample_boundary: n must be >= 1")
    # Legacy: (bounds, t_window) where bounds has 4 entries.
    if len(args) == 2 and not np.isscalar(args[0]):
        try:
            bounds = tuple(float(v) for v in args[0])  # type: ignore[union-attr]
            t_window = tuple(float(v) for v in args[1])  # type: ignore[union-attr]
            if len(bounds) == 4:
                xmin, xmax, ymin, ymax = bounds
                t0, t1 = t_window
                W, H = xmax - xmin, ymax - ymin
                P = 2 * (W + H)
                s = r.uniform(0, P, 4 * n)
                x = np.empty_like(s)
                y = np.empty_like(s)
                m0 = s < W
                x[m0], y[m0] = xmin + s[m0], ymin
                m1 = (s >= W) & (s < W + H)
                x[m1], y[m1] = xmax, ymin + (s[m1] - W)
                m2 = (s >= W + H) & (s < 2 * W + H)
                x[m2], y[m2] = xmax - (s[m2] - W - H), ymax
                m3 = s >= 2 * W + H
                x[m3], y[m3] = xmin, ymax - (s[m3] - 2 * W - H)
                return {"x": x, "y": y, "t": r.uniform(t0, t1, len(s)), "s": s, "perimeter": P}
        except TypeError:
            pass
    # Task form: (s_perimeter, t_bounds).
    if len(args) == 2:
        P = float(args[0])  # type: ignore[arg-type]
        t_bounds = tuple(float(v) for v in args[1])  # type: ignore[union-attr]
    elif "s_perimeter" in kwargs:
        P = float(kwargs["s_perimeter"])  # type: ignore[arg-type]
        t_bounds = tuple(float(v) for v in kwargs.get("t_bounds", (0.0, 1.0)))  # type: ignore[union-attr]
    elif "bounds" in kwargs:
        return sample_boundary(n, kwargs["bounds"], kwargs["t_window"], seed=seed)
    else:
        raise TypeError("sample_boundary: use (n, s_perimeter, t_bounds) or (n_per_edge, bounds, t_window)")
    if P <= 0:
        raise ValueError("sample_boundary: s_perimeter (P) must be positive")
    t0, t1 = t_bounds
    return {"s": r.uniform(0.0, P, n), "t": r.uniform(t0, t1, n) if t1 > t0 else np.full(n, t0)}


def sample_initial(
    n: int,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    t0: float,
    seed: int | None = 0,
) -> dict[str, np.ndarray]:
    """Initial-condition samples: uniform ``(x, y)`` at fixed ``t0``."""
    if n < 1:
        raise ValueError("sample_initial: n must be >= 1")
    r = _rng(seed)
    return {
        "x": r.uniform(x_bounds[0], x_bounds[1], n),
        "y": r.uniform(y_bounds[0], y_bounds[1], n),
        "t": np.full(n, float(t0)),
    }


def curriculum_window(
    t0: float,
    t1: float,
    stage: int,
    n_stages: int,
    min_frac: float = 0.2,
) -> tuple[float, float]:
    """Expanding time window ``[t0, t0 + Δ(stage)]`` for the curriculum.

    Grows linearly from ``min_frac`` of ``[t0, t1]`` (stage 0) to the
    full range (last stage). Used by `train/curriculum.py` to gate
    collocation sampling and the data loss.
    """
    if n_stages < 1:
        raise ValueError("curriculum_window: n_stages must be >= 1")
    stage = int(np.clip(stage, 0, n_stages - 1))
    frac = float(min_frac) + (1.0 - float(min_frac)) * (stage / max(n_stages - 1, 1))
    return float(t0), float(t0 + frac * (float(t1) - float(t0)))


def residual_weighted_resample(
    points: dict[str, np.ndarray],
    residuals: np.ndarray,
    n_add: int,
    seed: int | None = 0,
    jitter: float = 0.0,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """RAR-style resampling: replicate high-residual points (+ jitter).

    Args:
        points: dict of equal-length coordinate arrays (e.g. x/y/t).
        residuals: non-negative magnitudes aligned to points.
        n_add: number of points to append (sampled ∝ residual + eps).
        jitter: relative uniform jitter amplitude (fraction of per-key
            range; 0 = exact replicates).
        bounds: optional ``{key: (lo, hi)}`` to clip jittered points;
            inferred from data when omitted (no hardcoded scaling).
    """
    keys = list(points.keys())
    n = len(points[keys[0]])
    if any(len(points[k]) != n for k in keys):
        raise ValueError("residual_weighted_resample: ragged point dict")
    r = np.abs(np.asarray(residuals, dtype=float).ravel())
    if r.size != n:
        raise ValueError("residual_weighted_resample: residuals length mismatch")
    if n_add < 1:
        return {k: np.asarray(points[k]).copy() for k in keys}
    w = r + 1e-12
    p = w / w.sum()
    rng = _rng(seed)
    idx = rng.choice(n, size=int(n_add), replace=True, p=p)
    out = {k: np.concatenate([np.asarray(points[k]).ravel(), np.asarray(points[k]).ravel()[idx]]) for k in keys}
    if jitter and jitter > 0:
        for k in keys:
            v = out[k]
            lo, hi = bounds[k] if bounds and k in bounds else (float(v.min()), float(v.max()))
            amp = float(jitter) * (hi - lo)
            if amp > 0:
                out[k][n:] = np.clip(out[k][n:] + rng.uniform(-amp, amp, size=int(n_add)), lo, hi)
    return out


__all__ = [
    "sample_interior",
    "sample_boundary",
    "sample_initial",
    "curriculum_window",
    "residual_weighted_resample",
]
