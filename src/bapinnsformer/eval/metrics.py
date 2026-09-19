"""Recovery and validation metrics (PRD §4.3 `eval/metrics.py`, §7.2).

Conventions:
- E1 (synthetic): relative L2 on recovered ``C_b`` and ``S`` plus the
  identifiability margin.
- E2 (real data): RMSE / MAE / R² against withheld perimeter (and
  held-out interior) stations.
- E4: Spearman correlation utilities (rank-based, robust to scale).

All functions accept array-likes (numpy arrays, lists, torch tensors)
and ignore NaN pairs. Torch is imported lazily; numpy suffices.
"""

from __future__ import annotations

import math

_EPS = 1e-12


def to_numpy(a):
    """Convert torch tensor / list / scalar to a numpy ``float64`` array."""
    try:
        import torch

        if isinstance(a, torch.Tensor):
            return a.detach().cpu().to(dtype=torch.float64).numpy()
    except ImportError:
        pass
    try:
        import numpy as np

        return np.asarray(a, dtype=np.float64)
    except ImportError:  # minimal fallback
        if isinstance(a, (list, tuple)):
            import array as _array

            return _array.array("d", a)
        raise


def _valid_pair(est, true):
    import numpy as np

    est = np.asarray(to_numpy(est)).ravel()
    true = np.asarray(to_numpy(true)).ravel()
    if est.shape != true.shape:
        raise ValueError(f"shape mismatch: {est.shape} vs {true.shape}")
    mask = np.isfinite(est) & np.isfinite(true)
    if int(mask.sum()) == 0:
        raise ValueError("no finite (est, true) pairs")
    return est[mask], true[mask]


def rel_l2(est, true) -> float:
    """Relative L2 error ``||est − true||₂ / ||true||₂`` (E1 primary)."""
    import numpy as np

    est_v, true_v = _valid_pair(est, true)
    denom = float(np.linalg.norm(true_v))
    if denom < _EPS:
        return float("inf") if float(np.linalg.norm(est_v)) >= _EPS else 0.0
    return float(np.linalg.norm(est_v - true_v) / denom)


def rmse(est, true) -> float:
    """Root-mean-square error (E2 primary)."""
    import numpy as np

    est_v, true_v = _valid_pair(est, true)
    return float(np.sqrt(np.mean((est_v - true_v) ** 2)))


def mae(est, true) -> float:
    """Mean absolute error (E2)."""
    import numpy as np

    est_v, true_v = _valid_pair(est, true)
    return float(np.mean(np.abs(est_v - true_v)))


def r2(est, true) -> float:
    """Coefficient of determination ``1 − SS_res/SS_tot`` (E2).

    Returns ``nan`` when the target has zero variance. May be negative
    for models worse than the mean predictor (reported honestly).
    """
    import numpy as np

    est_v, true_v = _valid_pair(est, true)
    ss_res = float(np.sum((true_v - est_v) ** 2))
    ss_tot = float(np.sum((true_v - np.mean(true_v)) ** 2))
    if ss_tot < _EPS:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _ranks(x):
    """Average ranks (1-based) with tie averaging; pure numpy."""
    import numpy as np

    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty_like(x, dtype=np.float64)
    n = x.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_with_p(x, y) -> tuple[float, float]:
    """Spearman rank correlation ``(rho, p_value)`` (E4).

    Uses ``scipy.stats.spearmanr`` when scipy is installed; otherwise a
    pure-numpy rank Pearson computation with a Student-t two-sided
    p-value (``nan`` p when it cannot be computed, e.g. n < 3 or
    constant ranks — rho is still returned).
    """
    import numpy as np

    xv, yv = _valid_pair(x, y)
    n = xv.size
    try:
        from scipy.stats import spearmanr  # type: ignore

        res = spearmanr(xv, yv)
        return float(res.statistic), float(res.pvalue)
    except ImportError:
        pass
    rx, ry = _ranks(xv), _ranks(yv)
    sx, sy = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((sx**2).sum() * (sy**2).sum()))
    if denom < _EPS:
        return float("nan"), float("nan")
    rho = float((sx * sy).sum() / denom)
    rho = max(-1.0, min(1.0, rho))
    if n < 3 or abs(rho) >= 1.0 - 1e-12:
        p = 0.0 if n >= 3 else float("nan")
        return rho, p
    t_stat = rho * math.sqrt((n - 2) / max(_EPS, 1.0 - rho * rho))
    # Two-sided p via normal approximation to the t distribution
    # (exact enough for screening; scipy path is exact when available).
    p = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return rho, float(max(0.0, min(1.0, p)))


def identifiability_margin(rel_l2_cb: float, rel_l2_s: float, threshold: float = 0.25) -> float:
    """Identifiability margin for one E1 sweep cell (PRD §7.2).

    ``margin = threshold − max(rel_l2(C_b), rel_l2(S))``. Positive
    means both unknowns recovered within tolerance (identifiable
    regime); negative means at least one is not. The worst of the two
    governs because the ``C_b`` vs ``S`` decomposition is only as good
    as its weaker half.
    """
    return float(threshold - max(float(rel_l2_cb), float(rel_l2_s)))


def regression_metrics(est, true) -> dict[str, float]:
    """Convenience bundle: ``{rel_l2, rmse, mae, r2, n}``."""
    import numpy as np

    est_v, true_v = _valid_pair(est, true)
    return {
        "rel_l2": rel_l2(est_v, true_v),
        "rmse": rmse(est_v, true_v),
        "mae": mae(est_v, true_v),
        "r2": r2(est_v, true_v),
        "n": int(np.size(est_v)),
    }


__all__ = [
    "to_numpy",
    "rel_l2",
    "rmse",
    "mae",
    "r2",
    "spearman_with_p",
    "identifiability_margin",
    "regression_metrics",
]
