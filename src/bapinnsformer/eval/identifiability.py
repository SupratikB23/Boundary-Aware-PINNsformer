"""E1 identifiability analysis (PRD §4.3 `eval/identifiability.py`, §6 E1).

Recovery error as a function of station count, station configuration,
wind directional entropy and noise. Fits a small log-linear response
surface with pure numpy (no sklearn) and reports the operating regime
/ threshold crossing (the "identifiability boundary").
"""

from __future__ import annotations

import math

CONFIG_ARCHETYPES = ("clustered", "uniform", "perimeter_biased")


def directional_entropy(directions_rad, n_bins: int = 8) -> float:
    """Shannon entropy (nats) of wind-direction histogram (RQ2 quantity)."""
    import numpy as np

    d = np.asarray(list(directions_rad), dtype=float).ravel() % (2 * math.pi)
    if d.size == 0:
        return float("nan")
    hist, _ = np.histogram(d, bins=int(n_bins), range=(0.0, 2 * math.pi))
    p = hist.astype(float) / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def cell_key(n_stations: int, config: str, entropy_bin: str, noise: float) -> str:
    if config not in CONFIG_ARCHETYPES:
        raise ValueError(f"unknown configuration archetype {config!r}")
    return f"n{n_stations}_{config}_{entropy_bin}_s{noise:g}"


def fit_surface(cells: list[dict]) -> dict:
    """Fit ``log err = a + b·log(n) + c·entropy + d·noise + config``.

    Each cell: ``{n_stations, config, entropy, noise, err}`` with
    ``err`` = max(rel_l2_Cb, rel_l2_S). Returns coefficients, R² and
    per-term interpretation. Pure numpy least squares.
    """
    import numpy as np

    if len(cells) < 6:
        raise ValueError("need ≥6 sweep cells to fit the surface")
    for cell in cells:
        for k in ("n_stations", "config", "entropy", "noise", "err"):
            if k not in cell:
                raise ValueError(f"cell missing {k!r}: {cell}")
    y = np.array([math.log(max(float(c["err"]), 1e-12)) for c in cells])
    n = np.array([math.log(max(int(c["n_stations"]), 1)) for c in cells])
    e = np.array([float(c["entropy"]) for c in cells])
    s = np.array([float(c["noise"]) for c in cells])
    cfgs = [str(c["config"]) for c in cells]
    # One-hot config with 'uniform' as reference level.
    X = np.column_stack(
        [
            np.ones(len(cells)),
            n,
            e,
            s,
            np.array([1.0 if c == "clustered" else 0.0 for c in cfgs]),
            np.array([1.0 if c == "perimeter_biased" else 0.0 for c in cfgs]),
        ]
    )
    coef, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    names = ("intercept", "log_n", "entropy", "noise", "cfg_clustered", "cfg_perimeter")
    return {
        "coef": {k: float(v) for k, v in zip(names, coef)},
        "r2": float(r2),
        "n_cells": len(cells),
        "model": "log err = a + b·log(n) + c·entropy + d·noise + cfg",
        "interpretation": {
            "more_stations_help": bool(coef[1] < 0),
            "entropy_helps": bool(coef[2] < 0),
            "noise_hurts": bool(coef[3] > 0),
        },
    }


def predict_surface(surface: dict, n_stations: int, config: str, entropy: float, noise: float) -> float:
    """Predict error (natural scale) from a fitted surface."""
    c = surface["coef"]
    val = (
        c["intercept"]
        + c["log_n"] * math.log(max(int(n_stations), 1))
        + c["entropy"] * float(entropy)
        + c["noise"] * float(noise)
    )
    if config == "clustered":
        val += c["cfg_clustered"]
    elif config == "perimeter_biased":
        val += c["cfg_perimeter"]
    elif config != "uniform":
        raise ValueError(f"unknown configuration archetype {config!r}")
    return float(math.exp(val))


def stations_for_threshold(
    surface: dict, config: str, entropy: float, noise: float, threshold: float = 0.25
) -> float:
    """Invert the surface: stations needed for predicted err ≤ threshold.

    Returns ``inf`` when the slope is degenerate (more stations do not
    help within the fitted model — itself a reportable finding).
    """
    import math as _math

    c = surface["coef"]
    if c["log_n"] >= 0:
        return float("inf")
    cfg_term = 0.0
    if config == "clustered":
        cfg_term = c["cfg_clustered"]
    elif config == "perimeter_biased":
        cfg_term = c["cfg_perimeter"]
    rhs = _math.log(threshold) - c["intercept"] - c["entropy"] * entropy - c["noise"] * noise - cfg_term
    return float(_math.exp(rhs / c["log_n"]))


__all__ = [
    "CONFIG_ARCHETYPES",
    "directional_entropy",
    "cell_key",
    "fit_surface",
    "predict_surface",
    "stations_for_threshold",
]
