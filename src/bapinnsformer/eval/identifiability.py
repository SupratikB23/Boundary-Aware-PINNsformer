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


# ======================================================================
# Linear-Gaussian identifiability theory (RUN_PLAN §1, contribution C2)
# ======================================================================
# With the wind, K and lambda fixed, every observation is linear in the
# unknown coefficients theta = [c_b; s; c_0] (physics/greens.py):
#     y = G theta + eps,  eps ~ N(0, sigma^2 I),  theta ~ N(mu, diag(p)).
# Everything below is exact for that model and is what "identifiable"
# means quantitatively in the paper.


def bayes_linear_inversion(G, y, prior_mean, prior_var, noise_var):
    """Closed-form Gaussian posterior for ``y = G θ + ε``.

    Returns ``(post_mean, post_cov)`` from the information form
    ``Σ = (GᵀG/σ² + P⁻¹)⁻¹``, ``m = Σ (Gᵀy/σ² + P⁻¹μ)`` (P diagonal).
    """
    import numpy as np

    G = np.asarray(G, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    mu = np.asarray(prior_mean, dtype=float).ravel()
    p = np.asarray(prior_var, dtype=float).ravel()
    if G.shape != (y.size, mu.size) or p.size != mu.size:
        raise ValueError(f"shape mismatch: G {G.shape}, y {y.shape}, mu {mu.shape}, p {p.shape}")
    if float(noise_var) <= 0 or np.any(p <= 0):
        raise ValueError("noise_var and prior_var must be positive")
    H = G.T @ G / float(noise_var) + np.diag(1.0 / p)
    L = np.linalg.cholesky(H)
    Linv = np.linalg.solve(L, np.eye(H.shape[0]))
    cov = Linv.T @ Linv
    mean = cov @ (G.T @ y / float(noise_var) + mu / p)
    return mean, 0.5 * (cov + cov.T)


def principal_angles(A, B, rtol: float = 1e-8):
    """Principal angles (radians, ascending) between ``range(A)`` and ``range(B)``.

    Orthonormal bases come from a rank-revealing SVD: directions with
    singular value below ``rtol·σ_max`` are dropped, because unobservable
    directions carry no information and must not count as "separable".
    """
    import numpy as np

    def _basis(M):
        M = np.asarray(M, dtype=float)
        if M.size == 0:
            return np.zeros((M.shape[0], 0))
        U, sv, _ = np.linalg.svd(M, full_matrices=False)
        if sv.size == 0 or sv[0] <= 0:
            return np.zeros((M.shape[0], 0))
        return U[:, sv > rtol * sv[0]]

    Qa, Qb = _basis(A), _basis(B)
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return np.array([])
    cosines = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.sort(np.arccos(np.clip(cosines, -1.0, 1.0)))


def whitened_blocks(sys, prior_var, noise_var):
    """``(Ã_b, Ã_rest)`` = ``σ⁻¹ A P^{1/2}`` for boundary vs source+IC blocks."""
    import numpy as np

    nb, ns, n0 = sys.sizes
    p = np.asarray(prior_var, dtype=float).ravel()
    sd = 1.0 / np.sqrt(float(noise_var))
    Ab = sys.A_b * np.sqrt(p[:nb])[None, :] * sd
    Ar = np.hstack([sys.A_s, sys.A_0]) * np.sqrt(p[nb:])[None, :] * sd
    return Ab, Ar


def confounding_coefficient(sys, post_cov):
    """Posterior correlation of boundary vs local receptor contributions.

    ``ρ = corr(h_bᵀθ, h_rᵀθ)`` under the posterior, where ``h_b`` / ``h_r``
    are the receptor operators of the boundary / (source + IC) blocks.
    ``ρ → −1``: the data constrain only the *sum* (confounded).
    ``ρ ≈ 0``: boundary and local contributions are separately identified.
    The posterior covariance does not depend on ``y`` (linear-Gaussian).
    """
    import numpy as np

    nb, ns, n0 = sys.sizes
    h_b = np.concatenate([sys.R_b, np.zeros(ns + n0)])
    h_r = np.concatenate([np.zeros(nb), sys.R_s, sys.R_0])
    S = np.asarray(post_cov, dtype=float)
    vb = float(h_b @ S @ h_b)
    vr = float(h_r @ S @ h_r)
    c = float(h_b @ S @ h_r)
    if vb <= 0 or vr <= 0:
        return float("nan")
    return c / float(np.sqrt(vb * vr))


def receptor_share(sys, theta):
    """Point share ``h_bᵀθ / h_allᵀθ`` (contributions clipped at 0)."""
    import numpy as np

    nb = sys.sizes[0]
    th = np.asarray(theta, dtype=float).ravel()
    cb = max(float(th[:nb] @ sys.R_b), 0.0)
    cr = max(float(th[nb:] @ np.concatenate([sys.R_s, sys.R_0])), 0.0)
    return cb / (cb + cr) if (cb + cr) > 0 else float("nan")


def share_posterior(sys, post_mean, post_cov, n_samples: int = 4000, seed: int = 0,
                    clip_nonneg: bool = True) -> dict:
    """Monte-Carlo posterior of the receptor transboundary share.

    Samples ``θ ~ N(m, Σ)``; negative contributions are clipped at 0 when
    ``clip_nonneg``. Returns ``{mean, median, q05, q95, sd, n}``.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    mean = np.asarray(post_mean, dtype=float)
    L = np.linalg.cholesky(np.asarray(post_cov, dtype=float) + 1e-12 * np.eye(mean.size))
    th = mean[None, :] + rng.standard_normal((n_samples, mean.size)) @ L.T
    nb = sys.sizes[0]
    cb = th[:, :nb] @ sys.R_b
    cr = th[:, nb:] @ np.concatenate([sys.R_s, sys.R_0])
    if clip_nonneg:
        cb, cr = np.maximum(cb, 0.0), np.maximum(cr, 0.0)
    tot = cb + cr
    sh = cb[tot > 0] / tot[tot > 0]
    if sh.size == 0:
        nan = float("nan")
        return {"mean": nan, "median": nan, "q05": nan, "q95": nan, "sd": nan, "n": 0}
    return {
        "mean": float(sh.mean()),
        "median": float(np.median(sh)),
        "q05": float(np.quantile(sh, 0.05)),
        "q95": float(np.quantile(sh, 0.95)),
        "sd": float(sh.std()),
        "n": int(sh.size),
    }


def variance_reduction(prior_var, post_cov):
    """Per-parameter ``1 − σ²_post/σ²_prior`` (0 = data uninformative, 1 = exact)."""
    import numpy as np

    p = np.asarray(prior_var, dtype=float).ravel()
    return 1.0 - np.clip(np.diag(np.asarray(post_cov)) / p, 0.0, 1.0)


def identifiability_report(sys, prior_mean, prior_var, noise_var, y=None,
                           theta_true=None, seed: int = 0) -> dict:
    """All E1 identifiability quantities for one Green's system.

    * ``min_angle_deg`` / ``median_angle_deg`` — principal angles between
      the whitened boundary range and the whitened local (source + IC)
      range. 0° = some boundary pattern is indistinguishable from a local one.
    * ``confounding_rho`` — see :func:`confounding_coefficient`.
    * ``boundary_observability`` — fraction of inflow-active boundary
      coefficients (inflow_frac ≥ 0.5) whose variance reduction ≥ 0.5.
    * ``share_num_var_reduction`` — how much the data shrink the variance
      of the boundary receptor contribution.

    With ``y`` (and optionally ``theta_true``) also returns the posterior
    estimate, the 90 % share interval, interval coverage and errors.
    Boundary error is computed on **inflow-active** coefficients only:
    outflow-boundary values never influence the interior and are not
    identifiable by construction (PRD §2.4).
    """
    import numpy as np

    G = sys.G
    nb, ns, n0 = sys.sizes
    yy = np.zeros(G.shape[0]) if y is None else np.asarray(y, dtype=float)
    m, S = bayes_linear_inversion(G, yy, prior_mean, prior_var, noise_var)
    Ab, Ar = whitened_blocks(sys, prior_var, noise_var)
    ang = principal_angles(Ab, Ar)
    vr = variance_reduction(prior_var, S)
    active = sys.b_inflow_frac >= 0.5
    h_b = np.concatenate([sys.R_b, np.zeros(ns + n0)])
    P = np.diag(np.asarray(prior_var, dtype=float))
    out = {
        "min_angle_deg": float(np.degrees(ang[0])) if ang.size else float("nan"),
        "median_angle_deg": float(np.degrees(np.median(ang))) if ang.size else float("nan"),
        "confounding_rho": confounding_coefficient(sys, S),
        "boundary_observability": float((vr[:nb][active] >= 0.5).mean()) if active.any() else 0.0,
        "share_num_var_reduction": float(1.0 - (h_b @ S @ h_b) / max(float(h_b @ P @ h_b), 1e-300)),
        "n_inflow_active": int(active.sum()),
        "n_params": int(G.shape[1]),
        "n_rows": int(G.shape[0]),
    }
    if y is not None:
        post = share_posterior(sys, m, S, seed=seed)
        out.update({f"share_post_{k}": v for k, v in post.items()})
        out["share_est"] = receptor_share(sys, m)
        if theta_true is not None:
            tt = np.asarray(theta_true, dtype=float).ravel()
            st = receptor_share(sys, tt)
            out["share_true"] = st
            out["share_abs_err"] = abs(out["share_est"] - st)
            out["share_covered_90"] = bool(post["q05"] <= st <= post["q95"])
            w = active.astype(float)
            denom = max(float(np.linalg.norm(tt[:nb] * w)), 1e-12)
            out["rel_l2_cb_inflow"] = float(np.linalg.norm((m[:nb] - tt[:nb]) * w) / denom)
            out["rel_l2_s"] = float(np.linalg.norm(m[nb:nb + ns] - tt[nb:nb + ns])
                                    / max(float(np.linalg.norm(tt[nb:nb + ns])), 1e-12))
    out["_post_mean"] = m
    out["_post_cov"] = S
    return out


__all__ = [
    "CONFIG_ARCHETYPES",
    "directional_entropy",
    "cell_key",
    "fit_surface",
    "predict_surface",
    "stations_for_threshold",
    "bayes_linear_inversion",
    "principal_angles",
    "whitened_blocks",
    "confounding_coefficient",
    "receptor_share",
    "share_posterior",
    "variance_reduction",
    "identifiability_report",
]
