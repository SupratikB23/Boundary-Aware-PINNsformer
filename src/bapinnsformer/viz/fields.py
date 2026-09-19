"""Recovered-field panels (PRD §4.3 `viz/fields.py`, paper Figs. 5, 7).

Snapshots of ``C`` / ``S`` and ground-truth-vs-recovered panels for
E1. Headless: save to path, no ``show()``.
"""

from __future__ import annotations


def field_snapshot(
    field_xy,
    bounds: tuple[float, float, float, float],
    save_path: str = "field.png",
    title: str = "Concentration field",
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
) -> str:
    """Render one 2-D field snapshot (scatter/tricontour) to file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .style import apply_style, ensure_parent

    apply_style()
    import numpy as np

    arr = np.asarray(field_xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("field_xy must be (n, 3) with columns (x, y, value)")
    xmin, xmax, ymin, ymax = (float(v) for v in bounds)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sc = ax.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2], cmap=cmap, s=8, vmin=vmin, vmax=vmax)
    fig.colorbar(sc, ax=ax, label="value")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    fig.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path)


def truth_vs_recovered(
    true_xy,
    est_xy,
    bounds: tuple[float, float, float, float],
    save_path: str = "truth_vs_recovered.png",
    title: str = "Ground truth vs recovered",
) -> str:
    """Side-by-side truth / recovered / absolute-error panels to file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .style import apply_style, ensure_parent

    apply_style()
    import numpy as np

    true = np.asarray(true_xy, dtype=float)
    est = np.asarray(est_xy, dtype=float)
    if true.shape != est.shape or true.ndim != 2 or true.shape[1] != 3:
        raise ValueError("true_xy and est_xy must both be (n, 3)")
    err = np.abs(est[:, 2] - true[:, 2])
    xmin, xmax, ymin, ymax = (float(v) for v in bounds)
    vmin = float(min(true[:, 2].min(), est[:, 2].min()))
    vmax = float(max(true[:, 2].max(), est[:, 2].max()))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True)
    for ax, vals, ttl, cmap in zip(
        axes,
        (true[:, 2], est[:, 2], err),
        ("ground truth", "recovered", "|error|"),
        ("viridis", "viridis", "magma"),
    ):
        sc = ax.scatter(true[:, 0], true[:, 1], c=vals, cmap=cmap, s=8,
                        vmin=(vmin if ttl != "|error|" else 0.0),
                        vmax=(vmax if ttl != "|error|" else max(vmax - vmin, 1e-12)))
        fig.colorbar(sc, ax=ax)
        ax.set_title(ttl)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    fig.suptitle(title)
    fig.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path)


__all__ = ["field_snapshot", "truth_vs_recovered"]
