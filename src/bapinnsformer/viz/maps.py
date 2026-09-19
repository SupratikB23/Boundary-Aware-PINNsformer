"""Domain maps (PRD §4.3 `viz/maps.py`, paper Fig. 1).

Domain rectangle + interior/perimeter stations + boundary arc-length
origin + example wind quiver. All functions save to ``save_path`` and
never call ``plt.show()``.
"""

from __future__ import annotations


def domain_map(
    bounds: tuple[float, float, float, float],
    interior_xy,
    perimeter_xy,
    wind_xyuv=None,
    save_path: str = "domain_map.png",
    title: str = "Delhi-NCR domain: interior vs perimeter stations",
) -> str:
    """Scatter the domain box, stations and optional wind field to file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .style import SECTOR_COLORS, apply_style, ensure_parent

    apply_style()
    import numpy as np

    xmin, xmax, ymin, ymax = (float(v) for v in bounds)
    interior = np.asarray(list(interior_xy), dtype=float).reshape(-1, 2)
    perimeter = np.asarray(list(perimeter_xy), dtype=float).reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.add_patch(
        plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, edgecolor="black", linewidth=1.4,
        )
    )
    # Arc-length origin marker (SW corner, PRD §2.1: CCW from SW).
    ax.plot([xmin], [ymin], marker="s", color=SECTOR_COLORS["S"], ms=7, label="s=0 (SW corner)")
    if interior.size:
        ax.scatter(interior[:, 0], interior[:, 1], s=26, c="#0072B2", label=f"interior (n={len(interior)})", zorder=3)
    if perimeter.size:
        ax.scatter(perimeter[:, 0], perimeter[:, 1], s=34, c="#D55E00", marker="^",
                   label=f"perimeter withheld (n={len(perimeter)})", zorder=3)
    if wind_xyuv is not None:
        w = np.asarray(list(wind_xyuv), dtype=float).reshape(-1, 4)
        if w.size:
            ax.quiver(w[:, 0], w[:, 1], w[:, 2], w[:, 3], color="#009E73", width=0.004, label="wind")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path)


__all__ = ["domain_map"]
