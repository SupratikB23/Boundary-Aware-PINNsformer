"""Boundary-inflow plots (PRD §4.3 `viz/boundary_plots.py`).

``C_b(s, t)`` Hovmöller diagrams, sector time series and the
recovered-NW-inflow vs withheld-FRP overlay (paper Figs. 6, 10).
Headless: save to path, no ``show()``.
"""

from __future__ import annotations


def hovmoller(
    Cb_st,
    save_path: str = "hovmoller.png",
    title: str = "Recovered boundary inflow $C_b(s, t)$",
    xlabel: str = "arc-length s [m]",
    ylabel: str = "time [h]",
) -> str:
    """Hovmöller (s × t) image of boundary inflow to file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .style import apply_style, ensure_parent

    apply_style()
    import numpy as np

    Cb = np.asarray(Cb_st, dtype=float)
    if Cb.ndim != 2:
        raise ValueError("Cb_st must be (n_s, n_t)")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    im = ax.imshow(Cb, aspect="auto", origin="lower", cmap="magma",
                   extent=(0, Cb.shape[0], 0, Cb.shape[1]))
    fig.colorbar(im, ax=ax, label="$C_b$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path)


def sector_timeseries(
    series_by_sector: dict[str, object],
    save_path: str = "sectors.png",
    title: str = "Sector inflow time series",
) -> str:
    """Line plot per sector (NW/N/E/S/W) to file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .style import SECTOR_COLORS, apply_style, ensure_parent

    apply_style()
    import numpy as np

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for sector, series in series_by_sector.items():
        y = np.asarray(list(series), dtype=float)
        ax.plot(y, label=str(sector), color=SECTOR_COLORS.get(str(sector)))
    ax.set_xlabel("time step")
    ax.set_ylabel("inflow rate [mass/time]")
    ax.set_title(title)
    ax.legend(ncol=5, loc="best")
    fig.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path)


def inflow_vs_frp(
    nw_inflow,
    frp,
    lags: tuple[int, ...] = (0,),
    save_path: str = "inflow_vs_frp.png",
    title: str = "Recovered NW inflow vs withheld VIIRS FRP (daily)",
) -> str:
    """Twin-axis daily overlay of NW inflow and withheld FRP to file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .style import apply_style, ensure_parent

    apply_style()
    import numpy as np

    q = np.asarray(list(nw_inflow), dtype=float)
    f = np.asarray(list(frp), dtype=float)
    if q.shape != f.shape:
        raise ValueError("nw_inflow and frp must share length")
    lag = int(lags[0]) if lags else 0
    if lag:
        f = np.roll(f, lag)
    fig, ax1 = plt.subplots(figsize=(7.0, 3.6))
    ax1.plot(q, color="#D55E00", label="NW inflow (recovered)")
    ax1.set_xlabel("day")
    ax1.set_ylabel("NW inflow [mass/time]", color="#D55E00")
    ax2 = ax1.twinx()
    ax2.bar(range(len(f)), f, alpha=0.45, color="#0072B2", label="FRP (withheld)")
    ax2.set_ylabel("FRP [MW]", color="#0072B2")
    ax1.set_title(title + (f" (FRP lag {lag}d)" if lag else ""))
    fig.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path)


__all__ = ["hovmoller", "sector_timeseries", "inflow_vs_frp"]
