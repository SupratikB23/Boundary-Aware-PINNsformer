"""Publication style (PRD §4.3 `viz/style.py`).

Single source of matplotlib rcParams: serif-ish body, colour-blind-safe
Okabe–Ito palette, consistent sizes. Importing this module does NOT
touch matplotlib; call :func:`apply_style` explicitly.
"""

from __future__ import annotations

# Okabe–Ito colour-blind-safe palette (hex).
PALETTE: dict[str, str] = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
    "grey": "#999999",
}

VARIANT_COLORS: dict[str, str] = {
    "uniform_forward": "#999999",
    "uniform_backward": "#56B4E9",
    "advection_backward": "#D55E00",
    "advection_jitter": "#009E73",
}

SECTOR_COLORS: dict[str, str] = {
    "NW": "#D55E00",
    "N": "#0072B2",
    "E": "#009E73",
    "S": "#CC79A7",
    "W": "#E69F00",
}

RC_PARAMS: dict = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.prop_cycle": None,  # filled by apply_style with PALETTE order
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}


def apply_style() -> dict:
    """Apply publication rcParams; returns the applied dict. Lazy import."""
    import matplotlib as mpl

    params = dict(RC_PARAMS)
    try:
        from cycler import cycler

        params["axes.prop_cycle"] = cycler(
            color=[PALETTE[k] for k in ("blue", "vermillion", "bluish_green", "sky", "orange", "reddish_purple", "yellow", "grey")]
        )
    except ImportError:
        params.pop("axes.prop_cycle", None)
    params.pop("axes.prop_cycle", None) if params.get("axes.prop_cycle") is None else None
    for k, v in params.items():
        if v is not None:
            mpl.rcParams[k] = v
    return params


def ensure_parent(path: str) -> str:
    import os

    parent = os.path.dirname(os.path.abspath(str(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return str(path)


__all__ = ["PALETTE", "VARIANT_COLORS", "SECTOR_COLORS", "RC_PARAMS", "apply_style", "ensure_parent"]
