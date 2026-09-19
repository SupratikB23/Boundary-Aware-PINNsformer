"""Attention visualisation (PRD §4.3 `viz/attention_plots.py`, paper Fig. 9).

Attention weights over pseudo-tokens with boundary-hitting tokens
highlighted — the figure that visually justifies advection alignment.
Headless: save to path, no ``show()``.
"""

from __future__ import annotations


def attention_over_tokens(
    weights,
    boundary_hit=None,
    save_path: str = "attention.png",
    title: str = "Attention over pseudo-tokens",
) -> str:
    """Bar plot of mean attention per token; boundary-hit tokens hatched.

    Args:
        weights: ``(n_heads?, L)`` or ``(L,)`` attention weights.
        boundary_hit: optional boolean mask of length ``L`` flagging
            tokens whose backward characteristic hit the boundary.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .style import apply_style, ensure_parent

    apply_style()
    import numpy as np

    w = np.asarray(weights, dtype=float)
    if w.ndim == 2:
        w = w.mean(axis=0)
    if w.ndim != 1:
        raise ValueError("weights must be (L,) or (n_heads, L)")
    L = w.shape[0]
    hit = None
    if boundary_hit is not None:
        hit = np.asarray(list(boundary_hit), dtype=bool).reshape(-1)
        if hit.shape[0] != L:
            raise ValueError("boundary_hit must match L")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = ["#D55E00" if (hit is not None and h) else "#0072B2" for h in (hit if hit is not None else [False] * L)]
    bars = ax.bar(range(L), w, color=colors)
    if hit is not None and hit.any():
        for bar, h in zip(bars, hit):
            if h:
                bar.set_hatch("///")
                bar.set_edgecolor("black")
    ax.set_xlabel("pseudo-token k (0 = query)")
    ax.set_ylabel("mean attention weight")
    ax.set_title(title)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(facecolor="#D55E00", label="boundary-hitting token"),
                       Patch(facecolor="#0072B2", label="interior token")], loc="best")
    fig.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path)


__all__ = ["attention_over_tokens"]
