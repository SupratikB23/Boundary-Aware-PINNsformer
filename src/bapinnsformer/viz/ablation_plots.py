"""Ablation plots (PRD §4.3 `viz/ablation_plots.py`, paper Fig. 8).

Accuracy vs sequence length / wall-clock / memory per pseudo-sequence
variant (E3). Headless: save to path, no ``show()``.
"""

from __future__ import annotations


def _rows_to_series(rows: list[dict], x_key: str, y_key: str) -> dict[str, tuple[list, list]]:
    series: dict[str, tuple[list, list]] = {}
    for r in rows:
        series.setdefault(str(r.get("variant", "?")), ([], []))[0].append(r.get(x_key))
        series[str(r.get("variant", "?"))][1].append(r.get(y_key))
    for v in series:
        paired = sorted(zip(series[v][0], series[v][1]), key=lambda p: (p[0] is None, p[0]))
        series[v] = ([p[0] for p in paired], [p[1] for p in paired])
    return series


def accuracy_vs_length(
    rows: list[dict],
    save_path: str = "accuracy_vs_L.png",
    metric: str = "rmse",
    title: str = "Accuracy vs sequence length",
) -> str:
    """Line plot of ``metric`` vs ``L`` per variant to file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .style import VARIANT_COLORS, apply_style, ensure_parent

    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for variant, (xs, ys) in sorted(_rows_to_series(rows, "L", metric).items()):
        ax.plot(xs, ys, marker="o", label=variant, color=VARIANT_COLORS.get(variant))
    ax.set_xlabel("sequence length L")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path)


def accuracy_vs_cost(
    rows: list[dict],
    save_path: str = "accuracy_vs_cost.png",
    metric: str = "rmse",
    cost: str = "wall_s",
    title: str = "Accuracy vs cost",
) -> str:
    """Scatter of ``metric`` vs ``cost`` (wall_s / peak_mb) per variant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .style import VARIANT_COLORS, apply_style, ensure_parent

    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for variant, _ in sorted(_rows_to_series(rows, cost, metric).items()):
        xs = [r[cost] for r in rows if str(r.get("variant")) == variant]
        ys = [r[metric] for r in rows if str(r.get("variant")) == variant]
        ls = [r.get("L") for r in rows if str(r.get("variant")) == variant]
        ax.scatter(xs, ys, label=variant, color=VARIANT_COLORS.get(variant), s=48)
        for x, y, lab in zip(xs, ys, ls):
            ax.annotate(str(lab), (x, y), fontsize=7)
    ax.set_xlabel(cost)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path)


def efficiency_vs_length(
    rows: list[dict],
    save_path: str = "efficiency_vs_L.png",
    cost: str = "wall_s",
    title: str = "Cost vs sequence length",
) -> str:
    """Line plot of ``cost`` (``wall_s``/``peak_mb``) vs ``L`` per variant.

    Works on timing-only ablation rows (no accuracy metric needed), so
    ``python tasks.py figures`` always emits the E3 efficiency panel.
    """
    return accuracy_vs_length(rows, save_path=save_path, metric=cost, title=title)


__all__ = ["accuracy_vs_length", "accuracy_vs_cost", "efficiency_vs_length"]
