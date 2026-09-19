"""External comparison: ours vs DSS / WRF-Chem shares.

PRD §4.3 (`eval/compare_external.py`), E4/E2. Published
inventory-based shares are an external *reference point*, never a
trainable baseline. This module therefore contains **no hardcoded
literature numbers** — pass them in (or via :func:`load_external`)
from the paper's comparison table so the provenance of each figure is
explicit. All math is dependency-light (numpy optional).
"""

from __future__ import annotations

import math


def divergence_row(
    ours: float,
    external: float,
    label_ours: str = "ours",
    label_ext: str = "external",
    tolerance: float = 0.10,
) -> dict:
    """One comparison row: absolute/relative divergence + agreement flag."""
    ours_f, ext_f = float(ours), float(external)
    abs_diff = ours_f - ext_f
    rel_diff = (abs_diff / ext_f) if abs(abs_diff) > 0 and abs(ext_f) > 0 else (
        0.0 if abs_diff == 0 else float("nan")
    )
    if math.isnan(ours_f) or math.isnan(ext_f):
        agreement = "missing"
    elif abs(abs_diff) <= tolerance:
        agreement = "agree"
    else:
        agreement = "divergent"
    return {
        "ours": ours_f,
        "external": ext_f,
        "label_ours": label_ours,
        "label_ext": label_ext,
        "abs_diff": float(abs_diff),
        "rel_diff": float(rel_diff),
        "agreement": agreement,
        "tolerance": float(tolerance),
    }


def divergence_table(
    ours_by_key: dict[str, float],
    external_by_source: dict[str, dict[str, float]],
    tolerance: float = 0.10,
) -> list[dict]:
    """Build the full divergence table over keys × external sources.

    Args:
        ours_by_key: e.g. ``{"Oct-Nov PM2.5": 0.42}``.
        external_by_source: e.g. ``{"DSS": {"Oct-Nov PM2.5": 0.38}}``.
    """
    rows: list[dict] = []
    for key, ours in ours_by_key.items():
        for source, table in external_by_source.items():
            if key not in table:
                rows.append(
                    {"key": key, "source": source, "agreement": "missing",
                     "ours": float(ours), "external": float("nan"),
                     "abs_diff": float("nan"), "rel_diff": float("nan"),
                     "tolerance": float(tolerance)}
                )
                continue
            row = divergence_row(ours, table[key], tolerance=tolerance)
            row["key"] = key
            row["source"] = source
            rows.append(row)
    return rows


def load_external(path: str) -> dict:
    """Load an external-shares file ``{source: {key: share}}`` (JSON/YAML)."""
    text_path = str(path)
    if text_path.endswith(".json"):
        import json

        with open(text_path, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML required for YAML external files") from exc
        with open(text_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("external file must map source -> {key: share}")
    return {str(s): {str(k): float(v) for k, v in t.items()} for s, t in data.items()}


def to_latex(rows: list[dict], caption: str = "External attribution comparison.") -> str:
    """Render divergence rows as a LaTeX tabular (paper/tables input)."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Window & Source & Ours & External & $\\Delta$ & Verdict \\\\",
        "\\midrule",
    ]
    for r in rows:
        def fmt(v):
            try:
                return f"{float(v):.2f}"
            except (TypeError, ValueError):
                return "--"
        lines.append(
            f"{r.get('key','--')} & {r.get('source','--')} & {fmt(r.get('ours'))} "
            f"& {fmt(r.get('external'))} & {fmt(r.get('abs_diff'))} "
            f"& {r.get('agreement','--')} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", "\\end{table}"]
    return "\n".join(lines)


__all__ = ["divergence_row", "divergence_table", "load_external", "to_latex"]
