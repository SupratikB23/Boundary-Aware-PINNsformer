"""Validation splits with hash-stable serialization.

PRD §4.3 `data/splits.py` (§6 E2): interior-fit/perimeter-hold-out split
(E2 primary), leave-one-interior-station-out (secondary), temporal
blocks. Split definitions are serialized to disk and referenced by the
sha256 of their canonical JSON so every experiment uses the *same*
split. Stdlib + numpy/pandas only; CPU-runnable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _canonical(obj: Any) -> str:
    """Canonical JSON string (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def split_hash(split: dict[str, Any]) -> str:
    """Stable sha256 hex over the canonical JSON of a split definition."""
    return hashlib.sha256(_canonical(split).encode("utf-8")).hexdigest()


def _station_id(st: Any) -> str:
    if isinstance(st, dict):
        for k in ("id", "station_id", "code", "name"):
            if k in st:
                return str(st[k])
        raise ValueError(f"station dict has no id field: {st}")
    return str(st)


def _coords(st: Any) -> tuple[float, float]:
    if isinstance(st, dict):
        if "x" in st and "y" in st:
            return float(st["x"]), float(st["y"])
        if "lon" in st and "lat" in st:
            return float(st["lon"]), float(st["lat"])
    raise ValueError(f"station has no (x, y) or (lon, lat): {st}")


def split_interior_perimeter(stations: Any, bounds: tuple[float, float, float, float], buffer: float = 0.0) -> dict[str, Any]:
    """Classify stations by distance-to-edge rule (bounds-based protocol).

    Stations within ``buffer`` (same units as coordinates) of any domain
    edge are **perimeter** (withheld); the rest are **interior** (fit).
    ``buffer <= 0`` falls back to a median-distance split so both sets
    are non-empty. Returns both ``fit_ids``/``val_ids`` and
    ``fit``/``holdout`` aliases plus ``hash``.
    """
    xmin, xmax, ymin, ymax = (float(v) for v in bounds)
    if isinstance(stations, pd.DataFrame):
        # DataFrame path: expect longitude/latitude or x/y columns.
        lon_col = "longitude" if "longitude" in stations.columns else ("lon" if "lon" in stations.columns else ("x" if "x" in stations.columns else None))
        lat_col = "latitude" if "latitude" in stations.columns else ("lat" if "lat" in stations.columns else ("y" if "y" in stations.columns else None))
        id_col = "station_id" if "station_id" in stations.columns else ("id" if "id" in stations.columns else stations.columns[0])
        if lon_col is None or lat_col is None:
            raise ValueError("split_interior_perimeter: DataFrame needs lon/lat or x/y columns")
        recs = [
            {"station_id": str(r[id_col]), "x": float(r[lon_col]), "y": float(r[lat_col])}
            for _, r in stations.iterrows()
        ]
        return split_interior_perimeter(recs, (xmin, xmax, ymin, ymax), buffer=buffer)
    ids = [_station_id(s) for s in stations]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate station ids")
    dist_edge = []
    for s in stations:
        x, y = _coords(s)
        dist_edge.append(min(x - xmin, xmax - x, y - ymin, ymax - y))
    dist_edge = np.asarray(dist_edge, dtype=float)
    if buffer and buffer > 0:
        is_perim = dist_edge <= float(buffer)
        if not is_perim.any() or is_perim.all():
            is_perim = dist_edge <= float(np.median(dist_edge))
    else:
        is_perim = dist_edge <= float(np.median(dist_edge))
    fit_ids = sorted(i for i, p in zip(ids, is_perim) if not p)
    val_ids = sorted(i for i, p in zip(ids, is_perim) if p)
    if not fit_ids or not val_ids:
        raise ValueError("split produced an empty set; adjust buffer/stations")
    body = {
        "protocol": "interior_fit__perimeter_holdout",
        "bounds": [xmin, xmax, ymin, ymax],
        "buffer": float(buffer or 0.0),
        "fit_ids": fit_ids,
        "val_ids": val_ids,
        "fit": list(fit_ids),
        "holdout": list(val_ids),
    }
    return {**body, "hash": split_hash(body)}


def interior_perimeter_split(
    stations: pd.DataFrame,
    split_col: str = "split",
    station_col: str = "station_id",
) -> dict[str, Any]:
    """Build the E2 primary split from a classified station table.

    Expects ``split_col`` with ``interior``/``perimeter`` labels (see
    `domain.classify_stations`). Returns ``{name, fit, holdout, hash}``
    (plus ``fit_ids``/``val_ids`` aliases). Perimeter stations never
    appear in the fit set — enforced and asserted here.
    """
    if split_col not in stations.columns:
        # Fallback: treat `stations` as raw records + derive via median split.
        raise ValueError(f"interior_perimeter_split: missing column '{split_col}'")
    fit = sorted(stations.loc[stations[split_col] == "interior", station_col].astype(str).unique().tolist())
    holdout = sorted(stations.loc[stations[split_col] == "perimeter", station_col].astype(str).unique().tolist())
    if not fit:
        raise ValueError("interior_perimeter_split: empty fit set (no interior stations)")
    if not holdout:
        raise ValueError("interior_perimeter_split: empty holdout set (no perimeter stations)")
    leak = set(fit) & set(holdout)
    if leak:
        raise ValueError(f"interior_perimeter_split: stations in both sets: {sorted(leak)}")
    body = {"name": "interior_fit__perimeter_holdout", "fit": fit, "holdout": holdout,
            "fit_ids": list(fit), "val_ids": list(holdout)}
    return {**body, "hash": split_hash(body)}


def leave_one_out(fit_ids: list[str]) -> Any:
    """Yield ``(train_ids, held_ids)`` for leave-one-interior-station-out."""
    ids = list(fit_ids)
    for k in range(len(ids)):
        yield [i for m, i in enumerate(ids) if m != k], [ids[k]]


def loiso_splits(
    stations: Any,
    split_col: str = "split",
    station_col: str = "station_id",
) -> list[dict[str, Any]]:
    """Leave-one-interior-station-out splits (E2 secondary protocol).

    Accepts either a classified DataFrame (preferred) or an explicit
    iterable of fit-station ids.
    """
    if isinstance(stations, pd.DataFrame):
        primary = interior_perimeter_split(stations, split_col, station_col)
        fit_ids: list[str] = primary["fit"]
        perimeter: list[str] = primary["holdout"]
    else:
        fit_ids = [str(s) for s in stations]
        perimeter = []
    out: list[dict[str, Any]] = []
    for tr, held in leave_one_out(fit_ids):
        body = {"name": f"loiso__holdout_{held[0]}", "fit": tr, "holdout": held, "perimeter_ring": perimeter,
                "fit_ids": list(tr), "val_ids": list(held)}
        out.append({**body, "hash": split_hash(body)})
    return out


def temporal_blocks(clock_or_n: Any, n_blocks: int = 4, name: str = "temporal_blocks") -> Any:
    """Contiguous temporal blocks (dual-signature for compat).

    - Task signature: ``temporal_blocks(clock: DatetimeIndex, n_blocks,
      name)`` → list of ``{name, start, end, n_hours, hash}`` dicts.
    - Legacy signature: ``temporal_blocks(n_times: int, n_blocks)`` →
      list of ``(start, stop)`` index tuples.
    """
    if isinstance(clock_or_n, (int, np.integer)):
        n, b = int(clock_or_n), int(n_blocks)
        if n <= 0 or b <= 0:
            raise ValueError("n_times and n_blocks must be positive")
        edges = [round(i * n / b) for i in range(b + 1)]
        return [(edges[i], edges[i + 1]) for i in range(b) if edges[i + 1] > edges[i]]
    clock = pd.DatetimeIndex(pd.to_datetime(clock_or_n))
    if int(n_blocks) < 1:
        raise ValueError("temporal_blocks: n_blocks must be >= 1")
    edges = np.linspace(0, len(clock), int(n_blocks) + 1).astype(int)
    blocks: list[dict[str, Any]] = []
    for i in range(int(n_blocks)):
        s, e = int(edges[i]), int(edges[i + 1]) - 1
        if e < s:
            continue
        body = {
            "name": f"{name}__block{i}",
            "start": pd.Timestamp(clock[s]).isoformat(),
            "end": pd.Timestamp(clock[e]).isoformat(),
            "n_hours": int(e - s + 1),
        }
        blocks.append({**body, "hash": split_hash(body)})
    return blocks


def save_split(split: dict[str, Any], path: str | Path) -> Path:
    """Write a split dict as canonical JSON (hash-verified on load)."""
    path = Path(path)
    body = {k: v for k, v in split.items() if k != "hash"}
    h = split_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(body)
    payload["hash"] = h
    path.write_text(_canonical(payload), encoding="utf-8")
    return path


def load_split(path: str | Path) -> dict[str, Any]:
    """Load a split JSON and verify its embedded hash."""
    import json as _json

    path = Path(path)
    payload = _json.loads(path.read_text(encoding="utf-8"))
    h = payload.get("hash")
    body = {k: v for k, v in payload.items() if k != "hash"}
    if h != split_hash(body):
        raise ValueError(f"load_split: hash mismatch in {path} (file may be edited)")
    payload["hash"] = h
    return payload


__all__ = [
    "split_hash",
    "split_interior_perimeter",
    "interior_perimeter_split",
    "leave_one_out",
    "loiso_splits",
    "temporal_blocks",
    "save_split",
    "load_split",
]
