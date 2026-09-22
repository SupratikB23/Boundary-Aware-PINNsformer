"""ERA5 retrieval + cache.

PRD §4.3 `data/era5_ingest.py`: CDS API retrieval of hourly 10 m
``u10``/``v10``, boundary-layer height, temperature, humidity over the
domain box; caching by (box, variable, month); write to NetCDF.

``cdsapi`` is lazily imported inside :func:`fetch_era5` so that
``import bapinnsformer.data.era5_ingest`` never requires CDS
credentials or the client library (unit-test / offline safe).
``xarray`` is an optional import with a netCDF4/numpy fallback.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:  # optional; guarded so offline/CI import always works
    import xarray as xr
except Exception:  # pragma: no cover
    xr = None  # type: ignore[assignment]

ERA5_PRODUCT = "reanalysis-era5-single-levels"
DEFAULT_VARS = ("10m_u_component_of_wind", "10m_v_component_of_wind", "boundary_layer_height")


def _norm_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Validate (lon_min, lon_max, lat_min, lat_max); raise on inversion."""
    lon_min, lon_max, lat_min, lat_max = (float(v) for v in box)
    if not (lon_min < lon_max and lat_min < lat_max):
        raise ValueError(f"Invalid box {box}: need lon_min<lon_max, lat_min<lat_max")
    return (lon_min, lon_max, lat_min, lat_max)


def build_cds_request(
    box: tuple[float, float, float, float],
    variables: tuple[str, ...] | list[str],
    year: int,
    month: int,
    day: list[str] | None = None,
    time: list[str] | None = None,
    product: str = ERA5_PRODUCT,
) -> dict[str, Any]:
    """Build a CDS API request dict for ERA5 single levels.

    Args:
        box: (lon_min, lon_max, lat_min, lat_max) in degrees.
        variables: CDS short names, e.g. ``10m_u_component_of_wind``.
        year, month: calendar month to retrieve.
        day/time: CDS ``day``/``time`` lists; default all days/hours.
        product: CDS dataset name.

    Returns:
        Dict with ``dataset`` + ``request`` keys ready for
        ``cdsapi.Client().retrieve(dataset, request, target)``.
    """
    lon_min, lon_max, lat_min, lat_max = _norm_box(box)
    if not (1 <= month <= 12):
        raise ValueError(f"month must be 1..12, got {month}")
    days = day or [f"{d:02d}" for d in range(1, 32)]
    times = time or [f"{h:02d}:00" for h in range(24)]
    # CDS area convention: [North, West, South, East]
    request = {
        "product_type": ["reanalysis"],
        "variable": list(variables),
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": days,
        "time": times,
        "area": [lat_max, lon_min, lat_min, lon_max],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    return {"dataset": product, "request": request}


def cache_path(
    cache_dir: str | Path,
    box: tuple[float, float, float, float],
    variables: tuple[str, ...] | list[str],
    year: int,
    month: int,
) -> Path:
    """Deterministic cache path keyed by (box, variables, year, month).

    Filename embeds a short sha1 of the canonical key plus a human
    readable ``era5_YYYYMM`` prefix.
    """
    lon_min, lon_max, lat_min, lat_max = _norm_box(box)
    key = json.dumps(
        {
            "box": [lon_min, lon_max, lat_min, lat_max],
            "vars": sorted(map(str, variables)),
            "year": int(year),
            "month": int(month),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return Path(cache_dir) / f"era5_{year:04d}{month:02d}_{digest}.nc"


def save_era5(dataset: Any, path: str | Path) -> Path:
    """Write an xarray Dataset (or numpy-dict fallback) to NetCDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if xr is not None and hasattr(dataset, "to_netcdf"):
        dataset.to_netcdf(str(path))
        return path
    # Fallback: dict of ndarrays with coords
    try:
        from netCDF4 import Dataset as NCDataset
    except Exception as exc:  # pragma: no cover
        raise ImportError("No xarray nor netCDF4 available to save ERA5") from exc
    data = dict(dataset) if isinstance(dataset, dict) else {}
    with NCDataset(str(path), "w") as nc:
        for dim, size in data.get("__dims__", {}).items():
            nc.createDimension(dim, size)
        for name, arr in data.items():
            if name == "__dims__":
                continue
            arr = np.asarray(arr)
            dims = data.get(f"__dims__{name}", tuple(data.get("__dims__", {}).keys())[: arr.ndim])
            var = nc.createVariable(name, arr.dtype, dims)
            var[:] = arr
    return path


def load_era5(path: str | Path) -> Any:
    """Load a cached ERA5 NetCDF month; returns xarray Dataset if available."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ERA5 cache miss: {path}")
    if xr is not None:
        return xr.open_dataset(str(path))
    try:
        from netCDF4 import Dataset as NCDataset
    except Exception as exc:  # pragma: no cover
        raise ImportError("No xarray nor netCDF4 available to load ERA5") from exc
    return NCDataset(str(path), "r")


def fetch_era5(
    box: tuple[float, float, float, float],
    variables: tuple[str, ...] | list[str] = DEFAULT_VARS,
    year: int = 2023,
    month: int = 1,
    cache_dir: str | Path = "data/raw/era5",
    overwrite: bool = False,
) -> Path:
    """Retrieve one ERA5 month via CDS, using the on-disk cache.

    Lazy-imports ``cdsapi`` only when a download is actually needed and
    a cache file is absent (or ``overwrite=True``). Returns the NetCDF
    path (cached or freshly downloaded).
    """
    target = cache_path(cache_dir, box, tuple(variables), year, month)
    if target.exists() and not overwrite:
        return target
    try:
        import cdsapi  # lazy: must not be required at module import
    except Exception as exc:
        raise ImportError(
            "cdsapi is required for ERA5 download but is not installed. "
            "Install cdsapi and set ~/.cdsapirc credentials, or place the "
            f"expected NetCDF at {target}."
        ) from exc
    spec = build_cds_request(box, tuple(variables), year, month)
    target.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve(spec["dataset"], spec["request"], str(target))
    return target
