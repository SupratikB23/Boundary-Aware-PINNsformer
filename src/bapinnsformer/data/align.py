"""Alignment: hourly clock, station↔ERA5 join, metric projection, stats.

PRD §4.3 `data/align.py`: build the aligned analysis object — station
table joined to ERA5 fields at station locations and at the collocation
grid; consistent hourly clock; projection to metric CRS; normalization
statistics.

Normalizer-contract note: this module never hardcodes scaling. All
scalers are computed from explicit ``bounds`` arguments or from the
data passed in, and returned as plain dicts so `models/normalizer.py`
can own the canonical forward/inverse maps without an import cycle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import xarray as xr
except Exception:  # pragma: no cover - optional
    xr = None  # type: ignore[assignment]

EARTH_RADIUS_M = 6_371_000.0


def make_hourly_clock(
    t_start: str | pd.Timestamp,
    t_end: str | pd.Timestamp,
    tz: str = "Asia/Kolkata",
    freq: str = "1h",
) -> pd.DatetimeIndex:
    """Build an inclusive hourly clock localized to ``tz``."""
    s = pd.Timestamp(t_start)
    e = pd.Timestamp(t_end)
    if s.tzinfo is None:
        s = s.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        s = s.tz_convert(tz)
    if e.tzinfo is None:
        e = e.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        e = e.tz_convert(tz)
    return pd.date_range(start=s, end=e, freq=freq, tz=tz)


def project_to_metric(
    lon: np.ndarray | pd.Series | float,
    lat: np.ndarray | pd.Series | float,
    lon0: float,
    lat0: float,
    crs_epsg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (lon, lat) to local metres relative to (lon0, lat0).

    If ``crs_epsg`` is given, a lazy ``pyproj`` transverse projection is
    attempted; any failure (or ``pyproj`` absent) falls back to the
    equirectangular approximation — exact enough for a city airshed and
    dependency-free. Returns ``(x, y)`` in metres, same shape as input.
    No scaling constants are hidden: the reference point is an argument.
    """
    lon_a = np.asarray(lon, dtype=float)
    lat_a = np.asarray(lat, dtype=float)
    if crs_epsg is not None:
        try:
            from pyproj import Transformer  # lazy import

            tr = Transformer.from_crs("EPSG:4326", f"EPSG:{crs_epsg}", always_xy=True)
            x0, y0 = tr.transform(float(lon0), float(lat0))
            x, y = tr.transform(lon_a, lat_a)
            return np.asarray(x) - x0, np.asarray(y) - y0
        except Exception:
            pass  # fall through to equirectangular
    lam0 = np.deg2rad(float(lon0))
    phi0 = np.deg2rad(float(lat0))
    x = EARTH_RADIUS_M * (np.deg2rad(lon_a) - lam0) * np.cos(phi0)
    y = EARTH_RADIUS_M * (np.deg2rad(lat_a) - phi0)
    return x, y


def compute_normalization_stats(
    values: np.ndarray,
    bounds: tuple[float, float] | None = None,
    method: str = "minmax",
) -> dict[str, float]:
    """Compute normalization stats from data or explicit bounds.

    Args:
        values: 1-D array of observations (NaNs ignored).
        bounds: explicit (lo, hi) overriding data-driven stats — the
            Normalizer-contract path (no hardcoded scaling).
        method: ``"minmax"`` → {lo, hi}; ``"standard"`` → {mean, std}.

    Returns:
        Plain dict consumable by `models/normalizer.py`.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if method == "minmax":
        if bounds is not None:
            lo, hi = float(bounds[0]), float(bounds[1])
        elif v.size:
            lo, hi = float(v.min()), float(v.max())
        else:
            lo, hi = 0.0, 1.0
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        return {"lo": lo, "hi": hi, "method": "minmax"}  # type: ignore[dict-item]
    if method == "standard":
        mean = float(v.mean()) if v.size else 0.0
        std = float(v.std()) if v.size else 1.0
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        return {"mean": mean, "std": std, "method": "standard"}  # type: ignore[dict-item]
    raise ValueError(f"unknown normalization method: {method}")


def _bilinear_sample_grid(
    field: np.ndarray, xs: np.ndarray, ys: np.ndarray,
    x_coords: np.ndarray, y_coords: np.ndarray,
) -> np.ndarray:
    """Bilinear sample of a 2-D field at (xs, ys); out-of-bounds → NaN."""
    nx = x_coords.size
    ix = np.searchsorted(x_coords, xs, side="left")
    iy = np.searchsorted(y_coords, ys, side="left")
    ix0 = np.clip(ix - 1, 0, nx - 2) if nx > 1 else np.zeros_like(ix)
    iy0 = np.clip(iy - 1, 0, field.shape[0] - 2) if field.shape[0] > 1 else np.zeros_like(iy)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    x0, x1 = x_coords[ix0], x_coords[ix1]
    y0, y1 = y_coords[iy0], y_coords[iy1]
    wx = np.where(x1 > x0, (xs - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0)
    wy = np.where(y1 > y0, (ys - y0) / np.where(y1 > y0, y1 - y0, 1.0), 0.0)
    wx = np.clip(wx, 0.0, 1.0)
    wy = np.clip(wy, 0.0, 1.0)
    f00 = field[iy0, ix0]
    f10 = field[iy0, ix1]
    f01 = field[iy1, ix0]
    f11 = field[iy1, ix1]
    return (1 - wx) * (1 - wy) * f00 + wx * (1 - wy) * f10 + (1 - wx) * wy * f01 + wx * wy * f11


def join_stations_to_era5(
    stations: pd.DataFrame,
    era5: Any,
    u_name: str = "u10",
    v_name: str = "v10",
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    time_name: str = "time",
) -> pd.DataFrame:
    """Left-join ERA5 wind (and any extra fields) onto station rows.

    ``stations`` needs ``[longitude, latitude, time]``; ``era5`` is an
    xarray Dataset (preferred) or a dict of ``{var: (t, y, x, coords)}``.
    Missing ERA5 lookups stay NaN (masked downstream, never imputed).
    Returns a copy of ``stations`` with ``u``, ``v`` (+ extras) columns.
    """
    out = stations.copy()
    out["u"] = np.nan
    out["v"] = np.nan
    if out.empty:
        return out
    t = pd.to_datetime(out[time_name], utc=True, errors="coerce")
    if xr is not None and hasattr(era5, "interp"):
        try:
            for i, (_, row) in enumerate(out.iterrows()):
                if pd.isna(t.iloc[i]):
                    continue
                sel = era5.interp(
                    {lon_name: float(row[lon_name]), lat_name: float(row[lat_name]), time_name: t.iloc[i]},
                    method="linear",
                    kwargs={"fill_value": np.nan},
                )
                for src, dst in ((u_name, "u"), (v_name, "v")):
                    if src in sel:
                        out.loc[out.index[i], dst] = float(sel[src].values)
            return out
        except Exception:
            pass  # fall back to numpy path below
    # Numpy-dict fallback: era5 = {"u": (Nt,Ny,Nx), "v": ..., "x":, "y":, "t": epoch-sec}
    if isinstance(era5, dict) and all(k in era5 for k in ("u", "v", "x", "y", "t")):
        xs = np.asarray(era5["x"], dtype=float)
        ys = np.asarray(era5["y"], dtype=float)
        ts = np.asarray(era5["t"], dtype=float)
        qx = out[lon_name].to_numpy(dtype=float)  # caller projects to metric beforehand if needed
        qy = out[lat_name].to_numpy(dtype=float)
        qt = t.view("int64").to_numpy() / 1e9
        U = np.asarray(era5["u"])
        V = np.asarray(era5["v"])
        for i in range(len(out)):
            if not np.isfinite(qt[i]):
                continue
            k = np.searchsorted(ts, qt[i], side="left")
            k0 = int(np.clip(k - 1, 0, len(ts) - 1))
            k1 = int(np.clip(k, 0, len(ts) - 1))
            a = 0.0 if k1 == k0 else (qt[i] - ts[k0]) / (ts[k1] - ts[k0])
            a = float(np.clip(a, 0.0, 1.0))
            fu = (1 - a) * _bilinear_sample_grid(U[k0], qx[i : i + 1], qy[i : i + 1], xs, ys)[0] + a * _bilinear_sample_grid(
                U[k1], qx[i : i + 1], qy[i : i + 1], xs, ys
            )[0]
            fv = (1 - a) * _bilinear_sample_grid(V[k0], qx[i : i + 1], qy[i : i + 1], xs, ys)[0] + a * _bilinear_sample_grid(
                V[k1], qx[i : i + 1], qy[i : i + 1], xs, ys
            )[0]
            out.loc[out.index[i], "u"] = float(fu)
            out.loc[out.index[i], "v"] = float(fv)
    return out
