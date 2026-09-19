"""Rectangular airshed domain, boundary arc-length, normals, splits of stations.

PRD §4.3 `data/domain.py` + §2.1 contracts: rectangular lat/lon box
enclosing the airshed, projected to a local metric CRS (metres);
boundary parameterized by arc-length ``s in [0, P)`` traversed
counter-clockwise from the SW corner, with analytic outward normals;
interior-vs-perimeter station classification with buffer logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .align import project_to_metric


@dataclass(frozen=True)
class Domain:
    """Rectangular metric domain.

    Attributes:
        lon_min, lon_max, lat_min, lat_max: geographic box (degrees).
        lon0, lat0: projection reference (defaults to box centre).
        crs_epsg: optional target EPSG for `project_to_metric`.
        Lx, Ly: metric extents (m). P: perimeter (m).
    """

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    lon0: float
    lat0: float
    crs_epsg: int | None = None
    Lx: float = 0.0
    Ly: float = 0.0
    P: float = 0.0


def build_domain(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    lon0: float | None = None,
    lat0: float | None = None,
    crs_epsg: int | None = None,
) -> Domain:
    """Construct a rectangular domain and its metric extents."""
    if not (lon_min < lon_max and lat_min < lat_max):
        raise ValueError("build_domain: need lon_min<lon_max and lat_min<lat_max")
    lon0 = float(lon0) if lon0 is not None else (lon_min + lon_max) / 2.0
    lat0 = float(lat0) if lat0 is not None else (lat_min + lat_max) / 2.0
    xs, ys = project_to_metric(
        np.array([lon_min, lon_max]), np.array([lat_min, lat_max]), lon0, lat0, crs_epsg
    )
    Lx = float(xs[1] - xs[0])
    Ly = float(ys[1] - ys[0])
    if Lx <= 0 or Ly <= 0:
        raise ValueError("build_domain: non-positive metric extents")
    return Domain(
        lon_min=float(lon_min), lon_max=float(lon_max),
        lat_min=float(lat_min), lat_max=float(lat_max),
        lon0=float(lon0), lat0=float(lat0), crs_epsg=crs_epsg,
        Lx=Lx, Ly=Ly, P=2.0 * (Lx + Ly),
    )


def _metric_origin(domain: Domain) -> tuple[np.ndarray, np.ndarray]:
    """Metric coordinates of SW corner (x_sw, y_sw baseline arrays)."""
    x_sw, y_sw = project_to_metric(
        np.array([domain.lon_min]), np.array([domain.lat_min]),
        domain.lon0, domain.lat0, domain.crs_epsg,
    )
    return x_sw, y_sw


def s_to_xy(s: np.ndarray | float, domain: Domain) -> tuple[np.ndarray, np.ndarray]:
    """Map arc-length ``s in [0, P)`` (CCW from SW corner) to metric (x, y).

    Traversal: south edge (SW→SE), east edge (SE→NE), north edge
    (NE→NW), west edge (NW→SW). Vectorized over ``s``.
    """
    ss = np.mod(np.asarray(s, dtype=float), domain.P)
    x_sw, y_sw = _metric_origin(domain)
    x0 = float(x_sw[0])
    y0 = float(y_sw[0])
    Lx, Ly = domain.Lx, domain.Ly
    x = np.empty_like(ss)
    y = np.empty_like(ss)
    m_south = ss < Lx
    m_east = (ss >= Lx) & (ss < Lx + Ly)
    m_north = (ss >= Lx + Ly) & (ss < 2 * Lx + Ly)
    m_west = ss >= 2 * Lx + Ly
    x[m_south] = x0 + ss[m_south]
    y[m_south] = y0
    x[m_east] = x0 + Lx
    y[m_east] = y0 + (ss[m_east] - Lx)
    x[m_north] = x0 + Lx - (ss[m_north] - Lx - Ly)
    y[m_north] = y0 + Ly
    x[m_west] = x0
    y[m_west] = y0 + Ly - (ss[m_west] - 2 * Lx - Ly)
    return x, y


def outward_normals(s: np.ndarray | float, domain: Domain) -> np.ndarray:
    """Analytic outward unit normals ``(N, 2)`` at arc-lengths ``s``."""
    ss = np.mod(np.asarray(s, dtype=float), domain.P)
    Lx, Ly = domain.Lx, domain.Ly
    n = np.zeros((ss.size, 2))
    n[ss < Lx] = (0.0, -1.0)  # south
    n[(ss >= Lx) & (ss < Lx + Ly)] = (1.0, 0.0)  # east
    n[(ss >= Lx + Ly) & (ss < 2 * Lx + Ly)] = (0.0, 1.0)  # north
    n[ss >= 2 * Lx + Ly] = (-1.0, 0.0)  # west
    return n


def boundary_points(n_s: int, domain: Domain) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniform boundary discretization: ``(s, x, y, normals)``."""
    if n_s < 4:
        raise ValueError("boundary_points: need n_s >= 4")
    s = np.linspace(0.0, domain.P, n_s, endpoint=False)
    x, y = s_to_xy(s, domain)
    return s, x, y, outward_normals(s, domain)


def xy_to_metric(
    lon: np.ndarray | pd.Series, lat: np.ndarray | pd.Series, domain: Domain
) -> tuple[np.ndarray, np.ndarray]:
    """Project station lon/lat to domain metric coordinates."""
    return project_to_metric(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float), domain.lon0, domain.lat0, domain.crs_epsg)


def distance_to_boundary(x: np.ndarray, y: np.ndarray, domain: Domain) -> np.ndarray:
    """Min distance (m) from interior points to the rectangle boundary."""
    x_sw, y_sw = _metric_origin(domain)
    x0, y0 = float(x_sw[0]), float(y_sw[0])
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    return np.minimum.reduce([xa - x0, x0 + domain.Lx - xa, ya - y0, y0 + domain.Ly - ya])


def classify_stations(
    stations: pd.DataFrame,
    domain: Domain,
    buffer_m: float = 15_000.0,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> pd.DataFrame:
    """Label stations as ``interior`` vs ``perimeter`` via buffer logic.

    Stations within ``buffer_m`` of any boundary edge (or outside the
    box) are ``perimeter`` (E2 hold-out ring); the rest are ``interior``
    (fit set). Adds ``[x, y, dist_to_bdy, split]`` columns.
    """
    out = stations.copy()
    x, y = xy_to_metric(out[lon_col].to_numpy(), out[lat_col].to_numpy(), domain)
    out["x"] = x
    out["y"] = y
    x_sw, y_sw = _metric_origin(domain)
    x0, y0 = float(x_sw[0]), float(y_sw[0])
    inside = (x >= x0) & (x <= x0 + domain.Lx) & (y >= y0) & (y <= y0 + domain.Ly)
    d = distance_to_boundary(x, y, domain)
    d = np.where(inside, d, 0.0)  # outside counts as perimeter
    out["dist_to_bdy"] = d
    out["split"] = np.where(inside & (d >= float(buffer_m)), "interior", "perimeter")
    return out
