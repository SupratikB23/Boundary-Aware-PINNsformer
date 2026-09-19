"""Data characterization: missingness, wind agreement, directional entropy.

PRD §4.3 `data/stats.py`: missingness characterization, station-vs-ERA5
wind agreement metrics, **wind directional entropy** per fitting window
(central to RQ2; used in E1/E2 stratification and E5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def missingness_table(df_qc: pd.DataFrame) -> pd.DataFrame:
    """Fraction of valid observations per (station, pollutant) + overall.

    Expects the ``valid``/``qc_flag`` columns from `qc.apply_qc`.
    Returns per-cell ``[n, n_valid, frac_valid, frac_missing]`` sorted
    by ``frac_valid`` ascending (worst stations first for the E0 table).
    """
    if df_qc.empty:
        return pd.DataFrame(columns=["station_id", "pollutant", "n", "n_valid", "frac_valid", "frac_missing"])
    rows = []
    for (sid, pol), g in df_qc.groupby(["station_id", "pollutant"]):
        n = len(g)
        nv = int(g["valid"].sum())
        rows.append(
            {
                "station_id": sid,
                "pollutant": pol,
                "n": n,
                "n_valid": nv,
                "frac_valid": float(nv / n) if n else 0.0,
                "frac_missing": float((g["qc_flag"] == "missing").mean()) if n else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("frac_valid").reset_index(drop=True)


def _circular_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Fisher–Lee circular correlation for angles in radians."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return float("nan")
    sa = np.sin(a - np.arctan2(np.sin(a).mean(), np.cos(a).mean()))
    sb = np.sin(b - np.arctan2(np.sin(b).mean(), np.cos(b).mean()))
    denom = np.sqrt((sa**2).sum() * (sb**2).sum())
    if denom == 0:
        return float("nan")
    return float((sa * sb).sum() / denom)


def wind_agreement(
    station_ws: np.ndarray,
    station_wd: np.ndarray,
    era5_u: np.ndarray,
    era5_v: np.ndarray,
) -> dict[str, float]:
    """Station-vs-ERA5 wind agreement: RMSE, bias, circular correlation.

    Args:
        station_ws: station wind speed (m/s).
        station_wd: station wind direction, degrees clockwise-from-north
            (meteorological convention).
        era5_u, era5_v: collocated ERA5 components (m/s, east/north).

    Returns:
        ``{speed_rmse, speed_bias, u_rmse, v_rmse, dir_circ_corr, n}``.
        Pairs with any NaN are excluded (masked, never imputed).
    """
    ws = np.asarray(station_ws, dtype=float).ravel()
    wd = np.asarray(station_wd, dtype=float).ravel()
    u = np.asarray(era5_u, dtype=float).ravel()
    v = np.asarray(era5_v, dtype=float).ravel()
    n = min(ws.size, wd.size, u.size, v.size)
    ws, wd, u, v = ws[:n], wd[:n], u[:n], v[:n]
    # station (ws, wd-met) → (u, v): u = -ws·sin(wd), v = -ws·cos(wd)
    wdr = np.deg2rad(wd)
    su = -ws * np.sin(wdr)
    sv = -ws * np.cos(wdr)
    m = np.isfinite(su) & np.isfinite(sv) & np.isfinite(u) & np.isfinite(v)
    su, sv, u, v = su[m], sv[m], u[m], v[m]
    if su.size == 0:
        return {"speed_rmse": float("nan"), "speed_bias": float("nan"), "u_rmse": float("nan"),
                "v_rmse": float("nan"), "dir_circ_corr": float("nan"), "n": 0}
    sp_e = np.hypot(u, v)
    sp_s = np.hypot(su, sv)
    dir_e = np.arctan2(u, v)  # math angle from north, comparable up to convention
    dir_s = np.arctan2(su, sv)
    return {
        "speed_rmse": float(np.sqrt(np.mean((sp_s - sp_e) ** 2))),
        "speed_bias": float(np.mean(sp_s - sp_e)),
        "u_rmse": float(np.sqrt(np.mean((su - u) ** 2))),
        "v_rmse": float(np.sqrt(np.mean((sv - v) ** 2))),
        "dir_circ_corr": float(_circular_corr(dir_s, dir_e)),
        "n": int(su.size),
    }


def directional_entropy(
    u: np.ndarray,
    v: np.ndarray,
    n_bins: int = 16,
    speed_min: float = 0.5,
) -> float:
    """Normalized histogram entropy of wind direction in [0, 1].

    Directions from ``arctan2(v, u)`` binned into ``n_bins`` equal
    sectors; calm samples (speed < ``speed_min``) excluded. Entropy is
    normalized by ``log(n_bins)``: 0 = unidirectional (poor tomography),
    1 = uniform rose (maximal angular diversity, RQ2 hypothesis).
    Returns NaN when no valid samples remain.
    """
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    m = np.isfinite(u) & np.isfinite(v) & (np.hypot(u, v) >= float(speed_min))
    u, v = u[m], v[m]
    if u.size == 0:
        return float("nan")
    ang = np.arctan2(v, u)  # (-pi, pi]
    hist, _ = np.histogram(ang, bins=int(n_bins), range=(-np.pi, np.pi))
    p = hist.astype(float) / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(n_bins))


def entropy_per_window(
    times: pd.DatetimeIndex | pd.Series,
    u: np.ndarray,
    v: np.ndarray,
    freq: str = "7D",
    n_bins: int = 16,
) -> pd.DataFrame:
    """Directional entropy stratified per time window (E1/E2 tables)."""
    t = pd.DatetimeIndex(pd.to_datetime(times))
    df = pd.DataFrame({"time": t, "u": np.asarray(u, dtype=float).ravel()[: len(t)],
                       "v": np.asarray(v, dtype=float).ravel()[: len(t)]})
    rows = []
    for start, g in df.groupby(pd.Grouper(key="time", freq=freq)):
        rows.append({"window_start": pd.Timestamp(start), "n": len(g),
                     "directional_entropy": directional_entropy(g["u"].to_numpy(), g["v"].to_numpy(), n_bins)})
    return pd.DataFrame(rows)
