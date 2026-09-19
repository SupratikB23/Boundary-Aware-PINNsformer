"""Per-station quality control → validity mask (no imputation).

PRD §4.3 `data/qc.py`: range checks, flatline detection, spike
detection, duplicate timestamps, timezone normalization (IST), unit
checks. Produces a boolean validity mask column — **rows are flagged,
never filled or dropped** here so the masked loss can ignore them
without bias (PRD §2.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Plausible ambient ranges in native CPCB units (µg/m³ for PM/gases
# except CO mg/m³, met units noted). Values outside → invalid.
RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "PM2.5": (0.0, 2000.0),
    "PM10": (0.0, 3000.0),
    "PM1": (0.0, 1500.0),
    "NO2": (0.0, 1000.0),
    "NO": (0.0, 1000.0),
    "NOX": (0.0, 1500.0),
    "SO2": (0.0, 1500.0),
    "CO": (0.0, 50.0),
    "O3": (0.0, 800.0),
    "NH3": (0.0, 1000.0),
    "WS": (0.0, 60.0),
    "WD": (0.0, 360.0),
    "TEMP": (-10.0, 55.0),
    "AT": (-10.0, 55.0),
    "RH": (0.0, 100.0),
    "BP": (800.0, 1100.0),
}

DEFAULT_FLATLINE_WINDOW = 12  # hours of identical value → suspect sensor
DEFAULT_FLATLINE_TOL = 1e-6
DEFAULT_SPIKE_Z = 6.0
DEFAULT_SPIKE_WINDOW = 25  # centered rolling window (hours)


def _to_ist(df: pd.DataFrame, col: str = "time") -> pd.DataFrame:
    """Normalize ``time`` to Asia/Kolkata-aware stamps.

    Naive stamps are assumed IST (CPCB convention); aware stamps are
    converted. Returns a copy.
    """
    out = df.copy()
    t = pd.to_datetime(out[col], errors="coerce", utc=False)
    try:
        tz = "Asia/Kolkata"
        if t.dt.tz is None:
            t = t.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        else:
            t = t.dt.tz_convert(tz)
    except Exception:
        pass
    out[col] = t
    return out


def detect_flatline(series: pd.Series, window: int = DEFAULT_FLATLINE_WINDOW, tol: float = DEFAULT_FLATLINE_TOL) -> pd.Series:
    """Flag runs of near-constant values of length ≥ ``window``.

    Uses rolling range (max−min); points whose trailing window range
    is ≤ ``tol`` are flagged. NaNs never flag (they are handled by the
    missingness mask, not here). Returns boolean Series aligned to input.
    """
    s = pd.to_numeric(series, errors="coerce")
    roll_max = s.rolling(window, min_periods=window).max()
    roll_min = s.rolling(window, min_periods=window).min()
    flagged = (roll_max - roll_min) <= tol
    return flagged.fillna(False).astype(bool)


def detect_spikes(series: pd.Series, z_thresh: float = DEFAULT_SPIKE_Z, window: int = DEFAULT_SPIKE_WINDOW) -> pd.Series:
    """Flag isolated spikes via centered rolling z-score.

    ``|x − rolling_median| / (1.4826·MAD)`` ≥ ``z_thresh`` → spike.
    Windows with zero MAD fall back to rolling std. NaNs never flag.
    """
    s = pd.to_numeric(series, errors="coerce")
    med = s.rolling(window, center=True, min_periods=max(3, window // 3)).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=max(3, window // 3)).median()
    scale = 1.4826 * mad
    std = s.rolling(window, center=True, min_periods=max(3, window // 3)).std()
    scale = scale.where(scale > 1e-9, std)
    z = (s - med).abs() / scale.replace(0.0, np.nan)
    return (z >= z_thresh).fillna(False).astype(bool)


def apply_qc(
    df_long: pd.DataFrame,
    flatline_window: int = DEFAULT_FLATLINE_WINDOW,
    flatline_tol: float = DEFAULT_FLATLINE_TOL,
    spike_z: float = DEFAULT_SPIKE_Z,
    spike_window: int = DEFAULT_SPIKE_WINDOW,
    range_checks: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Apply QC to a tidy long table; add ``valid`` + ``qc_flag`` columns.

    Flags: ``ok`` | ``missing`` | ``range`` | ``flatline`` | ``spike``
    | ``duplicate``. Priority: missing > duplicate > range > flatline >
    spike > ok. Input rows are preserved 1:1 — no imputation, no drops.
    Timezone is normalized to IST.
    """
    ranges = range_checks or RANGE_CHECKS
    out = _to_ist(df_long.copy())
    out["qc_flag"] = "ok"
    out["valid"] = True

    is_missing = out["value"].isna()
    out.loc[is_missing, "qc_flag"] = "missing"
    out.loc[is_missing, "valid"] = False

    # Duplicates: same station × time × pollutant appearing >1 → keep
    # first valid, flag the rest as duplicate.
    dup = out.duplicated(subset=["station_id", "time", "pollutant"], keep="first")
    out.loc[dup & (out["qc_flag"] == "ok"), "qc_flag"] = "duplicate"
    out.loc[dup, "valid"] = False

    # Range checks per pollutant.
    for pol, (lo, hi) in ranges.items():
        m = (out["pollutant"] == pol) & out["value"].notna() & ((out["value"] < lo) | (out["value"] > hi))
        out.loc[m & (out["qc_flag"] == "ok"), "qc_flag"] = "range"
        out.loc[m, "valid"] = False
    # Generic negativity guard for pollutants not in the table (met WD etc. excluded).
    known_nonneg = {"PM2.5", "PM10", "PM1", "NO2", "NO", "NOX", "SO2", "CO", "O3", "NH3"}
    m = out["pollutant"].isin(known_nonneg) & out["value"].notna() & (out["value"] < 0.0)
    out.loc[m & (out["qc_flag"] == "ok"), "qc_flag"] = "range"
    out.loc[m, "valid"] = False

    # Flatline / spike per (station, pollutant) time-ordered series.
    out = out.sort_values(["station_id", "pollutant", "time"]).reset_index(drop=True)
    for (_, _), idx in out.groupby(["station_id", "pollutant"]).groups.items():
        ii = out.index[np.asarray(idx)]
        s = out.loc[ii, "value"]
        fl = detect_flatline(s, window=flatline_window, tol=flatline_tol)
        sp = detect_spikes(s, z_thresh=spike_z, window=spike_window)
        fl_idx = ii[fl.values]
        sp_idx = ii[sp.values]
        out.loc[fl_idx[ out.loc[fl_idx, "qc_flag"] == "ok"], "qc_flag"] = "flatline"
        out.loc[fl_idx, "valid"] = False
        rest = sp_idx[out.loc[sp_idx, "qc_flag"] == "ok"]
        out.loc[rest, "qc_flag"] = "spike"
        out.loc[sp_idx, "valid"] = False

    out["valid"] = out["valid"].astype(bool)
    return out.sort_values(["station_id", "time", "pollutant"]).reset_index(drop=True)


def qc_report(df_qc: pd.DataFrame) -> pd.DataFrame:
    """Summarize QC outcomes per (station, pollutant).

    Returns columns ``[station_id, pollutant, n, n_valid, frac_valid,
    frac_missing, frac_range, frac_flatline, frac_spike,
    frac_duplicate]``. Pure aggregation — no thresholding here;
    inclusion thresholds live in configs (`data/cpcb.yaml`).
    """
    if df_qc.empty:
        return pd.DataFrame(
            columns=["station_id", "pollutant", "n", "n_valid", "frac_valid",
                     "frac_missing", "frac_range", "frac_flatline", "frac_spike", "frac_duplicate"]
        )
    rows: list[dict[str, object]] = []
    for (sid, pol), g in df_qc.groupby(["station_id", "pollutant"]):
        n = len(g)
        flags = g["qc_flag"].value_counts()
        rows.append(
            {
                "station_id": sid,
                "pollutant": pol,
                "n": n,
                "n_valid": int(g["valid"].sum()),
                "frac_valid": float(g["valid"].mean()),
                "frac_missing": float((g["qc_flag"] == "missing").mean()),
                "frac_range": float((g["qc_flag"] == "range").mean()),
                "frac_flatline": float((g["qc_flag"] == "flatline").mean()),
                "frac_spike": float((g["qc_flag"] == "spike").mean()),
                "frac_duplicate": float((g["qc_flag"] == "duplicate").mean()),
                **{f"n_{k}": int(flags.get(k, 0)) for k in ("ok",)},
            }
        )
    return pd.DataFrame(rows).sort_values(["station_id", "pollutant"]).reset_index(drop=True)
