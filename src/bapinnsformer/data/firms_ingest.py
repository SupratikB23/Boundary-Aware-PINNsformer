"""FIRMS active-fire / FRP ingest — HELD-OUT ONLY.

PRD §4.3 `data/firms_ingest.py` + §3.1/§10 leakage guards.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  BANNER — READ BEFORE IMPORTING                                    !!
!!  This module handles NASA FIRMS VIIRS/MODIS fire data which is     !!
!!  WITHHELD for E4 independent validation. It must NEVER be imported !!
!!  by anything under `models/`, `physics/` or `train/`, and it must  !!
!!  NEVER write outside `data/heldout/`. `tests/test_leakage.py`      !!
!!  enforces this structurally (import-graph assertion + path scan).  !!
!!  Training code reading `data/heldout/` is a BLOCKING BUG (PRD §8.3)!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Functions here fetch/parse daily fire archives and aggregate Fire
Radiative Power (FRP) for later correlation against recovered NW
inflow. They are used only by `scripts/02_fetch_firms.py` and
`scripts/40_validate_fires.py`. VIIRS/MODIS active-fire + FRP
retrieval (held-out validation only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HELDOUT_DIRNAME = "heldout"
HELDOUT_SUBDIR = "data/heldout/firms"

# Expected FIRMS columns (VIIRS VNP14IMGT / MODIS MCD14ML compatible subset).
FIRMS_COLUMNS = (
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_t31",
    "frp",
    "daynight",
)


def _assert_heldout_path(path: str | Path) -> Path:
    """Enforce that all FIRMS writes land under a `heldout/` directory."""
    p = Path(path)
    if HELDOUT_DIRNAME not in p.parts:
        raise PermissionError(
            f"FIRMS data may only be written under a '{HELDOUT_DIRNAME}/' directory; "
            f"refused: {p}. This guard protects E4 validity (PRD §10)."
        )
    return p


def parse_firms_csv(csv_path: str | Path) -> pd.DataFrame:
    """Parse a FIRMS VIIRS/MODIS CSV export to a canonical frame.

    Returns columns ``[time, latitude, longitude, frp, confidence,
    satellite, instrument, daynight]`` with ``time`` as UTC-naive
    hourly-truncated stamps derived from ``acq_date`` + ``acq_time``.
    Rows with bad coordinates are dropped; missing FRP → 0.0 is NOT
    imputation (absence of detection = no observed fire).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"FIRMS CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ("latitude", "longitude", "acq_date", "acq_time"):
        if col not in df.columns:
            raise ValueError(f"parse_firms_csv: required column '{col}' missing in {csv_path.name}")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    tstr = df["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    df["time"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " + tstr.str[:2] + ":" + tstr.str[2:],
        errors="coerce",
    )
    df = df.dropna(subset=["time"])
    if "frp" in df.columns:
        df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df["frp"] = 0.0
    for col in ("confidence", "satellite", "instrument", "daynight"):
        if col not in df.columns:
            df[col] = "unknown"
    keep = ["time", "latitude", "longitude", "frp", "confidence", "satellite", "instrument", "daynight"]
    return df[keep].sort_values("time").reset_index(drop=True)


def aggregate_frp_daily(
    df: pd.DataFrame,
    box: tuple[float, float, float, float] | None = None,
    freq: str = "1D",
) -> pd.DataFrame:
    """Aggregate FRP over a region to a regular time series.

    Args:
        df: output of :func:`parse_firms_csv`.
        box: optional (lon_min, lon_max, lat_min, lat_max) crop;
            e.g. Punjab/Haryana upwind sector for E4.
        freq: resample rule (default daily).

    Returns:
        DataFrame ``[time, fire_count, frp_sum, frp_mean, frp_max]``.
    """
    if df.empty:
        return pd.DataFrame(columns=["time", "fire_count", "frp_sum", "frp_mean", "frp_max"])
    work = df
    if box is not None:
        lon_min, lon_max, lat_min, lat_max = box
        work = work[
            (work["longitude"] >= lon_min)
            & (work["longitude"] <= lon_max)
            & (work["latitude"] >= lat_min)
            & (work["latitude"] <= lat_max)
        ]
    g = work.set_index("time").sort_index()
    agg = g["frp"].resample(freq).agg(["count", "sum", "mean", "max"])
    agg.columns = ["fire_count", "frp_sum", "frp_mean", "frp_max"]
    return agg.reset_index()


def save_heldout(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a held-out FIRMS frame to parquet/CSV under `heldout/` only."""
    out = _assert_heldout_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    return out


def fetch_firms_archive(
    url_or_path: str | Path,
    dest: str | Path,
    box: tuple[float, float, float, float] | None = None,
) -> Path:
    """Stage a FIRMS archive into `data/heldout/` (no training import).

    If ``url_or_path`` is a local CSV it is parsed + optionally cropped
    and written to ``dest`` (which must contain ``heldout/``). Remote
    HTTP(S) download uses stdlib ``urllib`` so no new dependency is
    required. Returns the destination path.
    """
    import urllib.request

    dest_p = _assert_heldout_path(dest)
    src = str(url_or_path)
    tmp = dest_p.with_suffix(".tmp_download.csv")
    if src.startswith("http://") or src.startswith("https://"):
        tmp.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(src, str(tmp))  # noqa: S310 — trusted NASA FIRMS endpoint
        df = parse_firms_csv(tmp)
        tmp.unlink(missing_ok=True)
    else:
        df = parse_firms_csv(Path(src))
    if box is not None:
        lon_min, lon_max, lat_min, lat_max = box
        df = df[
            (df["longitude"] >= lon_min)
            & (df["longitude"] <= lon_max)
            & (df["latitude"] >= lat_min)
            & (df["latitude"] <= lat_max)
        ].reset_index(drop=True)
    return save_heldout(df, dest_p)


__all__ = [
    "HELDOUT_DIRNAME",
    "HELDOUT_SUBDIR",
    "FIRMS_COLUMNS",
    "parse_firms_csv",
    "aggregate_frp_daily",
    "save_heldout",
    "fetch_firms_archive",
]
