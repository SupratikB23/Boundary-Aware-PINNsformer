"""CPCB CCR + OpenCity ingest → tidy long table.

PRD §4.3 `data/cpcb_ingest.py`: fetch/parse CPCB CCR exports and the
OpenCity bulk CSV archive; normalize station IDs, names, coordinates;
harmonize column naming across vintages; emit a tidy long-format table
``(station_id, time, pollutant, value, source)``.

Masked-loss contract: this module never imputes; missing values are
preserved as NaN for downstream masking in ``qc.py`` / ``losses.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical pollutants (upper-case, underscore). Anything else passes
# through unchanged but upper-cased so new vintages do not crash.
KNOWN_POLLUTANTS: tuple[str, ...] = (
    "PM2.5",
    "PM10",
    "PM1",
    "NO2",
    "NO",
    "NOX",
    "SO2",
    "CO",
    "O3",
    "NH3",
    "BENZENE",
    "TOLUENE",
    "WS",
    "WD",
    "TEMP",
    "RH",
    "AT",
    "BP",
)

# Alias → canonical column name (all keys lower-case, stripped).
_COLUMN_ALIASES: dict[str, str] = {
    "station": "station_id",
    "station id": "station_id",
    "station_id": "station_id",
    "stationid": "station_id",
    "site": "station_id",
    "location": "station_id",
    "monitoring station": "station_id",
    "station name": "station_name",
    "station_name": "station_name",
    "site name": "station_name",
    "lat": "latitude",
    "lattitude": "latitude",
    "latitude": "latitude",
    "lon": "longitude",
    "long": "longitude",
    "lng": "longitude",
    "longitude": "longitude",
    "date": "time",
    "datetime": "time",
    "timestamp": "time",
    "sampling date": "time",
    "from date": "time",
    "from_date": "time",
    "date_time": "time",
    "time": "time",
}

_WS_PAT = re.compile(r"\s+")


def normalize_station_id(raw: str | float | None) -> str:
    """Normalize a raw station identifier to a stable slug.

    Lower-cases, strips, collapses whitespace to single underscores,
    removes non-alphanumeric/underscore characters. Empty input maps
    to ``"unknown"`` so merges never produce NaN keys.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "unknown"
    s = str(raw).strip().lower()
    s = _WS_PAT.sub("_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename vintage-dependent columns to canonical names.

    Canonical columns: ``station_id``, ``station_name``, ``latitude``,
    ``longitude``, ``time``. Pollutant columns are upper-cased
    (``pm2.5`` → ``PM2.5``). Unrecognised columns are kept as-is
    (stripped) so no data is silently dropped.
    """
    out = df.copy()
    renamed: dict[str, str] = {}
    for col in out.columns:
        key = str(col).strip().lower()
        if key in _COLUMN_ALIASES:
            renamed[col] = _COLUMN_ALIASES[key]
        else:
            # pollutant-like header? normalise case/whitespace
            canon = str(col).strip().upper().replace(" ", "")
            if canon in {p.replace(".", "").replace("_", "") for p in KNOWN_POLLUTANTS} or canon in KNOWN_POLLUTANTS:
                renamed[col] = canon
            else:
                renamed[col] = str(col).strip()
    out = out.rename(columns=renamed)
    # Fix common PM variants: PM2_5 / PM25 → PM2.5
    fixups = {"PM2_5": "PM2.5", "PM25": "PM2.5", "PM_2_5": "PM2.5", "PM1_0": "PM10"}
    out = out.rename(columns={c: fixups[c] for c in out.columns if c in fixups})
    return out


def _coerce_time(series: pd.Series) -> pd.Series:
    """Coerce mixed datetime strings to timezone-naive UTC-intended stamps.

    Day-first is tried only as a fallback; IST localization happens in
    ``qc.py``. Unparseable entries become NaT (never filled).
    """
    t = pd.to_datetime(series, errors="coerce", utc=False)
    if t.isna().all():
        t = pd.to_datetime(series, errors="coerce", dayfirst=True, utc=False)
    return t


def to_long_table(
    df: pd.DataFrame,
    source: str = "cpcb",
    id_cols: tuple[str, ...] = ("station_id", "station_name", "latitude", "longitude", "time"),
) -> pd.DataFrame:
    """Melt a wide harmonized frame to tidy long format.

    Returns columns ``[station_id, station_name, latitude, longitude,
    time, pollutant, value, source]`` with one row per
    station × time × pollutant. NaN values are retained (masking, not
    imputation). ``station_id`` is normalized via :func:`normalize_station_id`.
    """
    df = harmonize_columns(df)
    if "station_id" not in df.columns:
        raise ValueError("to_long_table: missing 'station_id' column after harmonization")
    if "time" not in df.columns:
        raise ValueError("to_long_table: missing 'time' column after harmonization")
    df = df.copy()
    df["station_id"] = df["station_id"].map(normalize_station_id)
    df["time"] = _coerce_time(df["time"])
    df = df.dropna(subset=["time"])
    keep_ids = [c for c in id_cols if c in df.columns]
    value_vars = [c for c in df.columns if c not in keep_ids]
    if not value_vars:
        raise ValueError("to_long_table: no pollutant/value columns found")
    long_df = df.melt(
        id_vars=keep_ids,
        value_vars=value_vars,
        var_name="pollutant",
        value_name="value",
    )
    long_df["pollutant"] = long_df["pollutant"].astype(str).str.strip().str.upper()
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["source"] = str(source)
    # canonical column order
    cols = ["station_id", "time", "pollutant", "value", "source"]
    for extra in ("station_name", "latitude", "longitude"):
        if extra in long_df.columns:
            cols.insert(1, extra)
    # order: station_id, [station_name, latitude, longitude], time, pollutant, value, source
    ordered = ["station_id"] + [c for c in ("station_name", "latitude", "longitude") if c in long_df.columns]
    ordered += ["time", "pollutant", "value", "source"]
    return long_df[ordered].sort_values(["station_id", "time", "pollutant"]).reset_index(drop=True)


def ingest_cpcb_dir(
    raw_dir: str | Path,
    pattern: str = "*.csv",
    source: str = "cpcb",
) -> pd.DataFrame:
    """Parse every CPCB CCR CSV under ``raw_dir`` to one long table.

    Files are read with the python engine to tolerate vintage header
    quirks; unreadable files are skipped with a warning row count of 0
    rather than aborting the whole ingest (provenance of skips is
    returned via attrs).
    """
    raw_dir = Path(raw_dir)
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    files = sorted(raw_dir.rglob(pattern)) if raw_dir.exists() else []
    for fp in files:
        try:
            wide = pd.read_csv(fp, engine="python")
        except Exception:
            skipped.append(str(fp))
            continue
        if wide.empty:
            continue
        try:
            frames.append(to_long_table(wide, source=source))
        except ValueError:
            skipped.append(str(fp))
            continue
    if not frames:
        out = pd.DataFrame(
            columns=["station_id", "time", "pollutant", "value", "source"]
        )
    else:
        out = pd.concat(frames, ignore_index=True)
    out.attrs["skipped_files"] = skipped
    out.attrs["source"] = source
    return out


def ingest_opencity_csv(
    csv_path: str | Path,
    source: str = "opencity",
    chunksize: int | None = None,
) -> pd.DataFrame:
    """Parse an OpenCity bulk CSV (wide) to the same long schema as CCR.

    Supports chunked reading for large archives; when ``chunksize`` is
    given, chunks are melted independently and concatenated.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"OpenCity CSV not found: {csv_path}")
    if chunksize is None:
        wide = pd.read_csv(csv_path, engine="python")
        return to_long_table(wide, source=source)
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, engine="python", chunksize=chunksize):
        parts.append(to_long_table(chunk, source=source))
    if not parts:
        return pd.DataFrame(columns=["station_id", "time", "pollutant", "value", "source"])
    return pd.concat(parts, ignore_index=True)
