"""Data plane package.

PRD §4.3 (`data/`): ingest (CPCB/ERA5/FIRMS-held-out) → QC/mask →
align to grid+clock → domain + splits → processed artifacts.

All modules here are CPU-only and must never impute missing
observations (masked loss downstream). FIRMS helpers write only
under `data/heldout/` and must never be imported by models/train.
"""

from __future__ import annotations

__all__: list[str] = [
    "cpcb_ingest",
    "era5_ingest",
    "firms_ingest",
    "qc",
    "align",
    "domain",
    "splits",
    "wind_field",
    "synthetic",
    "stats",
]
