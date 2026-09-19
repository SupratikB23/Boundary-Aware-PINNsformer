"""Utilities package.

PRD §4.3 (`utils/`): seeding, structured logging, string registries,
I/O helpers and provenance capture. Dependency-light on import
(torch/numpy only); heavier formats (parquet/NetCDF) are imported
lazily inside the functions that need them.
"""

from __future__ import annotations

__all__: list[str] = [
    "seeding",
    "logging",
    "registry",
    "io",
    "provenance",
]
