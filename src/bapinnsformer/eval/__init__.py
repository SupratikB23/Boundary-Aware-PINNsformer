"""Evaluation plane package.

PRD §4.3 (`eval/`): metrics, transboundary attribution accounting,
external comparison, profiling and identifiability analysis.
All modules are CPU-only, dependency-light (numpy + optional
torch/scipy) and operate on saved results — never on raw fire data.
"""

from __future__ import annotations

__all__: list[str] = [
    "metrics",
    "attribution",
    "compare_external",
    "profiling",
    "identifiability",
]
