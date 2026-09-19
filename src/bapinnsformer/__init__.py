"""Boundary-Aware PINNsformer package root.

PRD §4.3 (`src/bapinnsformer/`): top-level package for the inverse
advection–diffusion–deposition solver.

Governing PDE: dC/dt + div(uC) = div(K grad C) + S - lam*C.

This module owns the version string and the lazy public exports.
Heavy subpackages (`data`, `models`, `physics`, `train`, `eval`)
are imported lazily by consumers to keep `import bapinnsformer`
lightweight and GPU-free.
"""

from __future__ import annotations

__version__: str = "0.1.0"

__all__ = ["__version__"]
