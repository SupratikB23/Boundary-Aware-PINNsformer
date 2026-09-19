"""Pytest configuration: src layout, CPU-only, headless figures, no network."""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)

# Deterministic single-thread CPU numerics for the suite.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import torch

    torch.set_num_threads(1)
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


if torch is not None:  # hermetic suite: some tests set float64 globally
    import pytest

    @pytest.fixture(autouse=True)
    def _restore_default_dtype():
        yield
        torch.set_default_dtype(torch.float32)
