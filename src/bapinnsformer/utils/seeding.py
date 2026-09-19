"""Global seed control (PRD §4.3 `utils/seeding.py`, §10).

Single entry point :func:`seed_all` seeds Python ``random``, NumPy
and PyTorch (CPU + CUDA) and optionally forces deterministic cuDNN
behaviour. Import-light: torch is imported lazily so CPU-only and
torch-free environments still work.
"""

from __future__ import annotations

import os
import random


def seed_all(seed: int = 0, deterministic: bool = True) -> int:
    """Seed every RNG used by the project. Returns the seed.

    Args:
        seed: Non-negative integer seed.
        deterministic: If True, set ``torch.backends.cudnn.deterministic``
            and disable the cuDNN benchmark autotuner. No-op when torch
            is unavailable.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is effectively required, but stay light
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass
    return seed


def torch_generator(seed: int, device: str = "cpu"):
    """Return a ``torch.Generator`` seeded with ``seed`` (lazy torch)."""
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen
