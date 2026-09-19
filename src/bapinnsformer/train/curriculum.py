"""Temporal curriculum (PRD §§2.5, 4.3 `train/curriculum.py`).

Fit an expanding time window: the model sees ``[t0, t0 + Δ]`` before
``[t0, t0 + 2Δ]``, etc. Interacts with collocation sampling and early
stopping (``physics/collocation.py`` consumes :meth:`window_for_step`).
"""

from __future__ import annotations

import torch

__all__ = ["Curriculum"]


class Curriculum:
    """Expanding-window schedule over ``[t0, t1]``.

    Parameters
    ----------
    t0, t1:
        Window bounds in physical time units (must satisfy ``t0 < t1``).
    total_steps:
        Step at which the window first covers the full interval.
    initial_frac:
        Fraction of ``[t0, t1]`` visible at step 0 (in ``(0, 1]``).
    schedule:
        ``"linear"`` (default) or ``"exp"`` (concave, fast early growth).
    """

    def __init__(
        self,
        t0: float,
        t1: float,
        total_steps: int = 1000,
        initial_frac: float = 0.2,
        schedule: str = "linear",
    ) -> None:
        if not float(t0) < float(t1):
            raise ValueError(f"require t0 < t1, got {(t0, t1)}")
        if int(total_steps) <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 < float(initial_frac) <= 1.0:
            raise ValueError("initial_frac must be in (0, 1]")
        if schedule not in ("linear", "exp"):
            raise ValueError("schedule must be linear/exp")
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.total_steps = int(total_steps)
        self.initial_frac = float(initial_frac)
        self.schedule = schedule

    # ------------------------------------------------------------------
    def progress(self, step: int) -> float:
        """Fractional schedule progress in ``[0, 1]``."""
        return min(1.0, max(0.0, float(step) / float(self.total_steps)))

    def window_for_step(self, step: int) -> tuple[float, float]:
        """Return ``(t_start, t_end)`` visible at ``step`` (``t_start == t0``)."""
        p = self.progress(int(step))
        if self.schedule == "linear":
            frac = self.initial_frac + (1.0 - self.initial_frac) * p
        else:  # exp: concave — reaches ~full coverage faster
            import math

            frac = self.initial_frac + (1.0 - self.initial_frac) * (1.0 - math.exp(-3.0 * p)) / (
                1.0 - math.exp(-3.0)
            )
        t_end = self.t0 + frac * (self.t1 - self.t0)
        return (self.t0, min(self.t1, t_end))

    def in_window(self, t: torch.Tensor, step: int) -> torch.Tensor:
        """Bool mask of which times ``t`` are inside the current window."""
        _, t_end = self.window_for_step(int(step))
        t = torch.as_tensor(t, dtype=torch.float32)
        return t <= (t_end + 1e-9)

    def filter_batch(self, t: torch.Tensor, step: int) -> torch.Tensor:
        """Alias of :meth:`in_window` for call sites filtering batches."""
        return self.in_window(t, step)
