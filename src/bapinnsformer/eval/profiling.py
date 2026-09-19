"""Efficiency profiling (PRD §4.3 `eval/profiling.py`, E3).

Wall-clock, peak memory and parameter counts per pseudo-sequence
variant and sequence length ``L``. Stdlib + lazy torch; GPU memory
only queried when CUDA is available.
"""

from __future__ import annotations

import time
import tracemalloc


class Timer:
    """Context-manager wall-clock timer (seconds, ``.elapsed``)."""

    def __init__(self) -> None:
        self.elapsed: float = float("nan")
        self._t0: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0


def time_fn(fn, *args, repeat: int = 3, **kwargs) -> dict:
    """Run ``fn(*args, **kwargs)`` ``repeat`` times; timing stats."""
    times: list[float] = []
    out = None
    for _ in range(max(1, int(repeat))):
        with Timer() as t:
            out = fn(*args, **kwargs)
        times.append(t.elapsed)
    times_sorted = sorted(times)
    mid = times_sorted[len(times_sorted) // 2]
    return {
        "mean_s": sum(times) / len(times),
        "median_s": mid,
        "min_s": times_sorted[0],
        "max_s": times_sorted[-1],
        "repeat": len(times),
        "output": out,
    }


def count_parameters(model) -> dict[str, int]:
    """Count trainable/total parameters of a torch module (lazy torch)."""
    try:
        import torch

        if isinstance(model, torch.nn.Module):
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {"total": int(total), "trainable": int(trainable)}
    except ImportError:
        pass
    # Generic fallback: object with named_parameters / param list
    try:
        params = list(model.parameters())
        total = sum(int(getattr(p, "numel", lambda: 1)()) for p in params)
        trainable = sum(
            int(getattr(p, "numel", lambda: 1)())
            for p in params
            if getattr(p, "requires_grad", True)
        )
        return {"total": total, "trainable": trainable}
    except Exception:
        return {"total": 0, "trainable": 0}


def peak_memory_mb() -> dict[str, float]:
    """Current peak memory snapshot (CUDA when available, else RSS).

    POSIX uses ``resource.ru_maxrss``; on Windows (no ``resource``) falls
    back to ``psutil`` when installed, else ``tracemalloc`` peak (which
    only tracks Python allocations — documented in the value's absence).
    Always returns a dict; missing backends simply leave keys out.
    """
    info: dict[str, float] = {}
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_peak_mb"] = float(
                torch.cuda.max_memory_allocated() / (1024**2)
            )
            info["cuda_current_mb"] = float(
                torch.cuda.memory_allocated() / (1024**2)
            )
    except ImportError:
        pass
    try:
        import resource

        info["cpu_rss_peak_mb"] = float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        )
        return info
    except Exception:
        pass
    try:  # Windows fallback 1: psutil RSS
        import psutil  # type: ignore

        info["cpu_rss_peak_mb"] = float(
            psutil.Process().memory_info().rss / (1024**2)
        )
        return info
    except Exception:
        pass
    try:  # Windows fallback 2: tracemalloc (Python allocs only)
        import tracemalloc

        if tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            info["tracemalloc_peak_mb"] = float(peak / (1024**2))
    except Exception:
        pass
    return info


class GpuPeakTracker:
    """Track CUDA peak inside a block (resets stats on entry)."""

    def __init__(self) -> None:
        self.peak_mb: float = float("nan")

    def __enter__(self) -> "GpuPeakTracker":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._t0 = tracemalloc.get_traced_memory()[0]
        return self

    def __exit__(self, *exc) -> None:
        current, peak = tracemalloc.get_traced_memory()
        self.peak_mb = float(max(peak, 0) / (1024**2))
        try:
            import torch

            if torch.cuda.is_available():
                self.peak_mb = float(
                    torch.cuda.max_memory_allocated() / (1024**2)
                )
        except ImportError:
            pass


def profile_record(variant: str, L: int, stats: dict, params: dict) -> dict:
    """Assemble one tidy E3 profiling row (variant × L)."""
    return {
        "variant": str(variant),
        "L": int(L),
        "wall_s": float(stats.get("median_s", stats.get("mean_s", float("nan")))),
        "peak_mb": float(stats.get("peak_mb", float("nan"))),
        "n_params": int(params.get("trainable", params.get("total", 0))),
    }


__all__ = [
    "Timer",
    "time_fn",
    "count_parameters",
    "peak_memory_mb",
    "GpuPeakTracker",
    "profile_record",
]
