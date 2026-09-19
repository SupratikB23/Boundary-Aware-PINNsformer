"""Optimizers: Adam with cosine schedule → L-BFGS refinement (PRD §4.3).

``build_adam`` owns the Adam + cosine-annealing pair; ``lbfgs_refine``
runs the closure-based L-BFGS loop with a divergence guard that restores
the best-seen parameters on NaN/inf or sustained loss increase.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch

__all__ = ["build_adam", "lbfgs_refine"]


def build_adam(
    params: Iterable[torch.Tensor],
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    T_max: int = 1000,
    eta_min: float = 0.0,
) -> tuple[torch.optim.Adam, torch.optim.lr_scheduler.CosineAnnealingLR]:
    """Adam + :class:`CosineAnnealingLR` (CPU-safe; caller steps scheduler)."""
    ps = list(params)
    if not ps:
        raise ValueError("build_adam got an empty parameter list")
    opt = torch.optim.Adam(ps, lr=float(lr), weight_decay=float(weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(T_max), eta_min=float(eta_min)
    )
    return opt, sched


def lbfgs_refine(
    closure: Callable[[], torch.Tensor],
    params: Iterable[torch.Tensor],
    max_iter: int = 200,
    lr: float = 1.0,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    divergence_patience: int = 10,
) -> dict:
    """Run L-BFGS to refine Adam-converged parameters.

    Parameters
    ----------
    closure:
        Zero-arg callable that zeroes grads, recomputes the loss,
        calls ``backward()`` and returns the loss tensor.
    params:
        Parameters to optimize (passed to ``torch.optim.LBFGS``).
    divergence_patience:
        Consecutive strictly-increasing steps tolerated before stopping
        and restoring the best state.

    Returns
    -------
    ``{"best_loss", "final_loss", "steps", "diverged", "restored"}``.
    The guard guarantees the model is left at the best-seen state, never
    at a diverged one.
    """
    ps = list(params)
    opt = torch.optim.LBFGS(
        ps,
        lr=float(lr),
        max_iter=int(max_iter),
        tolerance_grad=float(tolerance_grad),
        tolerance_change=float(tolerance_change),
        line_search_fn="strong_wolfe",
    )
    best_loss = float("inf")
    best_state = [p.detach().clone() for p in ps]
    final_loss = float("inf")
    diverged = False
    bad_streak = 0
    steps = 0

    def _guarded_closure():
        nonlocal best_loss, final_loss, bad_streak, diverged, steps, best_state
        opt.zero_grad()
        loss = closure()
        if not torch.isfinite(loss).all():
            diverged = True
            # Return a dummy finite value so LBFGS internals do not crash;
            # the outer loop restores best_state afterwards.
            return torch.zeros((), dtype=loss.dtype, device=loss.device)
        val = float(loss.detach().item())
        final_loss = val
        steps += 1
        if val < best_loss - 1e-12:
            best_loss = val
            best_state = [p.detach().clone() for p in ps]
            bad_streak = 0
        elif val > best_loss + 1e-12:
            bad_streak += 1
        loss.backward()
        return loss

    try:
        opt.step(_guarded_closure)
    except RuntimeError:
        diverged = True

    if bad_streak >= int(divergence_patience):
        diverged = True

    restored = False
    if diverged or final_loss > best_loss:
        with torch.no_grad():
            for p, b in zip(ps, best_state):
                p.copy_(b)
        restored = True
        final_loss = best_loss

    return {
        "best_loss": best_loss,
        "final_loss": final_loss,
        "steps": steps,
        "diverged": diverged,
        "restored": restored,
    }
