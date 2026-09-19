"""Gradient-norm loss balancing (PRD §§2.5, 4.3 `train/balancing.py`).

Learning-rate-annealing style: every ``update_every`` steps the term
weights ``w_*`` are recomputed from per-term gradient norms so no single
term (data / PDE / BC / reg) dominates. The weight trajectory is logged
for the diagnostic figure in the paper.
"""

from __future__ import annotations

from typing import Iterable

import torch

__all__ = ["GradNormBalancer", "grad_norm"]


def grad_norm(loss: torch.Tensor, params: Iterable[torch.Tensor]) -> float:
    """L2 norm of ``d(loss)/d(params)`` (zero for grad-free / zero losses)."""
    ps = [p for p in params if isinstance(p, torch.Tensor) and p.requires_grad]
    if not ps:
        return 0.0
    if not loss.requires_grad or loss.numel() == 0:
        return 0.0
    try:
        grads = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
    except RuntimeError:
        return 0.0
    total = 0.0
    for g in grads:
        if g is not None:
            total += float((g.detach() ** 2).sum().item())
    return total ** 0.5


class GradNormBalancer:
    """Periodic GradNorm-style reweighting.

    Parameters
    ----------
    term_names:
        e.g. ``["data", "pde", "bc", "reg"]``.
    alpha:
        Restoring-force exponent (0.16 is the literature default).
    update_every:
        Recompute period in steps.
    init_weights:
        Optional ``{term: w}``; defaults to 1.0 each.
    renormalize:
        Rescale so ``sum(w) == len(terms)`` after each update.
    """

    def __init__(
        self,
        term_names: list[str] | tuple[str, ...] = ("data", "pde", "bc", "reg"),
        alpha: float = 0.16,
        update_every: int = 100,
        init_weights: dict[str, float] | None = None,
        renormalize: bool = True,
    ) -> None:
        self.term_names = list(term_names)
        if not self.term_names:
            raise ValueError("term_names must be non-empty")
        self.alpha = float(alpha)
        self.update_every = int(update_every)
        self.renormalize = bool(renormalize)
        self.weights: dict[str, float] = {
            k: float(init_weights[k]) if init_weights and k in init_weights else 1.0
            for k in self.term_names
        }
        self.history: list[dict] = []  # {"step": int, "weights": {...}, "grad_norms": {...}}

    # ------------------------------------------------------------------
    def should_update(self, step: int) -> bool:
        return int(step) % max(1, self.update_every) == 0

    def get_weights(self) -> dict[str, float]:
        return dict(self.weights)

    def update(self, grad_norms: dict[str, float], step: int | None = None) -> dict[str, float]:
        """Recompute weights from per-term gradient norms.

        ``w_i <- w_i * (mean_grad / (g_i + eps))^alpha``; zero-norm terms
        keep their current weight (never divide by zero, never kill a term).
        """
        eps = 1e-12
        gs = {k: max(0.0, float(grad_norms.get(k, 0.0))) for k in self.term_names}
        positive = [g for g in gs.values() if g > eps]
        if not positive:
            self._log(step, gs)
            return self.get_weights()
        mean_g = sum(positive) / len(positive)
        for k in self.term_names:
            g = gs[k]
            if g <= eps:
                continue
            self.weights[k] *= (mean_g / (g + eps)) ** self.alpha
        if self.renormalize:
            s = sum(self.weights.values())
            if s > eps:
                n = len(self.term_names)
                for k in self.term_names:
                    self.weights[k] *= n / s
        self._log(step, gs)
        return self.get_weights()

    def balance_from_losses(
        self,
        losses: dict[str, torch.Tensor],
        params: Iterable[torch.Tensor],
        step: int | None = None,
    ) -> dict[str, float]:
        """Convenience: compute grad norms from ``losses`` then :meth:`update`."""
        norms = {k: grad_norm(losses[k], params) for k in self.term_names if k in losses}
        return self.update(norms, step=step)

    # ------------------------------------------------------------------
    def _log(self, step: int | None, grad_norms: dict[str, float]) -> None:
        self.history.append(
            {"step": step, "weights": self.get_weights(), "grad_norms": dict(grad_norms)}
        )

    def trajectory(self) -> list[dict]:
        """Logged ``[{step, weights, grad_norms}]`` for the diagnostic figure."""
        return [dict(h) for h in self.history]
