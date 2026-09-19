"""Per-instance fit loop (PRD §4.3 `train/trainer.py`).

``Trainer`` assembles ``C_net`` (PINNsformer) + ``Cb_net`` + ``S_net`` +
``(K, lam)`` for one ``(city, season, pollutant)`` instance and runs
``Adam(cosine) → [optional L-BFGS]`` under curriculum + GradNorm balancing.

Batch format (all physical units, 1-D tensors of shape ``(B,)`` unless noted)
-----------------------------------------------------------------------------
``batch`` is a plain dict; every key optional except the data block:

* Data: ``x_data, y_data, t_data, C_obs, mask`` (mask bool/float, True = valid).
* PDE collocation: ``x_pde, y_pde, t_pde``.
* Boundary: ``x_bc, y_bc, s_bc, t_bc, nx, ny`` (outward normals, unit).
* Optional ``hysteresis`` override for the ``u·n`` switch.

PDE residual (autodiff, PRD §1.1)
---------------------------------
``r = dC/dt + u·∇C + C·div(u) − K·ΔC − S + lam·C`` where ``C`` flows through
the pseudo-sequence generator (differentiable RK path) + ``C_net``.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from ..models.baselines import MLPPINN
from ..models.boundary_net import CbNet
from ..models.params import PhysParams
from ..models.pinnsformer import PINNsformer
from ..models.pseudoseq import PseudoSequenceGenerator, create_generator
from .balancing import GradNormBalancer
from .curriculum import Curriculum
from .losses import (
    bc_loss,
    cb_temporal_smooth,
    masked_mse,
    pde_loss,
    source_l1,
)
from .optim import build_adam, lbfgs_refine

__all__ = ["Trainer", "default_config"]


def default_config() -> dict[str, Any]:
    """Minimal CPU-safe default for one instance fit."""
    return {
        "city": "delhi_ncr",
        "season": "postmonsoon",  # canonical label (utils.config.normalize_season)
        "pollutant": "PM2.5",
        "seed": 0,
        "device": "cpu",
        "model": PINNsformer.default_config(),
        "cb_net": {"hidden_dim": 32, "num_layers": 2, "activation": "wavelet"},
        "s_net": {"hidden_dim": 32, "num_layers": 2, "activation": "wavelet"},
        "phys": {"K_init": 100.0, "lam_init": 1e-5},
        "pseudoseq": {
            "name": "advection_backward",
            "seq_len": 5,
            "dt": 3600.0,
            "integrator": "rk2",
            "substeps": 1,
            "ood_policy": "clamp",
        },
        "train": {
            "steps": 50,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "w_data": 1.0,
            "w_pde": 1.0,
            "w_bc": 1.0,
            "w_reg": 1e-3,
            "balance_every": 0,  # 0 = fixed weights; >0 enables GradNorm period
            "grad_clip": 1.0,
            "hysteresis": 1e-3,
            "lbfgs_refine": False,
            "lbfgs_max_iter": 50,
            "checkpoint_every": 0,
        },
        "curriculum": None,  # or {"t0":.., "t1":.., "total_steps":.., ...}
    }


class Trainer:
    """One ``(city, season, pollutant)`` fit (PRD §4.3)."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        normalizer: Any = None,
        wind_field: Any = None,
        C_net: nn.Module | None = None,
        Cb_net: nn.Module | None = None,
        S_net: nn.Module | None = None,
        phys_params: PhysParams | None = None,
        generator: PseudoSequenceGenerator | None = None,
        device: str | torch.device | None = None,
        checkpoint_hook: Callable[[int, dict], None] | None = None,
    ) -> None:
        self.config = config or default_config()
        self.normalizer = normalizer
        # Default to quiescent (zero) wind so the generator + BC paths work
        # out of the box; pass a real WindField for science runs.
        self.wind_field = wind_field if wind_field is not None else (0.0, 0.0)
        self.checkpoint_hook = checkpoint_hook

        seed = int(self.config.get("seed", 0))
        torch.manual_seed(seed)

        from ..utils.io import resolve_device

        dev = resolve_device(device or self.config.get("device", "cpu"))
        self.device = torch.device(dev)

        mcfg = self.config.get("model", {})
        cbcfg = self.config.get("cb_net", {})
        scfg = self.config.get("s_net", {})
        pcfg = self.config.get("phys", {})
        gcfg = dict(self.config.get("pseudoseq", {}))

        if self.normalizer is not None and hasattr(self.normalizer, "x_bounds"):
            gcfg.setdefault("x_bounds", self.normalizer.x_bounds)
            gcfg.setdefault("y_bounds", self.normalizer.y_bounds)

        self.generator: PseudoSequenceGenerator
        if generator is not None:
            self.generator = generator
        else:
            gname = gcfg.pop("name", "advection_backward")
            self.generator = create_generator(gname, **gcfg)

        self.C_net: nn.Module = C_net or PINNsformer(**mcfg)
        # Allow the MLP-PINN control to be selected by config string.
        if isinstance(self.C_net, str):  # pragma: no cover - defensive
            raise TypeError("C_net must be a module, not a string")
        self.Cb_net: nn.Module = Cb_net or CbNet(**cbcfg)
        # S_net import is local to avoid circulars in tiny test envs.
        if S_net is None:
            from ..models.source_net import SNet

            S_net = SNet(**scfg)
        self.S_net: nn.Module = S_net
        self.phys: PhysParams = phys_params or PhysParams(**pcfg)

        for m in (self.C_net, self.Cb_net, self.S_net, self.phys):
            m.to(self.device)
        # Adopt the nets' dtype (users may construct float64 nets for PINN
        # precision or inherit a global float64 default); all batch tensors
        # below are cast to it instead of hardcoded float32.
        try:
            self.dtype = next(self.C_net.parameters()).dtype
        except StopIteration:  # pragma: no cover - parameterless net
            import torch as _t

            self.dtype = _t.float32

        tcfg = self.config.get("train", {})
        self.weights = {
            "data": float(tcfg.get("w_data", 1.0)),
            "pde": float(tcfg.get("w_pde", 1.0)),
            "bc": float(tcfg.get("w_bc", 1.0)),
            "reg": float(tcfg.get("w_reg", 1e-3)),
        }
        self.balance_every = int(tcfg.get("balance_every", 0))
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.hysteresis = float(tcfg.get("hysteresis", 1e-3))

        self.balancer: GradNormBalancer | None = None
        if self.balance_every > 0:
            self.balancer = GradNormBalancer(
                term_names=["data", "pde", "bc", "reg"],
                update_every=self.balance_every,
                init_weights=dict(self.weights),
            )

        self.curriculum: Curriculum | None = None
        ccfg = self.config.get("curriculum")
        if ccfg:
            self.curriculum = Curriculum(**ccfg)

        params = list(self.parameters())
        self.optimizer, self.scheduler = build_adam(
            params,
            lr=float(tcfg.get("lr", 1e-3)),
            weight_decay=float(tcfg.get("weight_decay", 0.0)),
            T_max=int(tcfg.get("steps", 50)),
        )
        self.history: list[dict[str, Any]] = []
        self._step = 0

    # ------------------------------------------------------------------
    def parameters(self):
        for m in (self.C_net, self.Cb_net, self.S_net, self.phys):
            yield from m.parameters()

    def train_mode(self):
        for m in (self.C_net, self.Cb_net, self.S_net, self.phys):
            m.train()

    def eval_mode(self):
        for m in (self.C_net, self.Cb_net, self.S_net, self.phys):
            m.eval()

    # ------------------------------------------------------------------
    # prediction helpers (physical in, physical out)
    # ------------------------------------------------------------------
    def _normalize_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.normalizer is None:
            return tokens
        x = self.normalizer.normalize_x(tokens[..., 0])
        y = self.normalizer.normalize_y(tokens[..., 1])
        t = self.normalizer.normalize_t(tokens[..., 2])
        return torch.stack([x, y, t], dim=-1)

    def predict_C(self, x, y, t) -> torch.Tensor:
        """Concentration ``C (B,)`` at physical ``(x, y, t)`` via generator + ``C_net``."""
        bx = torch.as_tensor(x, dtype=self.dtype, device=self.device).reshape(-1)
        by = torch.as_tensor(y, dtype=self.dtype, device=self.device).reshape(-1)
        bt = torch.as_tensor(t, dtype=self.dtype, device=self.device).reshape(-1)
        tokens, _ = self.generator.generate(bx, by, bt, self.wind_field)
        tokens = tokens.to(self.device)
        Tn = self._normalize_tokens(tokens)
        C = self.C_net(Tn).reshape(-1)
        return C

    def predict_Cb(self, s, t) -> torch.Tensor:
        s = torch.as_tensor(s, dtype=self.dtype, device=self.device).reshape(-1)
        t = torch.as_tensor(t, dtype=self.dtype, device=self.device).reshape(-1)
        if self.normalizer is not None and getattr(self.normalizer, "s_bounds", None) is not None:
            sh = self.normalizer.normalize_s(s)
        else:
            sh = s
        th = self.normalizer.normalize_t(t) if self.normalizer is not None else t
        Cb = self.Cb_net(sh, th)
        if isinstance(Cb, torch.Tensor):
            return Cb.reshape(-1)
        return torch.as_tensor(Cb, dtype=self.dtype, device=self.device).reshape(-1)

    def predict_S(self, x, y, t) -> torch.Tensor:
        bx = torch.as_tensor(x, dtype=self.dtype, device=self.device).reshape(-1)
        by = torch.as_tensor(y, dtype=self.dtype, device=self.device).reshape(-1)
        bt = torch.as_tensor(t, dtype=self.dtype, device=self.device).reshape(-1)
        if self.normalizer is not None:
            xh = self.normalizer.normalize_x(bx)
            yh = self.normalizer.normalize_y(by)
            th = self.normalizer.normalize_t(bt)
        else:
            xh, yh, th = bx, by, bt
        S = self.S_net(xh, yh, th).reshape(-1)
        return S

    def _wind_at(self, x, y, t) -> tuple[torch.Tensor, torch.Tensor]:
        wf = self.wind_field
        if wf is None:
            return torch.zeros_like(x), torch.zeros_like(y)
        if hasattr(wf, "sample"):
            u, v = wf.sample(x, y, t)
        elif callable(wf):
            u, v = wf(x, y, t)
        else:  # constant (u, v) pair, e.g. the default quiescent (0.0, 0.0)
            u, v = wf
            u = torch.full_like(x, float(u))
            v = torch.full_like(y, float(v))
        u = torch.as_tensor(u, dtype=self.dtype, device=self.device).reshape_as(x)
        v = torch.as_tensor(v, dtype=self.dtype, device=self.device).reshape_as(y)
        return u, v

    # ------------------------------------------------------------------
    def pde_residual(self, x, y, t) -> torch.Tensor:
        """Autodiff PDE residual at physical ``(x, y, t)`` (grad-carrying)."""
        xq = torch.as_tensor(x, dtype=self.dtype, device=self.device).reshape(-1).requires_grad_(True)
        yq = torch.as_tensor(y, dtype=self.dtype, device=self.device).reshape(-1).requires_grad_(True)
        tq = torch.as_tensor(t, dtype=self.dtype, device=self.device).reshape(-1).requires_grad_(True)

        tokens, _ = self.generator.generate(xq, yq, tq, self.wind_field)
        Tn = self._normalize_tokens(tokens)
        C = self.C_net(Tn).reshape(-1)

        grads = torch.autograd.grad(C.sum(), (xq, yq, tq), create_graph=True, retain_graph=True)
        dCdx, dCdy, dCdt = grads
        d2x = torch.autograd.grad(dCdx.sum(), xq, create_graph=True, retain_graph=True)[0]
        d2y = torch.autograd.grad(dCdy.sum(), yq, create_graph=True, retain_graph=True)[0]
        lap = d2x + d2y

        u, v = self._wind_at(xq, yq, tq)
        # div(u): zero for uniform/prescribed wind unless the field is
        # itself a differentiable torch function of (x, y).
        try:
            dudx = torch.autograd.grad(u.sum(), xq, create_graph=True, retain_graph=True,
                                       allow_unused=True)[0]
            dvdy = torch.autograd.grad(v.sum(), yq, create_graph=True, retain_graph=True,
                                       allow_unused=True)[0]
            div_u = torch.zeros_like(C)
            if dudx is not None:
                div_u = div_u + dudx
            if dvdy is not None:
                div_u = div_u + dvdy
        except RuntimeError:
            div_u = torch.zeros_like(C)

        S = self.predict_S(xq, yq, tq)
        K, lam = self.phys.forward()
        # Physical scales: derivatives above are w.r.t. physical x/y/t only if
        # the generator + normalizer chain preserves physical units through
        # the affine map — the affine Jacobian is folded in here explicitly.
        if self.normalizer is not None:
            sx = 2.0 / (self.normalizer.x_bounds[1] - self.normalizer.x_bounds[0])
            sy = 2.0 / (self.normalizer.y_bounds[1] - self.normalizer.y_bounds[0])
            st = 2.0 / (self.normalizer.t_bounds[1] - self.normalizer.t_bounds[0])
            # NOTE: Tn = s_*x + b, and C_net sees Tn; dC/dx_phys already
            # includes the chain rule via autograd (Tn depends on x through
            # the affine map), so no extra rescaling is applied here. The
            # scales are documented for the physics/operators.py reference.
            _ = (sx, sy, st)
        r = dCdt + u * dCdx + v * dCdy + C * div_u - K * lap - S + lam * C
        return r

    def boundary_terms(self, x_b, y_b, s_b, t_b, nx, ny):
        """``(C_pred, Cb_pred, dCdn, undotn)`` at boundary points."""
        xb = torch.as_tensor(x_b, dtype=self.dtype, device=self.device).reshape(-1).requires_grad_(True)
        yb = torch.as_tensor(y_b, dtype=self.dtype, device=self.device).reshape(-1).requires_grad_(True)
        tb = torch.as_tensor(t_b, dtype=self.dtype, device=self.device).reshape(-1)
        sb = torch.as_tensor(s_b, dtype=self.dtype, device=self.device).reshape(-1)
        nxx = torch.as_tensor(nx, dtype=self.dtype, device=self.device).reshape(-1)
        nyy = torch.as_tensor(ny, dtype=self.dtype, device=self.device).reshape(-1)

        tokens, _ = self.generator.generate(xb, yb, tb, self.wind_field)
        Tn = self._normalize_tokens(tokens)
        C = self.C_net(Tn).reshape(-1)
        dCdx = torch.autograd.grad(C.sum(), xb, create_graph=True, retain_graph=True)[0]
        dCdy = torch.autograd.grad(C.sum(), yb, create_graph=True, retain_graph=True)[0]
        dCdn = dCdx * nxx + dCdy * nyy

        if self.normalizer is not None and getattr(self.normalizer, "s_bounds", None) is not None:
            sh = self.normalizer.normalize_s(sb.detach())
        else:
            sh = sb.detach()
        th = self.normalizer.normalize_t(tb.detach()) if self.normalizer is not None else tb.detach()
        Cb = self.Cb_net(sh, th).reshape(-1)

        u, v = self._wind_at(xb.detach(), yb.detach(), tb.detach())
        undotn = u * nxx + v * nyy
        return C, Cb, dCdn, undotn

    # ------------------------------------------------------------------
    def compute_losses(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Unweighted term losses ``{data, pde, bc, reg}`` (each scalar)."""
        dev = self.device
        # ---- data ----
        xd = torch.as_tensor(batch["x_data"], dtype=self.dtype, device=dev).reshape(-1)
        yd = torch.as_tensor(batch["y_data"], dtype=self.dtype, device=dev).reshape(-1)
        td = torch.as_tensor(batch["t_data"], dtype=self.dtype, device=dev).reshape(-1)
        obs = torch.as_tensor(batch["C_obs"], dtype=self.dtype, device=dev).reshape(-1)
        m = batch.get("mask", torch.ones_like(obs))
        m = torch.as_tensor(m, device=dev).reshape(-1)
        if self.curriculum is not None:
            cw = self.curriculum.in_window(td.detach(), self._step).to(dtype=self.dtype)
            m = m.to(dtype=self.dtype) * cw
        C_pred = self.predict_C(xd, yd, td)
        L_data = masked_mse(C_pred, obs, m)

        # ---- pde ----
        if "x_pde" in batch:
            xp = torch.as_tensor(batch["x_pde"], dtype=self.dtype, device=dev).reshape(-1)
            yp = torch.as_tensor(batch["y_pde"], dtype=self.dtype, device=dev).reshape(-1)
            tp = torch.as_tensor(batch["t_pde"], dtype=self.dtype, device=dev).reshape(-1)
            if self.curriculum is not None and tp.numel():
                keep = self.curriculum.in_window(tp.detach(), self._step)
                xp, yp, tp = xp[keep], yp[keep], tp[keep]
            r = self.pde_residual(xp, yp, tp) if xp.numel() else torch.zeros((), device=dev)
            L_pde = pde_loss(r)
            S_col = self.predict_S(xp.detach(), yp.detach(), tp.detach()) if xp.numel() else torch.zeros((), device=dev)
        else:
            L_pde = torch.zeros((), device=dev)
            S_col = torch.zeros((), device=dev)

        # ---- bc ----
        if "x_bc" in batch:
            C_b, Cb_b, dCdn, undotn = self.boundary_terms(
                batch["x_bc"], batch["y_bc"], batch["s_bc"],
                batch["t_bc"], batch["nx"], batch["ny"],
            )
            hyst = float(batch.get("hysteresis", self.hysteresis))
            # Match physics/boundary.py::partition exactly: inflow q < -h,
            # outflow q > +h; the hysteresis buffer (|q| <= h) contributes
            # to neither side (no unphysical Neumann on tangent flow).
            inflow = undotn < -hyst
            outflow = undotn > hyst
            L_bc = bc_loss(C_b, Cb_b, dCdn, inflow, outflow)
            Cb_seq = Cb_b
        else:
            L_bc = torch.zeros((), device=dev)
            Cb_seq = torch.zeros((), device=dev)

        # ---- reg ----
        L_reg = source_l1(S_col) + 0.1 * cb_temporal_smooth(Cb_seq, order=1)
        return {"data": L_data, "pde": L_pde, "bc": L_bc, "reg": L_reg}

    # ------------------------------------------------------------------
    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Single Adam step; returns detached ``{term: value, total}``."""
        self.train_mode()
        losses = self.compute_losses(batch)

        if self.balancer is not None and self.balancer.should_update(self._step):
            with torch.enable_grad():
                norms = {}
                for k, L in losses.items():
                    try:
                        g = torch.autograd.grad(
                            L, list(self.C_net.parameters()),
                            retain_graph=True, allow_unused=True,
                        )
                        tot = sum(float((gg.detach() ** 2).sum()) for gg in g if gg is not None)
                        norms[k] = tot ** 0.5
                    except RuntimeError:
                        norms[k] = 0.0
                self.weights = self.balancer.update(norms, step=self._step)

        total = sum(self.weights[k] * losses[k] for k in losses)
        self.optimizer.zero_grad()
        if torch.isfinite(total):
            total.backward()
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.grad_clip)
            self.optimizer.step()
        self.scheduler.step()
        self._step += 1

        out = {k: float(v.detach().item()) for k, v in losses.items()}
        out["total"] = float(total.detach().item()) if torch.isfinite(total) else float("nan")
        out["step"] = self._step
        out["weights"] = dict(self.weights)
        self.history.append(out)
        return out

    # ------------------------------------------------------------------
    def fit(
        self,
        batch: dict[str, Any] | Callable[[int], dict[str, Any]],
        steps: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run the fit loop.

        ``batch`` is either a fixed dict reused each step or a callable
        ``step -> batch`` (resamples collocation points per step).
        Applies curriculum + balancing + ``checkpoint_hook``.
        """
        tcfg = self.config.get("train", {})
        n = int(steps if steps is not None else tcfg.get("steps", 50))
        ckpt_every = int(tcfg.get("checkpoint_every", 0))
        for i in range(n):
            b = batch(i) if callable(batch) else batch
            info = self.train_step(b)
            if ckpt_every > 0 and self.checkpoint_hook is not None and (i + 1) % ckpt_every == 0:
                self.checkpoint_hook(i + 1, self.state())
        if bool(tcfg.get("lbfgs_refine", False)):
            self.lbfgs_stage(batch if not callable(batch) else batch(0))
        return self.history

    def lbfgs_stage(self, batch: dict[str, Any]) -> dict:
        """Optional L-BFGS refinement with divergence guard."""
        tcfg = self.config.get("train", {})

        def closure():
            self.optimizer.zero_grad()
            losses = self.compute_losses(batch)
            total = sum(self.weights[k] * losses[k] for k in losses)
            total.backward()
            return total

        return lbfgs_refine(
            closure, list(self.parameters()),
            max_iter=int(tcfg.get("lbfgs_max_iter", 50)),
        )

    # ------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "weights": dict(self.weights),
            "history": list(self.history),
            "config": self.config,
        }
