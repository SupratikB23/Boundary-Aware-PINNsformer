"""Per-instance fit loop (PRD §4.3 `train/trainer.py`, revised 2026-10-03).

``Trainer`` assembles ``C_net`` (PINNsformer) + ``Cb_net`` + ``S_net`` +
``(K, lam)`` for one ``(city, season, pollutant, window)`` instance and runs
``Adam(cosine) → [optional L-BFGS]`` under curriculum + GradNorm balancing.

Batch format (all physical units, 1-D tensors of shape ``(B,)`` unless noted)
-----------------------------------------------------------------------------
* Data: ``x_data, y_data, t_data, C_obs, mask`` (mask bool/float, True = valid).
* PDE collocation: ``x_pde, y_pde, t_pde``.
* Boundary: ``x_bc, y_bc, s_bc, t_bc, nx, ny`` (outward unit normals).
* Optional ``hysteresis`` override (m/s) for the ``u·n`` switch.

Scaling (fixes the unit imbalance found in the 2026-10-03 review)
-----------------------------------------------------------------
Raw residuals have wildly different magnitudes (data MSE ~10⁴ (µg/m³)²,
PDE residual ~10⁻⁴ (µg/m³/s)²), so with unit weights the physics was
numerically switched off. Every term is now made dimensionless with
reference scales ``C_ref`` (µg m⁻³), ``T_ref`` (s) and ``L_ref`` (m):

* nets output O(1) numbers; physical fields are
  ``C = C_ref·Ĉ``, ``C_b = C_ref·Ĉ_b``, ``S = (C_ref/T_ref)·Ŝ``;
* data:  ``(C − C_obs)/C_ref``;  PDE: ``r·T_ref/C_ref``;
  Dirichlet: ``(C − C_b)/C_ref``;  Neumann: ``∂C/∂n·L_ref/C_ref``;
  characteristic residual: ``r_char·T_ref/C_ref``.

``scales.mode: auto`` (default) sets ``C_ref`` = median of the first
batch's valid observations; ``T_ref`` = 3600 s and ``L_ref`` = 10 km unless
configured. ``mode: fixed`` uses the configured numbers (tests/back-compat).

PDE residual (autodiff)
-----------------------
advective (default):     r = ∂C/∂t + u·∇C − KΔC − S + λC
conservative (optional): r = ∂C/∂t + u·∇C + C ∇·u − KΔC − S + λC
The default is advective because interpolated surface winds are not
mass-consistent; their divergence is dominated by interpolation noise and
would act as a spurious source (RUN_PLAN §5). The E1 solver
(``physics/greens.py``) uses the same advective form, so E1 and E2 solve
the same equation.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from ..models.boundary_net import CbNet
from ..models.params import PhysParams
from ..models.pinnsformer import PINNsformer
from ..models.pseudoseq import (
    AdvectionBackwardGenerator,
    PseudoSequenceGenerator,
    create_generator,
)
from ..physics.characteristic import characteristic_residual
from .balancing import GradNormBalancer
from .curriculum import Curriculum
from .losses import bc_loss, masked_mse, pde_loss
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
        "physics": {"form": "advective"},
        "scales": {"mode": "auto", "C_ref": 1.0, "T_ref": 3600.0, "L_ref": 10_000.0},
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
            "w_char": 0.0,  # >0 enables the characteristic/FK loss (needs an advection generator)
            "char_n_mc": 1,  # SDE replicas per query (advection_jitter only)
            "reg_source_l1": 1.0,  # relative weight of mean|Ŝ| inside L_reg
            "reg_cb_smooth": 1.0,  # relative weight of mean (∂Ĉ_b/∂t̂)² inside L_reg
            "balance_every": 0,  # 0 = fixed weights; >0 enables GradNorm period
            "grad_clip": 1.0,
            "hysteresis": 1e-3,  # m/s
            "lbfgs_refine": False,
            "lbfgs_max_iter": 50,
            "checkpoint_every": 0,
        },
        "curriculum": None,  # or {"t0":.., "t1":.., "total_steps":.., ...}
    }


def _scaled(net: nn.Module) -> bool:
    """Nets flagged ``physical_output = True`` (baselines) are not rescaled."""
    return not bool(getattr(net, "physical_output", False))


class Trainer:
    """One ``(city, season, pollutant, window)`` fit (PRD §4.3)."""

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
        base = default_config()
        cfg = dict(config or base)
        for key in ("train", "scales", "physics"):
            merged = dict(base.get(key) or {})
            merged.update(cfg.get(key) or {})
            cfg[key] = merged
        self.config = cfg
        self.normalizer = normalizer
        # Default to quiescent (zero) wind so unit tests run; science runs
        # MUST pass a real WindField (scripts refuse to run without one).
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
        self.Cb_net: nn.Module = Cb_net or CbNet(**cbcfg)
        if S_net is None:
            from ..models.source_net import SNet

            S_net = SNet(**scfg)
        self.S_net: nn.Module = S_net
        self.phys: PhysParams = phys_params or PhysParams(**pcfg)

        for m in (self.C_net, self.Cb_net, self.S_net, self.phys):
            m.to(self.device)
        try:
            self.dtype = next(self.C_net.parameters()).dtype
        except StopIteration:  # pragma: no cover - parameterless net
            self.dtype = torch.float32

        # ---- physics form + scales -------------------------------------
        form = str(self.config["physics"].get("form", "advective")).lower()
        if form not in ("advective", "conservative"):
            raise ValueError(f"physics.form must be advective|conservative, got {form!r}")
        self.form = form
        sc = self.config["scales"]
        self.scales_mode = str(sc.get("mode", "auto")).lower()
        if self.scales_mode not in ("auto", "fixed"):
            raise ValueError("scales.mode must be auto|fixed")
        self.C_ref = float(sc.get("C_ref", 1.0))
        self.T_ref = float(sc.get("T_ref", 3600.0))
        self.L_ref = float(sc.get("L_ref", 10_000.0))
        if min(self.C_ref, self.T_ref, self.L_ref) <= 0:
            raise ValueError("scales must be positive")
        self._scales_locked = self.scales_mode == "fixed"

        tcfg = self.config["train"]
        self.weights = {
            "data": float(tcfg.get("w_data", 1.0)),
            "pde": float(tcfg.get("w_pde", 1.0)),
            "bc": float(tcfg.get("w_bc", 1.0)),
            "reg": float(tcfg.get("w_reg", 1e-3)),
        }
        self.use_char = float(tcfg.get("w_char", 0.0)) > 0.0
        if self.use_char:
            if not isinstance(self.generator, AdvectionBackwardGenerator):
                raise ValueError("w_char > 0 requires an advection_backward/advection_jitter generator")
            self.weights["char"] = float(tcfg["w_char"])
        self.char_n_mc = max(1, int(tcfg.get("char_n_mc", 1)))
        self.reg_source_l1 = float(tcfg.get("reg_source_l1", 1.0))
        self.reg_cb_smooth = float(tcfg.get("reg_cb_smooth", 1.0))
        self.balance_every = int(tcfg.get("balance_every", 0))
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.hysteresis = float(tcfg.get("hysteresis", 1e-3))

        self.balancer: GradNormBalancer | None = None
        if self.balance_every > 0:
            # Regularizers are priors, not competing objectives: their
            # weight stays fixed (balancing them would erase the prior).
            self.balancer = GradNormBalancer(
                term_names=[k for k in self.weights if k != "reg"],
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
        self.last_diagnostics: dict[str, float] = {}

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

    def set_scales(self, C_ref: float | None = None, T_ref: float | None = None,
                   L_ref: float | None = None) -> None:
        """Set reference scales explicitly and lock them."""
        if C_ref is not None:
            self.C_ref = float(C_ref)
        if T_ref is not None:
            self.T_ref = float(T_ref)
        if L_ref is not None:
            self.L_ref = float(L_ref)
        if min(self.C_ref, self.T_ref, self.L_ref) <= 0:
            raise ValueError("scales must be positive")
        self._scales_locked = True

    def scales(self) -> dict[str, float]:
        return {"C_ref": self.C_ref, "T_ref": self.T_ref, "L_ref": self.L_ref,
                "mode": self.scales_mode, "locked": self._scales_locked}

    def _maybe_auto_scale(self, obs: torch.Tensor, mask: torch.Tensor) -> None:
        if self._scales_locked:
            return
        valid = obs[(mask > 0) & torch.isfinite(obs)]
        if valid.numel() > 0:
            med = float(valid.abs().median().item())
            self.C_ref = max(med, 1e-6)
        self._scales_locked = True

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

    def _t(self, v) -> torch.Tensor:
        return torch.as_tensor(v, dtype=self.dtype, device=self.device).reshape(-1)

    def _C_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        C = self.C_net(self._normalize_tokens(tokens.to(self.device))).reshape(-1)
        return C * self.C_ref if _scaled(self.C_net) else C

    def predict_C(self, x, y, t) -> torch.Tensor:
        """Concentration ``C (B,)`` at physical ``(x, y, t)`` via generator + ``C_net``."""
        tokens, _ = self.generator.generate(self._t(x), self._t(y), self._t(t), self.wind_field)
        return self._C_from_tokens(tokens)

    def _cb_inputs(self, s: torch.Tensor, t: torch.Tensor):
        if self.normalizer is not None and getattr(self.normalizer, "s_bounds", None) is not None:
            sh = self.normalizer.normalize_s(s)
        else:
            sh = s
        th = self.normalizer.normalize_t(t) if self.normalizer is not None else t
        return sh, th

    def _Cb_from_hat(self, sh: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        Cb = self.Cb_net(sh, th)
        Cb = torch.as_tensor(Cb, dtype=self.dtype, device=self.device).reshape(-1)
        return Cb * self.C_ref if _scaled(self.Cb_net) else Cb

    def predict_Cb(self, s, t) -> torch.Tensor:
        sh, th = self._cb_inputs(self._t(s), self._t(t))
        return self._Cb_from_hat(sh, th)

    def predict_S(self, x, y, t) -> torch.Tensor:
        bx, by, bt = self._t(x), self._t(y), self._t(t)
        if self.normalizer is not None:
            xh = self.normalizer.normalize_x(bx)
            yh = self.normalizer.normalize_y(by)
            th = self.normalizer.normalize_t(bt)
        else:
            xh, yh, th = bx, by, bt
        S = self.S_net(xh, yh, th).reshape(-1)
        return S * (self.C_ref / self.T_ref) if _scaled(self.S_net) else S

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
        """Dimensional autodiff PDE residual (µg m⁻³ s⁻¹) at physical ``(x, y, t)``.

        ``C`` is a function of the query through the pseudo-sequence
        generator (differentiable RK path) and ``C_net``; autograd takes the
        total derivative, i.e. the derivative of the field actually used.
        """
        xq = self._t(x).requires_grad_(True)
        yq = self._t(y).requires_grad_(True)
        tq = self._t(t).requires_grad_(True)

        tokens, _ = self.generator.generate(xq, yq, tq, self.wind_field)
        C = self._C_from_tokens(tokens)

        dCdx, dCdy, dCdt = torch.autograd.grad(C.sum(), (xq, yq, tq), create_graph=True)
        d2x = torch.autograd.grad(dCdx.sum(), xq, create_graph=True)[0]
        d2y = torch.autograd.grad(dCdy.sum(), yq, create_graph=True)[0]
        lap = d2x + d2y

        u, v = self._wind_at(xq, yq, tq)
        div_term = torch.zeros_like(C)
        if self.form == "conservative":
            try:
                dudx = torch.autograd.grad(u.sum(), xq, create_graph=True, allow_unused=True)[0]
                dvdy = torch.autograd.grad(v.sum(), yq, create_graph=True, allow_unused=True)[0]
                div_u = torch.zeros_like(C)
                if dudx is not None:
                    div_u = div_u + dudx
                if dvdy is not None:
                    div_u = div_u + dvdy
                div_term = C * div_u
            except RuntimeError:
                pass

        S = self.predict_S(xq, yq, tq)
        K, lam = self.phys.forward()
        return dCdt + u * dCdx + v * dCdy + div_term - K * lap - S + lam * C

    def boundary_terms(self, x_b, y_b, s_b, t_b, nx, ny):
        """``(C_pred, Cb_pred, dCdn, undotn, dCb_hat_dt_hat)`` at boundary points."""
        xb = self._t(x_b).requires_grad_(True)
        yb = self._t(y_b).requires_grad_(True)
        tb = self._t(t_b)
        sb = self._t(s_b)
        nxx, nyy = self._t(nx), self._t(ny)

        tokens, _ = self.generator.generate(xb, yb, tb, self.wind_field)
        C = self._C_from_tokens(tokens)
        dCdx, dCdy = torch.autograd.grad(C.sum(), (xb, yb), create_graph=True)
        dCdn = dCdx * nxx + dCdy * nyy

        sh, th = self._cb_inputs(sb.detach(), tb.detach())
        th = th.clone().requires_grad_(True)
        Cb = self._Cb_from_hat(sh, th)
        if Cb.requires_grad:
            dCb_dth = torch.autograd.grad((Cb / self.C_ref).sum(), th, create_graph=True,
                                          allow_unused=True)[0]
        else:
            dCb_dth = None
        if dCb_dth is None:
            dCb_dth = torch.zeros_like(Cb)

        u, v = self._wind_at(xb.detach(), yb.detach(), tb.detach())
        undotn = u * nxx + v * nyy
        return C, Cb, dCdn, undotn, dCb_dth

    def characteristic_terms(self, x, y, t) -> dict[str, torch.Tensor]:
        """Characteristic/FK residual (contribution C1) at collocation points."""
        xq, yq, tq = self._t(x), self._t(y), self._t(t)
        n_mc = self.char_n_mc if self.generator.variant == "advection_jitter" else 1
        if n_mc > 1:
            xq, yq, tq = xq.repeat(n_mc), yq.repeat(n_mc), tq.repeat(n_mc)
        tokens, flags = self.generator.generate(xq, yq, tq, self.wind_field)
        tokens = tokens.to(self.device)
        flags = flags.to(self.device)
        K, lam = self.phys.forward()
        xb = getattr(self.generator, "x_bounds", None) or (
            self.normalizer.x_bounds if self.normalizer is not None else None)
        yb = getattr(self.generator, "y_bounds", None) or (
            self.normalizer.y_bounds if self.normalizer is not None else None)
        if xb is None or yb is None:
            raise ValueError("characteristic loss needs domain bounds (normalizer or generator)")
        return characteristic_residual(
            tokens, flags,
            C_fn=self.predict_C,
            Cb_fn=self.predict_Cb,
            S_fn=self.predict_S,
            lam=lam,
            wind_fn=lambda a, b, c: self._wind_at(a, b, c),
            x_bounds=xb, y_bounds=yb, n_mc=n_mc,
        )

    # ------------------------------------------------------------------
    def compute_losses(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Dimensionless term losses ``{data, pde, bc, reg[, char]}`` (scalars)."""
        dev = self.device
        # ---- data ----
        xd, yd, td = self._t(batch["x_data"]), self._t(batch["y_data"]), self._t(batch["t_data"])
        obs = self._t(batch["C_obs"])
        m = batch.get("mask", torch.ones_like(obs))
        m = torch.as_tensor(m, device=dev).reshape(-1).to(dtype=self.dtype)
        self._maybe_auto_scale(obs, m)
        if self.curriculum is not None:
            cw = self.curriculum.in_window(td.detach(), self._step).to(dtype=self.dtype)
            m = m * cw
        C_pred = self.predict_C(xd, yd, td)
        L_data = masked_mse(C_pred / self.C_ref, torch.nan_to_num(obs) / self.C_ref, m)

        zero = torch.zeros((), device=dev, dtype=self.dtype)
        # ---- pde ----
        S_col = zero
        xp = yp = tp = None
        if "x_pde" in batch:
            xp, yp, tp = self._t(batch["x_pde"]), self._t(batch["y_pde"]), self._t(batch["t_pde"])
            if self.curriculum is not None and tp.numel():
                keep = self.curriculum.in_window(tp.detach(), self._step)
                xp, yp, tp = xp[keep], yp[keep], tp[keep]
        if xp is not None and xp.numel():
            r = self.pde_residual(xp, yp, tp)
            L_pde = pde_loss(r * (self.T_ref / self.C_ref))
            S_col = self.predict_S(xp.detach(), yp.detach(), tp.detach())
        else:
            L_pde = zero

        # ---- bc ----
        dCb = zero
        if "x_bc" in batch:
            C_b, Cb_b, dCdn, undotn, dCb = self.boundary_terms(
                batch["x_bc"], batch["y_bc"], batch["s_bc"],
                batch["t_bc"], batch["nx"], batch["ny"],
            )
            hyst = float(batch.get("hysteresis", self.hysteresis))
            # Match physics/boundary.py::partition: inflow q < -h, outflow
            # q > +h; the hysteresis buffer contributes to neither side.
            inflow = undotn < -hyst
            outflow = undotn > hyst
            L_bc = bc_loss(C_b / self.C_ref, Cb_b / self.C_ref,
                           dCdn * (self.L_ref / self.C_ref), inflow, outflow)
        else:
            L_bc = zero

        # ---- reg (dimensionless) ----
        S_hat = S_col * (self.T_ref / self.C_ref)
        L_src = S_hat.abs().mean() if S_hat.numel() else zero
        L_smooth = (dCb * dCb).mean() if torch.is_tensor(dCb) and dCb.numel() else zero
        L_reg = self.reg_source_l1 * L_src + self.reg_cb_smooth * L_smooth
        out = {"data": L_data, "pde": L_pde, "bc": L_bc, "reg": L_reg}

        # ---- characteristic / Feynman–Kac (contribution C1) ----
        if self.use_char:
            if xp is not None and xp.numel():
                ch = self.characteristic_terms(xp, yp, tp)
                out["char"] = torch.mean((ch["residual"] * (self.T_ref / self.C_ref)) ** 2)
                self.last_diagnostics["boundary_reach"] = float(
                    ch["coupled"].any(dim=1).float().mean().item())
            else:
                out["char"] = zero
        return out

    # ------------------------------------------------------------------
    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Single Adam step; returns detached ``{term: value, total, ...}``."""
        self.train_mode()
        losses = self.compute_losses(batch)

        if self.balancer is not None and self.balancer.should_update(self._step):
            norms = {}
            for k, L in losses.items():
                try:
                    g = torch.autograd.grad(
                        L, list(self.C_net.parameters()), retain_graph=True, allow_unused=True,
                    )
                    tot = sum(float((gg.detach() ** 2).sum()) for gg in g if gg is not None)
                    norms[k] = tot ** 0.5
                except RuntimeError:
                    norms[k] = 0.0
            self.weights.update(self.balancer.update(
                {k: v for k, v in norms.items() if k != "reg"}, step=self._step))

        total = sum(self.weights[k] * losses[k] for k in losses)
        self.optimizer.zero_grad()
        finite = bool(torch.isfinite(total))
        if finite:
            total.backward()
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.grad_clip)
            self.optimizer.step()
        self.scheduler.step()
        self._step += 1

        K, lam = self.phys.forward()
        out: dict[str, Any] = {k: float(v.detach().item()) for k, v in losses.items()}
        out["total"] = float(total.detach().item()) if finite else float("nan")
        out["finite"] = finite
        out["step"] = self._step
        out["weights"] = dict(self.weights)
        out["K"] = float(K.detach())
        out["lam"] = float(lam.detach())
        out["C_ref"] = self.C_ref
        out.update(self.last_diagnostics)
        self.history.append(out)
        return out

    # ------------------------------------------------------------------
    def fit(
        self,
        batch: dict[str, Any] | Callable[[int], dict[str, Any]],
        steps: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run the fit loop (``batch`` is a dict or ``step -> dict``)."""
        tcfg = self.config.get("train", {})
        n = int(steps if steps is not None else tcfg.get("steps", 50))
        ckpt_every = int(tcfg.get("checkpoint_every", 0))
        for i in range(n):
            b = batch(i) if callable(batch) else batch
            self.train_step(b)
            if ckpt_every > 0 and self.checkpoint_hook is not None and (i + 1) % ckpt_every == 0:
                self.checkpoint_hook(i + 1, self.state())
        if bool(tcfg.get("lbfgs_refine", False)):
            self.lbfgs_stage(batch if not callable(batch) else batch(0))
        return self.history

    def lbfgs_stage(self, batch: dict[str, Any]) -> dict:
        """L-BFGS refinement on a fixed batch, with divergence guard."""
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
            "scales": self.scales(),
        }
