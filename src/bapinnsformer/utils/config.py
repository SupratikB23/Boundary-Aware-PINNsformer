"""Config adapters: composed YAML (PRD §4.2 key names) → constructor kwargs.

PRD §4.2 YAMLs use short keys (``depth/width``, ``n_heads/d_ff``,
``L/dt_hours``, ``adam_steps``...); model/train constructors use explicit
names. These adapters translate, so ``Model(**yaml)`` never raises
``TypeError: unexpected keyword`` (review fix). All functions ignore
unknown descriptive keys (``param_target``, ``note``...) and raise on
truly unexpected ones via the constructors.
"""

from __future__ import annotations

from typing import Any

SEASONS = ("postmonsoon", "winter")


def normalize_season(season: str) -> str:
    """Accept ``post-monsoon``/``postmonsoon``/``winter`` → canonical label."""
    s = str(season).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if s in ("postmonsoon", "post-monsoon"):
        return "postmonsoon"
    if s == "winter":
        return "winter"
    raise ValueError(f"unknown season {season!r}; expected postmonsoon|winter")


def pinnsformer_kwargs(y: dict | None) -> dict:
    """``configs/model/pinnsformer.yaml`` → :class:`PINNsformer` kwargs."""
    y = dict(y or {})
    out: dict[str, Any] = {}
    mapping = {
        "d_model": "d_model", "n_heads": "nhead", "nhead": "nhead",
        "n_encoder_layers": "num_encoder_layers",
        "n_decoder_layers": "num_decoder_layers",
        "d_ff": "dim_feedforward", "dim_feedforward": "dim_feedforward",
        "activation": "activation", "dropout": "dropout",
        "encoder_free": "encoder_free", "in_dim": "in_dim",
        "fourier_features": "fourier_features",
        "fourier_scale": "fourier_scale",
        "query_index": "query_index", "out_transform": "out_transform",
    }
    for k, v in y.items():
        if k in ("param_target", "note"):
            continue
        if k not in mapping:
            raise TypeError(f"pinnsformer config has unknown key {k!r}")
        out[mapping[k]] = v
    return out


def cb_kwargs(y: dict | None) -> dict:
    """``configs/model/cb_net.yaml`` → :class:`CbNet` kwargs."""
    y = dict(y or {})
    out: dict[str, Any] = {}
    mapping = {
        "width": "hidden_dim", "hidden_dim": "hidden_dim",
        "depth": "num_layers", "num_layers": "num_layers",
        "activation": "activation", "fourier_features": "fourier_features",
        "fourier_scale": "fourier_scale",
        "output_transform": "out_transform", "out_transform": "out_transform",
        "pollutant_list": "pollutant_list",
        "per_pollutant_heads": "per_pollutant_heads",
    }
    for k, v in y.items():
        if k in ("temporal_smoothness", "note"):
            continue
        if k not in mapping:
            raise TypeError(f"cb_net config has unknown key {k!r}")
        out[mapping[k]] = v
    return out


def s_kwargs(y: dict | None) -> dict:
    """``configs/model/s_net.yaml`` → :class:`SNet` kwargs."""
    y = dict(y or {})
    out: dict[str, Any] = {}
    mapping = {
        "width": "hidden_dim", "hidden_dim": "hidden_dim",
        "depth": "num_layers", "num_layers": "num_layers",
        "activation": "activation", "fourier_features": "fourier_features",
        "fourier_scale": "fourier_scale",
        "output_transform": "out_transform", "out_transform": "out_transform",
    }
    for k, v in y.items():
        if k in ("l1_weight", "tv_weight", "note"):
            continue
        if k not in mapping:
            raise TypeError(f"s_net config has unknown key {k!r}")
        out[mapping[k]] = v
    return out


def pseudoseq_kwargs(y: dict | None) -> dict:
    """``configs/pseudoseq/*.yaml`` → generator kwargs (+ resolved name)."""
    from bapinnsformer.models.pseudoseq import _translate_yaml_aliases

    y = dict(y or {})
    name = str(y.pop("variant", y.pop("name", "advection_backward")))
    return {"name": name, **_translate_yaml_aliases(y)}


def train_kwargs(y: dict | None) -> dict:
    """``configs/train/default.yaml`` → :class:`Trainer` ``train`` block."""
    y = dict(y or {})
    out: dict[str, Any] = {}
    if "lr" in y:
        out["lr"] = y["lr"]
    out["steps"] = int(y.get("adam_steps", y.get("steps", 100)))
    if y.get("lbfgs_steps", 0):
        out["lbfgs_refine"] = True
        out["lbfgs_max_iter"] = int(y.get("lbfgs_steps", 50))
    else:
        out["lbfgs_refine"] = bool(y.get("lbfgs_refine", False))
        out["lbfgs_max_iter"] = int(y.get("lbfgs_max_iter", 50))
    balancing = str(y.get("balancing", "gradnorm")).lower()
    out["balance_every"] = int(y.get("balancing_period", 0)) if balancing == "gradnorm" else 0
    for k in ("w_data", "w_pde", "w_bc", "w_reg", "w_char", "char_n_mc",
              "reg_source_l1", "reg_cb_smooth", "weight_decay",
              "grad_clip", "hysteresis", "checkpoint_every"):
        if k in y:
            out[k] = y[k]
    cur = y.get("curriculum")
    if isinstance(cur, dict) and cur.get("enabled", False):
        # Time bounds (t0/t1) come from the dataset at fit time (E0 output),
        # so only the stage count is kept here; build_trainer_config hoists
        # it to `curriculum_request` and leaves `curriculum` None until then.
        out["curriculum_request"] = {"stages": int(cur.get("stages", 4))}
    return out


def build_trainer_config(composed: dict) -> dict:
    """Assemble a :class:`Trainer` config dict from composed YAML sections."""
    from bapinnsformer.utils.io import resolve_device

    cfg = dict(composed)
    model_cfg = cfg.get("model") or cfg.get("pinnsformer") or {}
    train_block = train_kwargs(cfg.get("train") if isinstance(cfg.get("train"), dict) else {})
    # Curriculum time bounds (t0/t1) are only known once E0 data exists;
    # until then the fit runs without curriculum and the request is kept
    # visible (never silently dropped).
    curriculum_request = train_block.pop("curriculum_request", None)
    # Regularizer weights live next to the nets they regularize in the YAMLs.
    s_cfg = cfg.get("s_net") if isinstance(cfg.get("s_net"), dict) else {}
    cb_cfg = cfg.get("cb_net") if isinstance(cfg.get("cb_net"), dict) else {}
    if "l1_weight" in s_cfg and "reg_source_l1" not in train_block:
        train_block["reg_source_l1"] = float(s_cfg["l1_weight"])
    if "temporal_smoothness" in cb_cfg and "reg_cb_smooth" not in train_block:
        train_block["reg_cb_smooth"] = float(cb_cfg["temporal_smoothness"])
    return {
        "city": cfg.get("city", "delhi_ncr"),
        "season": normalize_season(cfg.get("season", "postmonsoon")),
        "pollutant": cfg.get("pollutant", "PM2.5"),
        "seed": int(cfg.get("seed", 0)),
        "device": resolve_device(cfg.get("device", "cpu")),
        "model": pinnsformer_kwargs(model_cfg if isinstance(model_cfg, dict) else {}),
        "cb_net": cb_kwargs(cfg.get("cb_net") if isinstance(cfg.get("cb_net"), dict) else {}),
        "s_net": s_kwargs(cfg.get("s_net") if isinstance(cfg.get("s_net"), dict) else {}),
        "phys": cfg.get("phys", {"K_init": 100.0, "lam_init": 1e-5}),
        "pseudoseq": pseudoseq_kwargs(cfg.get("pseudoseq") if isinstance(cfg.get("pseudoseq"), dict) else {}),
        "physics": dict(cfg.get("physics") or {"form": "advective"}),
        "scales": dict(cfg.get("scales") or {"mode": "auto"}),
        "train": train_block,
        "curriculum": None,
        "curriculum_request": curriculum_request,
    }


__all__ = [
    "SEASONS",
    "normalize_season",
    "pinnsformer_kwargs",
    "cb_kwargs",
    "s_kwargs",
    "pseudoseq_kwargs",
    "train_kwargs",
    "build_trainer_config",
]
