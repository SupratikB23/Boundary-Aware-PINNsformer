"""Checkpointing (PRD §4.3 `train/checkpoint.py`).

A checkpoint must be sufficient to regenerate every reported number:
all nets + scalars + normalizer + config hash + git commit + split hash.
Format is a ``torch.save`` dict; normalizer/config stay JSON-serializable
inside it so checkpoints remain inspectable without torch.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

__all__ = ["config_hash", "get_git_commit", "save_checkpoint", "load_checkpoint"]


def config_hash(config: dict[str, Any]) -> str:
    """Short SHA-256 over the canonical JSON of ``config``."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def get_git_commit(cwd: str | Path | None = None) -> str:
    """Current git SHA or ``"unknown"`` (never raises — safe on Kaggle)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = out.stdout.strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"


def save_checkpoint(
    path: str | Path,
    C_net: torch.nn.Module,
    Cb_net: torch.nn.Module,
    S_net: torch.nn.Module,
    phys_params: torch.nn.Module,
    normalizer: Any,
    config: dict[str, Any],
    split_hash: str = "",
    extra: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> Path:
    """Save nets + normalizer + provenance. Returns the checkpoint path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "C_net": C_net.state_dict(),
        "Cb_net": Cb_net.state_dict(),
        "S_net": S_net.state_dict(),
        "phys_params": phys_params.state_dict(),
        "normalizer": normalizer.to_dict() if hasattr(normalizer, "to_dict") else normalizer,
        "config": config,
        "config_hash": config_hash(config),
        "git_commit": get_git_commit(),
        "split_hash": split_hash,
        "torch_version": torch.__version__,
        "timestamp": time.time(),
        "extra": extra or {},
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    torch.save(payload, p)
    return p


def load_checkpoint(
    path: str | Path,
    C_net: torch.nn.Module | None = None,
    Cb_net: torch.nn.Module | None = None,
    S_net: torch.nn.Module | None = None,
    phys_params: torch.nn.Module | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint, restoring any nets passed in.

    Returns the full payload dict (normalizer dict under
    ``payload["normalizer"]`` — rebuild with ``Normalizer.from_dict``).
    ``map_location="cpu"`` default keeps loads CPU-safe.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    for net, key in (
        (C_net, "C_net"),
        (Cb_net, "Cb_net"),
        (S_net, "S_net"),
        (phys_params, "phys_params"),
    ):
        if net is not None and key in payload:
            net.load_state_dict(payload[key])
    return payload
