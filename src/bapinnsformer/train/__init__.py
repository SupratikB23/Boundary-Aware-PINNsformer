"""Training plane public exports (PRD §4.3 `train/`)."""

from __future__ import annotations

from .balancing import GradNormBalancer
from .checkpoint import (
    config_hash,
    get_git_commit,
    load_checkpoint,
    save_checkpoint,
)
from .curriculum import Curriculum
from .losses import (
    bc_loss,
    cb_temporal_smooth,
    masked_mse,
    pde_loss,
    positivity_penalty,
    source_l1,
    source_tv,
)
from .optim import build_adam, lbfgs_refine
from .trainer import Trainer

__all__ = [
    "GradNormBalancer",
    "Curriculum",
    "Trainer",
    "build_adam",
    "lbfgs_refine",
    "save_checkpoint",
    "load_checkpoint",
    "config_hash",
    "get_git_commit",
    "masked_mse",
    "pde_loss",
    "bc_loss",
    "source_l1",
    "source_tv",
    "cb_temporal_smooth",
    "positivity_penalty",
    "losses",
    "balancing",
    "curriculum",
    "optim",
    "trainer",
    "checkpoint",
]
