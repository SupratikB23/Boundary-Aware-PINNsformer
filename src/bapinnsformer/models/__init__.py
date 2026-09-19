"""Model plane public exports (PRD §4.3 `models/`).

Nets use physics symbols from CLAUDE.md / PRD §2:
``C`` (concentration), ``C_b`` (boundary inflow), ``S`` (source),
``K`` (diffusivity), ``lam`` (deposition), ``u``/``v`` (wind).
"""

from __future__ import annotations

from .activations import Sine, SIREN, Wavelet, get_activation
from .baselines import (
    ClimatologicalInflow,
    MLPPINN,
    TrajectoryRegression,
    ZeroInflow,
)
from .boundary_net import CbNet
from .embeddings import FourierFeatures, SpatioTemporalEmbedding
from .normalizer import Normalizer
from .params import PhysParams, enforce_ordering, ordered_lams_from_raw
from .pinnsformer import PINNsformer
from .pseudoseq import (
    AdvectionBackwardGenerator,
    AdvectionJitterGenerator,
    PseudoSequenceGenerator,
    UniformBackwardGenerator,
    UniformForwardGenerator,
    create_generator,
)
from .source_net import SNet

__all__ = [
    "Normalizer",
    "Wavelet",
    "Sine",
    "SIREN",
    "get_activation",
    "FourierFeatures",
    "SpatioTemporalEmbedding",
    "PseudoSequenceGenerator",
    "UniformForwardGenerator",
    "UniformBackwardGenerator",
    "AdvectionBackwardGenerator",
    "AdvectionJitterGenerator",
    "create_generator",
    "PINNsformer",
    "CbNet",
    "SNet",
    "PhysParams",
    "enforce_ordering",
    "ordered_lams_from_raw",
    "ZeroInflow",
    "ClimatologicalInflow",
    "MLPPINN",
    "TrajectoryRegression",
    "normalizer",
    "activations",
    "embeddings",
    "pseudoseq",
    "pinnsformer",
    "boundary_net",
    "source_net",
    "params",
    "baselines",
]
