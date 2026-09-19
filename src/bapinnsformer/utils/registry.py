"""Name→class registries (PRD §4.3 `utils/registry.py`).

Configs select pseudo-sequence variants, activations and baselines
by string; this module resolves them. Registration targets are lazy
``"module:Class"`` strings so importing this module never imports
torch or any model code.
"""

from __future__ import annotations

import importlib


class Registry:
    """Minimal string registry with lazy ``module:Class`` targets."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: dict[str, str | type] = {}

    def register(self, key: str, target: str | type | None = None):
        """Decorator or direct call: ``@REG.register("name")`` or
        ``REG.register("name", "pkg.mod:Cls")``."""
        if target is not None:
            self._entries[key] = target
            return target

        def deco(obj: str | type):
            self._entries[key] = obj
            return obj

        return deco

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def resolve(self, key: str) -> type:
        """Import (if needed) and return the class for ``key``."""
        if key not in self._entries:
            raise KeyError(
                f"Unknown {self.name} {key!r}. Available: {self.names()}"
            )
        target = self._entries[key]
        if isinstance(target, str):
            module_name, _, attr = target.partition(":")
            if not module_name or not attr:
                raise ValueError(
                    f"Bad registry target {target!r}; expected 'module:Class'"
                )
            module = importlib.import_module(module_name)
            target = getattr(module, attr)
            self._entries[key] = target  # cache resolved class
        return target

    def create(self, key: str, *args, **kwargs):
        """Resolve ``key`` and instantiate with ``args``/``kwargs``."""
        return self.resolve(key)(*args, **kwargs)


PSEUDOSEQ = Registry("pseudoseq")
PSEUDOSEQ.register(
    "uniform_forward", "bapinnsformer.models.pseudoseq:UniformForwardGenerator"
)
PSEUDOSEQ.register(
    "uniform_backward", "bapinnsformer.models.pseudoseq:UniformBackwardGenerator"
)
PSEUDOSEQ.register(
    "advection_backward",
    "bapinnsformer.models.pseudoseq:AdvectionBackwardGenerator",
)
PSEUDOSEQ.register(
    "advection_jitter", "bapinnsformer.models.pseudoseq:AdvectionJitterGenerator"
)

ACTIVATION = Registry("activation")
ACTIVATION.register("wavelet", "bapinnsformer.models.activations:Wavelet")
ACTIVATION.register("tanh", "bapinnsformer.models.activations:TanhAct")
ACTIVATION.register("gelu", "bapinnsformer.models.activations:GeluAct")
ACTIVATION.register("siren", "bapinnsformer.models.activations:SirenAct")

BASELINE = Registry("baseline")
BASELINE.register(
    "zero_inflow", "bapinnsformer.models.baselines:ZeroInflowBaseline"
)
BASELINE.register(
    "climatological_inflow",
    "bapinnsformer.models.baselines:ClimatologicalInflowBaseline",
)
BASELINE.register("mlp_pinn", "bapinnsformer.models.baselines:MlpPinnBaseline")
BASELINE.register(
    "uniform_psf", "bapinnsformer.models.baselines:UniformPsfBaseline"
)
BASELINE.register(
    "trajectory", "bapinnsformer.models.baselines:TrajectoryBaseline"
)

__all__ = ["Registry", "PSEUDOSEQ", "ACTIVATION", "BASELINE"]
