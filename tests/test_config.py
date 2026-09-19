"""Config composition + registry resolution (PRD §4.5 test_config.py).

Every YAML under configs/ must parse to a mapping; the root
`defaults` must point at existing files; every variant/activation/
baseline name referenced anywhere must resolve in utils/registry.py.
"""

import glob
import os

import pytest
import yaml

from bapinnsformer.utils.registry import ACTIVATION, BASELINE, PSEUDOSEQ

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(REPO, "configs")


def _all_configs():
    return sorted(glob.glob(os.path.join(CONFIGS, "**", "*.yaml"), recursive=True))


def test_configs_exist_and_parse():
    paths = _all_configs()
    assert len(paths) >= 17, f"expected ≥17 configs, found {len(paths)}"
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        assert isinstance(cfg, dict), f"{path} must be a YAML mapping"


def test_root_defaults_point_at_files():
    with open(os.path.join(CONFIGS, "config.yaml"), encoding="utf-8") as fh:
        root = yaml.safe_load(fh)
    assert isinstance(root.get("defaults"), list) and root["defaults"]
    for rel in root["defaults"]:
        assert os.path.isfile(os.path.join(CONFIGS, rel)), f"missing default {rel}"


def _collect(node, keys, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in keys:
                out.setdefault(k, []).extend(v if isinstance(v, list) else [v])
            _collect(v, keys, out)
    elif isinstance(node, list):
        for v in node:
            _collect(v, keys, out)


def test_every_registry_name_resolves():
    registries = {
        "variant": PSEUDOSEQ, "variants": PSEUDOSEQ, "pseudoseq_variant": PSEUDOSEQ,
        "activation": ACTIVATION, "activations": ACTIVATION,
        "baselines": BASELINE,
    }
    failures = []
    for path in _all_configs():
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        found: dict = {}
        _collect(cfg, set(registries), found)
        for key, names in found.items():
            for name in names:
                try:
                    registries[key].resolve(str(name))
                except (KeyError, ImportError, ValueError) as exc:
                    failures.append(f"{path}: {key}={name!r} ({exc})")
    assert not failures, "unresolvable registry names:\n" + "\n".join(failures)


def test_experiment_files_carry_grids():
    import glob as _g

    exp = _g.glob(os.path.join(CONFIGS, "experiment", "*.yaml"))
    assert len(exp) == 5
    e1 = yaml.safe_load(open(os.path.join(CONFIGS, "experiment", "e1_identifiability.yaml")))
    assert e1["station_counts"] == [7, 15, 25, 40]
