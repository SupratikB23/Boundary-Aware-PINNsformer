"""Path conventions + portable readers/writers + results validation.

PRD §4.3 (`utils/io.py`). Heavy format libraries (pandas, xarray,
netCDF4) are imported lazily inside the functions that need them so
that `import bapinnsformer.utils.io` works with torch+numpy only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def repo_root(start: str | None = None) -> str:
    """Walk up from ``start`` (or this file) to the repo root (PRD.md)."""
    here = os.path.abspath(start or __file__)
    if os.path.isfile(here):
        here = os.path.dirname(here)
    cur = here
    while True:
        if os.path.isfile(os.path.join(cur, "PRD.md")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(here)
        cur = parent


def ensure_dir(path: str) -> str:
    """Create ``path`` (and parents) if needed; return it."""
    path = os.fspath(path)
    os.makedirs(path, exist_ok=True)
    return path


def run_dir(results_root: str, run_id: str) -> str:
    """Return (creating) ``<results_root>/runs/<run_id>``."""
    return ensure_dir(os.path.join(os.fspath(results_root), "runs", run_id))


# ---------------------------------------------------------------------------
# JSON / YAML
# ---------------------------------------------------------------------------


def read_json(path: str):
    with open(os.fspath(path), encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj, indent: int = 2) -> str:
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=str)
        fh.write("\n")
    return path


def read_yaml(path: str):
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required to read YAML configs") from exc
    with open(os.fspath(path), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_yaml(path: str, obj) -> str:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required to write YAML configs") from exc
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(obj), fh, sort_keys=False)
    return path


def load_config(path: str) -> dict:
    """Load a YAML config file; must be a mapping."""
    cfg = read_yaml(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path!r} must be a YAML mapping")
    return cfg


def resolve_device(device: str | None) -> str:
    """Resolve ``auto | cpu | cuda[.validate]`` to ``"cpu"``/``"cuda"``.

    ``auto`` → ``"cuda"`` when torch reports CUDA, else ``"cpu"``.
    Raises on unknown strings instead of ``torch.device("auto")``.
    """
    d = str(device or "cpu").strip().lower()
    if d == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if d in ("cpu", "cuda", "cuda:0"):
        return "cuda" if d.startswith("cuda") else "cpu"
    raise ValueError(f"unknown device {device!r}; expected auto|cpu|cuda")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive mapping merge (override wins; lists replace)."""
    out = dict(base)
    for k, v in override.items():
        if k == "defaults":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _section_key(rel: str) -> str | None:
    """Map a ``defaults`` entry to its config section (``None`` = merge flat)."""
    rel = str(rel).replace("\\", "/")
    table = {
        "model/pinnsformer.yaml": "model",
        "model/cb_net.yaml": "cb_net",
        "model/s_net.yaml": "s_net",
        "data/cpcb.yaml": "cpcb",
        "data/era5.yaml": "era5",
        "train/default.yaml": "train",
    }
    if rel in table:
        return table[rel]
    if rel.startswith("domain/"):
        return "domain"
    if rel.startswith("pseudoseq/"):
        return "pseudoseq"
    return None


def compose_config(path: str) -> dict:
    """Load ``path`` plus its ``defaults:`` section files (later wins).

    Mirrors ``configs/config.yaml`` composition (PRD §4.2/§5) without a
    hydra dependency. Each ``defaults`` entry is resolved relative to the
    ``configs/`` directory and nested under its section key (``model``,
    ``cb_net``, ``s_net``, ``cpcb``, ``era5``, ``domain``, ``pseudoseq``,
    ``train``) so same-named keys in different sections (e.g. ``depth``)
    never collide. Root-file keys win over section files. Experiment
    files without ``defaults`` load as-is. ``device: auto`` resolves to
    ``cpu``/``cuda``.
    """
    root = load_config(path)
    merged: dict = {}
    defaults = root.get("defaults", [])
    if defaults:
        cfg_dir = os.path.dirname(os.path.abspath(path))
        # `defaults` entries are relative to configs/ (e.g. "domain/delhi_ncr.yaml").
        # If `path` itself lives directly under configs/, cfg_dir is right;
        # otherwise walk up to the enclosing configs/ dir.
        probe, configs_dir = cfg_dir, None
        while True:
            if os.path.basename(probe) == "configs" or os.path.isdir(os.path.join(probe, "configs")):
                configs_dir = probe if os.path.basename(probe) == "configs" else os.path.join(probe, "configs")
                break
            parent = os.path.dirname(probe)
            if parent == probe:
                configs_dir = cfg_dir
                break
            probe = parent
        for rel in defaults:
            section_path = os.path.join(configs_dir, str(rel))
            if not os.path.isfile(section_path):
                raise FileNotFoundError(f"config default not found: {section_path} (from {path})")
            section = load_config(section_path)
            key = _section_key(rel)
            if key is None:
                merged = _deep_merge(merged, section)
            else:
                merged[key] = _deep_merge(merged.get(key, {}), section)
    merged = _deep_merge(merged, {k: v for k, v in root.items() if k != "defaults"})
    if "device" in merged:
        merged["device"] = resolve_device(merged.get("device"))
    return merged


# ---------------------------------------------------------------------------
# Tabular / gridded helpers (lazy optional deps)
# ---------------------------------------------------------------------------


def write_parquet(path: str, rows: list[dict] | dict) -> str:
    """Write rows to parquet (pandas) with a CSV fallback."""
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if isinstance(rows, dict):
        rows = [rows]
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
        return path
    except ImportError:
        fallback = os.path.splitext(path)[0] + ".csv"
        write_csv(fallback, rows)
        return fallback


def read_parquet(path: str):
    """Read parquet (pandas); falls back to a CSV sidecar."""
    try:
        import pandas as pd

        return pd.read_parquet(os.fspath(path))
    except ImportError:
        return read_csv(os.path.splitext(os.fspath(path))[0] + ".csv")


def write_csv(path: str, rows: list[dict]) -> str:
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    rows = list(rows)
    fieldnames = sorted({k for r in rows for k in r}) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path: str) -> list[dict]:
    with open(os.fspath(path), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_netcdf(path: str, arrays: dict[str, object]) -> str:
    """Write named numpy arrays to NetCDF (netCDF4/scipy) or ``.npz``."""
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        from netCDF4 import Dataset  # type: ignore

        import numpy as np

        with Dataset(path, "w") as ds:
            for name, arr in arrays.items():
                arr = np.asarray(arr)
                dims = tuple(f"{name}_d{i}" for i in range(arr.ndim))
                for d, n in zip(dims, arr.shape):
                    if d not in ds.dimensions:
                        ds.createDimension(d, n)
                var = ds.createVariable(name, arr.dtype, dims)
                var[:] = arr
        return path
    except ImportError:
        pass
    try:
        from scipy.io import netcdf_file  # type: ignore

        import numpy as np

        with netcdf_file(path, "w") as fh:
            for name, arr in arrays.items():
                arr = np.asarray(arr)
                fh.createVariable(name, arr.dtype.char, arr.shape)[:] = arr
        return path
    except ImportError:
        import numpy as np

        fallback = os.path.splitext(path)[0] + ".npz"
        np.savez(fallback, **{k: np.asarray(v) for k, v in arrays.items()})
        return fallback


# ---------------------------------------------------------------------------
# Results schema
# ---------------------------------------------------------------------------

REQUIRED_RESULTS_KEYS = ("run_id", "config", "provenance", "metrics")
REQUIRED_PROVENANCE_KEYS = (
    "git_commit",
    "config_hash",
    "versions",
    "utc",
)


def validate_results_record(record: dict) -> dict:
    """Validate a results record (PRD §4.7); raises on violation."""
    if not isinstance(record, dict):
        raise TypeError("results record must be a dict")
    missing = [k for k in REQUIRED_RESULTS_KEYS if k not in record]
    if missing:
        raise ValueError(f"results record missing keys: {missing}")
    prov = record.get("provenance", {})
    missing_p = [k for k in REQUIRED_PROVENANCE_KEYS if k not in prov]
    if missing_p:
        raise ValueError(f"provenance block missing keys: {missing_p}")
    if not isinstance(record.get("metrics", {}), dict):
        raise ValueError("results record 'metrics' must be a dict")
    return record


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
