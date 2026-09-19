"""Run provenance capture (PRD §4.3 `utils/provenance.py`, §10).

Every results record carries: git commit, composed-config hash, split
hash (when applicable), raw-data checksums, package versions,
hardware, wall-clock. Stdlib + lazy torch/numpy probing only.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess


def git_commit(repo: str | None = None) -> str:
    """Return the current git SHA, or ``"unknown"`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = (out.stdout or "").strip()
        if out.returncode == 0 and sha:
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo or os.getcwd(),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if (dirty.stdout or "").strip():
                return sha + "-dirty"
            return sha
    except Exception:
        pass
    return "unknown"


def config_hash(config: dict) -> str:
    """Stable sha256 over the canonical JSON of a composed config."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_checksum(path: str, n_bytes: int = 0) -> str:
    """sha256 of a file (whole file unless ``n_bytes`` > 0)."""
    digest = hashlib.sha256()
    with open(os.fspath(path), "rb") as fh:
        if n_bytes and n_bytes > 0:
            digest.update(fh.read(n_bytes))
        else:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def package_versions(extra: tuple[str, ...] = ()) -> dict[str, str]:
    """Report versions for the pinned stack; ``"missing"`` if absent."""
    names = (
        "python",
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "xarray",
        "netCDF4",
        "matplotlib",
        "yaml",
        "sklearn",
        *extra,
    )
    versions: dict[str, str] = {"os": platform.platform(), "python": platform.python_version()}
    for name in names:
        if name in versions:
            continue
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[name] = "missing"
    try:  # torch device summary without importing at module level
        import torch

        versions["torch_cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                versions["torch_cuda_device"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    except ImportError:
        versions["torch_cuda_available"] = "missing"
    return versions


def collect_provenance(
    config: dict,
    split_hash: str | None = None,
    data_checksums: dict[str, str] | None = None,
    run_id: str | None = None,
    repo: str | None = None,
) -> dict:
    """Build the provenance block stored with every results record."""
    return {
        "run_id": run_id or "unknown",
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_commit": git_commit(repo),
        "config_hash": config_hash(config),
        "split_hash": split_hash,
        "data_checksums": dict(data_checksums or {}),
        "versions": package_versions(),
    }
