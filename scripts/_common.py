"""Shared CLI helpers for scripts/*.py (not a PRD entry point itself).

Thin argparse + config + run-dir + provenance plumbing so every script
stays small. Plain YAML, stdlib + package imports only.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make `src/` importable when invoked as `python scripts/XX_*.py`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default="configs/config.yaml",
                        help="YAML config file")
    parser.add_argument("--results-root", default=None,
                        help="Override results root (default from config or 'results')")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--run-id", default=None, help="Fixed run id (default: generated)")
    return parser


def init_run(args) -> tuple[dict, str, str, object]:
    """Load (+compose) config, seed RNGs, create run dir. Returns (cfg, run_id, rundir, log)."""
    from bapinnsformer.utils.io import compose_config, ensure_dir, load_config
    from bapinnsformer.utils.logging import JsonlLogger, get_logger, new_run_id
    from bapinnsformer.utils.seeding import seed_all

    try:
        cfg = compose_config(args.config)
    except (FileNotFoundError, ValueError):
        # Experiment files without a `defaults:` list compose to themselves.
        cfg = load_config(args.config)
    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 0))
    seed_all(seed)
    results_root = args.results_root or cfg.get("results_root", "results")
    sub = str(cfg.get("results_subdir", "runs"))
    run_id = args.run_id or new_run_id(prefix=sub.split("_")[0] if sub != "runs" else "run")
    rundir = ensure_dir(os.path.join(results_root, "runs", run_id))
    log = get_logger("bapinnsformer.scripts", log_file=os.path.join(rundir, "stdout.log"))
    JsonlLogger(rundir, logger=log).log("run_start", run_id=run_id, config=str(args.config), seed=seed)
    return cfg, run_id, rundir, log


def save_results(rundir: str, run_id: str, cfg: dict, metrics: dict,
                 split_hash=None, log=None) -> str:
    """Write provenance-stamped metrics.json + validate. Returns path."""
    from bapinnsformer.utils.io import validate_results_record, write_json
    from bapinnsformer.utils.provenance import collect_provenance

    record = {
        "run_id": run_id,
        "config": cfg,
        "provenance": collect_provenance(cfg, split_hash=split_hash, run_id=run_id, repo=_REPO),
        "metrics": dict(metrics),
    }
    validate_results_record(record)
    path = write_json(os.path.join(rundir, "metrics.json"), record)
    write_json(os.path.join(rundir, "config_snapshot.json"), cfg)
    if log is not None:
        log.info("wrote %s", path)
    return path
