"""Structured run logging (PRD §4.3 `utils/logging.py`).

Console logging via :func:`get_logger` plus append-only JSONL event
logs for tidy results tables. Stdlib only.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import uuid

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(
    name: str = "bapinnsformer",
    level: str | int = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """Return a named logger with a console handler (idempotent).

    Args:
        name: Logger name.
        level: Logging level name or number.
        log_file: Optional file path for an additional plain-text handler.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger = _LOGGERS.get(name)
    if logger is not None:
        logger.setLevel(level)
        return logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    if log_file is not None:
        log_file = os.fspath(log_file)
        directory = os.path.dirname(os.path.abspath(log_file))
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == os.path.abspath(log_file)
            for h in logger.handlers
        ):
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(fh)
    _LOGGERS[name] = logger
    return logger


def new_run_id(prefix: str = "run") -> str:
    """Return a unique run id ``<prefix>_YYYYMMDD-HHMMSS_<6hex>`` (UTC)."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


def _json_default(obj):
    try:
        import numpy as _np

        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    try:
        import torch as _torch

        if isinstance(obj, _torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass
    return str(obj)


def append_jsonl(path: str, record: dict) -> None:
    """Append one JSON record (one line) to ``path``, creating parents."""
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=_json_default) + "\n")


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file back into a list of dicts."""
    out: list[dict] = []
    with open(os.fspath(path), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class JsonlLogger:
    """Minimal run-event logger writing ``events.jsonl`` under a run dir."""

    def __init__(self, run_dir: str, logger: logging.Logger | None = None) -> None:
        self.run_dir = os.fspath(run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.path = os.path.join(self.run_dir, "events.jsonl")
        self.logger = logger or get_logger("bapinnsformer.run")

    def log(self, event: str, **fields) -> dict:
        record = {
            "event": event,
            "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            **fields,
        }
        append_jsonl(self.path, record)
        self.logger.info("%s %s", event, {k: v for k, v in fields.items()})
        return record
