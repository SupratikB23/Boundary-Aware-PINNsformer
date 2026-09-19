"""FIRMS leakage guard (PRD §4.5 test_leakage.py, §10).

E4's validity rests on fire data never entering training. Fails on:
1. any static ``firms_ingest`` import (incl. relative) under
   models|physics|train (AST import-graph assertion);
2. any read of ``data/heldout`` (literal ``heldout`` path usage) there.
Cited in the paper as the no-leakage audit trail.
"""

import ast
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARDED = ("models", "physics", "train")
FORBIDDEN_IMPORTS = ("firms_ingest",)
FORBIDDEN_PATH_FRAGMENTS = ("heldout",)


def _guarded_files():
    files = []
    for plane in GUARDED:
        root = os.path.join(REPO, "src", "bapinnsformer", plane)
        assert os.path.isdir(root), f"missing package dir {root}"
        for dirpath, _, names in os.walk(root):
            for name in sorted(names):
                if name.endswith(".py"):
                    files.append(os.path.join(dirpath, name))
    assert files, "no guarded sources found"
    return files


def _resolve_relative(node: ast.ImportFrom, path: str) -> str:
    # Map `from .x import y` / `from ..eval import z` to absolute dotted name.
    pkg = "bapinnsformer." + os.path.splitext(
        os.path.relpath(path, os.path.join(REPO, "src", "bapinnsformer"))
    )[0].replace(os.sep, ".")
    if node.level:
        parts = pkg.split(".")
        base = parts[: len(parts) - node.level + 1]
        mod = (node.module or "").strip()
        return ".".join([*base, mod]).rstrip(".")
    return node.module or ""


def test_no_firms_import_in_guarded_planes():
    offenders = []
    for path in _guarded_files():
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [_resolve_relative(node, path) + "." + a.name for a in node.names]
                names.append(_resolve_relative(node, path))
            else:
                continue
            if any(any(frag in n for frag in FORBIDDEN_IMPORTS) for n in names):
                offenders.append(f"{path}: {names}")
    assert not offenders, "FIRMS import inside guarded planes:\n" + "\n".join(offenders)


def test_no_heldout_reads_in_guarded_planes():
    offenders = []
    for path in _guarded_files():
        text = open(path, encoding="utf-8").read().lower()
        for frag in FORBIDDEN_PATH_FRAGMENTS:
            if frag in text:
                offenders.append(f"{path}: contains {frag!r}")
    assert not offenders, "heldout path usage inside guarded planes:\n" + "\n".join(offenders)


def test_firms_module_lives_only_in_data_plane():
    here = os.path.join(REPO, "src", "bapinnsformer", "data", "firms_ingest.py")
    assert os.path.isfile(here), "data/firms_ingest.py must exist (single allowed home)"
    for plane in GUARDED:
        clash = os.path.join(REPO, "src", "bapinnsformer", plane, "firms_ingest.py")
        assert not os.path.isfile(clash), f"firms_ingest must not live under {plane}/"
