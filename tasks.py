"""Cross-platform task runner (replaces the Makefile).

Why this file exists: `make` is not installed on Windows (PowerShell) and is
one more dependency on Kaggle. Every target below maps 1:1 to a
`scripts/*.py` entry point + its `configs/**/*.yaml` composition (PRD §4.4),
exactly like the old Makefile did.

Usage (repo root):
    python tasks.py data        # E0 pipeline
    python tasks.py e1          # E1 synthetic identifiability sweep
    python tasks.py e2          # E2 real-data fits (fan-out)
    python tasks.py e3          # E3 ablations + profiling
    python tasks.py e4          # E4 fire validation (held-out only)
    python tasks.py e5          # E5 transfer airshed
    python tasks.py figures     # tables + figures (figures implies tables)
    python tasks.py tables      # LaTeX tables from results/
    python tasks.py paper       # latexmk build (needs a TeX toolchain)
    python tasks.py test        # pytest suite
    python tasks.py smoke       # fastest end-to-end sanity (E1 quick + E2 smoke-fit)
    python tasks.py --list      # list targets

Extra script args go after `--`, e.g.:
    python tasks.py e1 -- --quick
    python tasks.py e2 -- --max-fits 2
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable or "python"

TASKS: dict[str, dict] = {
    "data": {"steps": [
        [PY, "scripts/03_build_dataset.py", "--config", "configs/config.yaml"],
        [PY, "scripts/04_characterize_data.py", "--config", "configs/config.yaml"],
    ], "help": "E0: QC, align, domain, splits -> data/processed (+ report)"},
    "e1": {"steps": [
        [PY, "scripts/10_run_synthetic_sweep.py", "--config",
         "configs/experiment/e1_identifiability.yaml"],
    ], "help": "E1: synthetic identifiability sweep"},
    "e2": {"steps": [
        [PY, "scripts/21_fit_all_real.py", "--config", "configs/experiment/e2_real.yaml"],
    ], "help": "E2: real Delhi-NCR fits (interior fit / perimeter validation)"},
    "e3": {"steps": [
        [PY, "scripts/30_run_ablations.py", "--config", "configs/experiment/e3_ablation.yaml"],
    ], "help": "E3: pseudo-sequence / activation / physics ablations + profiling"},
    "e4": {"steps": [
        [PY, "scripts/40_validate_fires.py", "--config", "configs/experiment/e4_validation.yaml"],
    ], "help": "E4: NW inflow vs withheld VIIRS FRP + external comparison"},
    "e5": {"steps": [
        [PY, "scripts/50_transfer_airshed.py", "--config", "configs/experiment/e5_transfer.yaml"],
    ], "help": "E5: transfer to second airshed"},
    "tables": {"steps": [
        [PY, "scripts/91_make_tables.py", "--config", "configs/config.yaml"],
    ], "help": "LaTeX tables from results/ -> paper/tables/"},
    "figures": {"steps": [
        [PY, "scripts/91_make_tables.py", "--config", "configs/config.yaml"],
        [PY, "scripts/90_make_figures.py", "--config", "configs/config.yaml"],
    ], "help": "Tables + paper/figures/ from results/"},
    "paper": {"steps": [
        [PY, "scripts/91_make_tables.py", "--config", "configs/config.yaml"],
        [PY, "scripts/90_make_figures.py", "--config", "configs/config.yaml"],
        ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"],
    ], "help": "Tables + figures + latexmk paper/main.pdf"},
    "test": {"steps": [
        [PY, "-m", "pytest", "-q"],
    ], "help": "pytest suite (manufactured-solution + leakage guards)"},
    "smoke": {"steps": [
        [PY, "scripts/10_run_synthetic_sweep.py", "--config",
         "configs/experiment/e1_identifiability.yaml", "--quick"],
        [PY, "scripts/20_fit_real.py", "--config", "configs/config.yaml",
         "--smoke-fit", "--smoke-steps", "2"],
    ], "help": "Fastest end-to-end sanity (CPU, ~1 min)"},
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--list":
        for name, spec in TASKS.items():
            print(f"{name:10s} {spec['help']}")
        return 0
    name, extra = argv[0], []
    if "--" in argv:
        i = argv.index("--")
        extra = argv[i + 1:]
        argv = argv[:i]
        name = argv[0]
    if name not in TASKS:
        print(f"unknown task {name!r}. Available: {', '.join(TASKS)}")
        return 2
    steps = [list(s) for s in TASKS[name]["steps"]]
    if extra:
        steps[-1] = steps[-1] + extra
    for cmd in steps:
        if cmd[0] == "latexmk":
            print(f"$ (cd paper && {' '.join(cmd)})")
            r = subprocess.run(cmd, cwd=os.path.join(REPO, "paper"))
        else:
            print(f"$ {' '.join(cmd)}")
            r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            print(f"task {name!r} failed at: {' '.join(cmd)} (exit {r.returncode})")
            return r.returncode
    print(f"task {name!r} done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
