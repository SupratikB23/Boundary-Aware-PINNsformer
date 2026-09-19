"""10_run_synthetic_sweep.py — E1 identifiability sweep driver (PRD §6 E1).

Sweeps station count × configuration × wind-entropy bin × noise with
replicates, using the FD synthetic solver as ground truth. CPU-first:
each cell runs a short forward solve and records the configured error
model; full inversion plugs in at `train/trainer.py` later. Writes one
tidy row per cell to results/e1_identifiability/.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from scripts._common import add_common_args, init_run  # noqa: E402


def main(argv=None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=(__doc__ or "").encode("ascii", "ignore").decode("ascii")))
    parser.add_argument("--quick", action="store_true", help="Single replicate, tiny grid (smoke test)")
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)

    import hashlib

    import numpy as np

    from bapinnsformer.data.synthetic import SyntheticSolver
    from bapinnsformer.eval.identifiability import cell_key, directional_entropy, fit_surface
    from bapinnsformer.eval.metrics import identifiability_margin, rel_l2
    from bapinnsformer.utils.io import ensure_dir, write_csv

    counts = [7] if args.quick else list(cfg.get("station_counts", [7, 15, 25, 40]))
    archetypes = ["uniform"] if args.quick else list(cfg.get("config_archetypes", ["uniform"]))
    noises = [0.0] if args.quick else list(cfg.get("noise_levels", [0.0, 0.05, 0.10, 0.20]))
    replicates = 1 if args.quick else int(cfg.get("replicates", 1))
    threshold = float(cfg.get("threshold", 0.25))

    out_dir = ensure_dir(os.path.join("results", str(cfg.get("results_subdir", "e1_identifiability"))))
    rows, cells = [], []
    solver = SyntheticSolver(nx=16, ny=16, dx=1000.0, dy=1000.0, dt=60.0, K=50.0, lam=1e-5)
    for n_st, arch, noise, rep in itertools.product(counts, archetypes, noises, range(replicates)):
        # Stable seed: sha256 of the cell key (Python hash() is salted per
        # process and would give different sweeps across runs).
        cell_seed = int(hashlib.sha256(f"{n_st}|{arch}|{noise}|{rep}".encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(cell_seed)
        C0 = rng.uniform(20, 80, (16, 16))
        wind_dir = rng.uniform(0, 2 * np.pi, 24)
        entropy = directional_entropy(wind_dir)
        Ctrue = solver.run(C0, 2.0, 0.5, S=1e-3, Cb={"west": 60.0}, n_steps=5)
        Cest = Ctrue + rng.normal(0, 1 + float(noise) * 50, Ctrue.shape)
        err = rel_l2(Cest, Ctrue)
        margin = identifiability_margin(err, err * 0.8, threshold=threshold)
        key = cell_key(n_st, arch, "*", float(noise))
        rows.append({"cell": key, "n_stations": n_st, "config": arch, "noise": noise,
                     "rep": rep, "entropy": entropy, "rel_l2": err, "margin": margin})
        cells.append({"n_stations": n_st, "config": arch, "entropy": float(entropy),
                      "noise": float(noise), "err": float(err)})
    write_csv(os.path.join(out_dir, f"sweep_{run_id}.csv"), rows)
    surface = fit_surface(cells) if len(cells) >= 6 else {"note": "too few cells for surface fit"}
    log.info("sweep complete: %d cells", len(rows))
    from scripts._common import save_results

    save_results(rundir, run_id, cfg, {"n_cells": len(rows), "surface": surface}, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
