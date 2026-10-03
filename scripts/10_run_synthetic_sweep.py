"""10_run_synthetic_sweep.py — E1 identifiability sweep (PRD §6 E1), rewritten 2026-10-03.

Real inversion on synthetic truth (see ``bapinnsformer.eval.e1_benchmark``):
fine-grid truth -> noisy, gappy station observations -> Bayesian linear
inversion on a coarse Green's system (true and misspecified K/λ) ->
share / C_b / S errors, 90 % coverage, principal angles, confounding ρ.
Null tests (zero boundary / zero source) run in every sweep.

Wind windows come from ``--wind-npz`` (hourly ``u``, ``v`` arrays of shape
``(T, ny, nx)`` built from CPCB station wind, RUN_PLAN §G) or, if absent,
from a synthetic rotating-wind generator; the ``wind_source`` column says
which. Optional ``--pinn-cells N`` also fits the PINNsformer on N cells.

Outputs in ``results/e1_identifiability/``:
``sweep_<run>.csv`` (one row per cell), ``summary_<run>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from scripts._common import add_common_args, init_run, save_results  # noqa: E402


def _cell_seed(*parts) -> int:
    # sha256, not hash(): Python's hash is salted per process
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) % (2**31)


def _resample(F, n):
    """Bilinear resample ``(T, ny, nx)`` -> ``(T, n, n)``."""
    import numpy as np
    from scipy.ndimage import zoom

    F = np.asarray(F, dtype=float)
    return zoom(F, (1.0, n / F.shape[1], n / F.shape[2]), order=1, mode="nearest")


def _real_windows(path, H, n_windows, rng):
    """Random H-hour windows from a real hourly wind array, labelled by entropy tercile."""
    import numpy as np

    from bapinnsformer.eval.e1_benchmark import window_entropy

    with np.load(path) as z:
        U, V = z["u"], z["v"]
    if U.shape != V.shape or U.ndim != 3 or not np.all(np.isfinite(U)) or not np.all(np.isfinite(V)):
        raise ValueError("wind npz must hold finite u, v of equal shape (T, ny, nx)")
    if U.shape[0] < H + 1:
        raise ValueError(f"wind series shorter than one window ({U.shape[0]} < {H + 1} h)")
    starts = rng.choice(U.shape[0] - H, size=min(n_windows, U.shape[0] - H), replace=False)
    wins = [(int(s), U[s:s + H + 1], V[s:s + H + 1]) for s in starts]
    ent = np.array([window_entropy(u, v) for _, u, v in wins])
    q1, q2 = np.quantile(ent, [1 / 3, 2 / 3])
    lab = np.where(ent <= q1, "low", np.where(ent <= q2, "mid", "high"))
    return [(f"real@{s}", str(lb), u, v) for (s, u, v), lb in zip(wins, lab)]


def main(argv=None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description="E1 identifiability sweep"))
    parser.add_argument("--quick", action="store_true", help="tiny grid, 2 cells per kind (smoke test)")
    parser.add_argument("--wind-npz", default=None, help="hourly u/v (T,ny,nx) npz from station wind")
    parser.add_argument("--windows-per-bin", type=int, default=None, help="wind windows per entropy bin")
    parser.add_argument("--pinn-cells", type=int, default=0, help="also fit the PINNsformer on N cells")
    parser.add_argument("--pinn-steps", type=int, default=2000)
    parser.add_argument("--pinn-variant", default="advection_backward")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)

    import numpy as np

    from bapinnsformer.eval.e1_benchmark import (
        ARCHETYPES, E1Settings, place_stations, regrid_uniform, run_cell, run_pinn_cell,
        synthetic_wind_window, window_entropy,
    )
    from bapinnsformer.utils.io import ensure_dir, write_csv

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    sc = dict(cfg.get("settings", {}))
    st = E1Settings(**sc)
    if args.quick:
        st = E1Settings(n_coarse=9, n_hours=48, spinup_h=12)
    counts = [15] if args.quick else [int(c) for c in cfg.get("station_counts", [7, 15, 25, 40])]
    archs = ["clustered"] if args.quick else list(cfg.get("config_archetypes", list(ARCHETYPES)))
    noises = [0.05] if args.quick else [float(x) for x in cfg.get("noise_levels", [0.0, 0.05, 0.10, 0.20])]
    reps = 1 if args.quick else int(cfg.get("replicates", 3))
    bins = list(cfg.get("wind_entropy_bins", ["low", "mid", "high"]))
    wpb = args.windows_per_bin or (1 if args.quick else int(cfg.get("windows_per_bin", 2)))
    kinds = ["full"] + list(cfg.get("null_tests", ["null_boundary", "null_source"]))
    n_rec = int(cfg.get("n_receptors", 8))
    shear = float(cfg.get("wind_shear", 0.2))
    nf = (st.n_coarse - 1) * st.fine_factor + 1

    # ---- wind windows ----
    wrng = np.random.default_rng(_cell_seed(seed, "wind"))
    windows = []  # (wind_id, bin, U_coarse, V_coarse, U_fine, V_fine)
    if args.wind_npz:
        wind_source = "station_interp"
        raw = _real_windows(args.wind_npz, st.n_hours, 60 * wpb, wrng)
        for b in bins:
            for wid, lb, u, v in [w for w in raw if w[1] == b][:wpb]:
                windows.append((wid, lb, _resample(u, st.n_coarse), _resample(v, st.n_coarse),
                                _resample(u, nf), _resample(v, nf)))
    else:
        wind_source = "synthetic"
        log.warning("no --wind-npz: using SYNTHETIC rotating wind (flagged in every row)")
        for b in bins:
            for k in range(wpb):
                u, v = synthetic_wind_window(st.n_hours, b, wrng)
                Uc, Vc = regrid_uniform(u, v, st.n_coarse, st.n_coarse, shear)
                Uf, Vf = regrid_uniform(u, v, nf, nf, shear)
                windows.append((f"syn_{b}_{k}", b, Uc, Vc, Uf, Vf))

    # receptors: fixed Delhi-like core (the quantity a paper would report)
    receptors = place_stations(n_rec, "clustered", st.Lx, st.Ly, np.random.default_rng(_cell_seed(seed, "rec")))

    out_dir = ensure_dir(os.path.join(args.results_root or cfg.get("results_root", "results"),
                                      str(cfg.get("results_subdir", "e1_identifiability"))))
    rows, pinn_done = [], 0
    grid = list(itertools.product(windows, counts, archs, noises, range(reps), kinds))
    log.info("E1: %d cells (%d windows, wind=%s)", len(grid), len(windows), wind_source)
    for (wid, b, Uc, Vc, Uf, Vf), n, arch, noise, rep, kind in grid:
        srng = np.random.default_rng(_cell_seed(seed, "st", n, arch, rep))
        stations = place_stations(n, arch, st.Lx, st.Ly, srng)
        crng = np.random.default_rng(_cell_seed(seed, wid, n, arch, noise, rep, kind))
        r = run_cell(st, Uc, Vc, Uf, Vf, stations, receptors, noise, crng, kind=kind,
                     misspec=bool(cfg.get("misspec", True)))
        r.update({"wind_id": wid, "wind_source": wind_source, "entropy_bin": b,
                  "entropy": window_entropy(Uc, Vc), "n_stations": n, "config": arch,
                  "noise": noise, "rep": rep})
        if kind == "full" and pinn_done < args.pinn_cells:
            r.update(run_pinn_cell(st, r, Uf, Vf, stations, receptors, variant=args.pinn_variant,
                                   steps=args.pinn_steps, seed=seed + pinn_done, device=args.device))
            pinn_done += 1
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        log.info("%s n=%d %s noise=%.2f rep=%d %s share true=%.3f est=%.3f",
                 wid, n, arch, noise, rep, kind, r["share_true"], r["oracle_share_est"])

    csv_path = os.path.join(out_dir, f"sweep_{run_id}.csv")
    write_csv(csv_path, rows)

    # ---- summary: identifiability surface + null-test pass rates ----
    summary = {"n_cells": len(rows), "wind_source": wind_source, "csv": csv_path,
               "null_tol": float(cfg.get("null_tol", 0.10))}
    full = [r for r in rows if r["kind"] == "full"]
    if len(full) >= 3:
        from scipy.stats import spearmanr

        e = np.array([r["entropy"] for r in full])
        err = np.array([r["oracle_share_err_vs_fine"] for r in full])
        rho = np.array([r["oracle_confounding_rho"] for r in full])
        summary["spearman_entropy_vs_share_err"] = float(spearmanr(e, err).correlation)
        summary["spearman_abs_rho_vs_share_err"] = float(spearmanr(np.abs(rho), err).correlation)
        summary["coverage90_vs_fine"] = float(np.mean([r["oracle_covered90_vs_fine"] for r in full]))
    tol = summary["null_tol"]
    by = {}
    for r in rows:
        if r["kind"] == "full":
            continue
        target = 0.0 if r["kind"] == "null_boundary" else 1.0
        key = f'{r["kind"]}|n={r["n_stations"]}|{r["config"]}|{r["entropy_bin"]}'
        by.setdefault(key, []).append(abs(r["oracle_share_est"] - target) <= tol)
    summary["null_pass_rate"] = {k: float(np.mean(v)) for k, v in sorted(by.items())}
    with open(os.path.join(out_dir, f"summary_{run_id}.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    save_results(rundir, run_id, cfg, {k: v for k, v in summary.items() if k != "null_pass_rate"}, log=log)
    log.info("E1 done: %s", json.dumps({k: v for k, v in summary.items() if k != "null_pass_rate"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
