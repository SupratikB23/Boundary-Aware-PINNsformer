"""20_fit_real.py — E2 single-instance fit (city, season, pollutant).

PRD §4.4. Builds the full Trainer (C_net + Cb_net + S_net + phys) from the
composed config via utils.config.build_trainer_config. Loads real CPCB data
from data/processed/ and, if available, ERA5 wind fields from data/raw/era5/.
Fit on interior Delhi stations only; perimeter ring withheld.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from scripts._common import add_common_args, init_run  # noqa: E402


def _load_wind_field(dom, t_bounds, era5_dir, log):
    """Try to load ERA5 wind data and build a WindField, else return constant zero."""
    from bapinnsformer.data.wind_field import WindField

    nc_files = sorted(glob.glob(os.path.join(era5_dir, "era5_*.nc")))
    if not nc_files:
        log.warning("No ERA5 NetCDF files found in %s. Using zero wind.", era5_dir)
        x_coords = np.linspace(0.0, dom.Lx, 3)
        y_coords = np.linspace(0.0, dom.Ly, 3)
        t_coords = np.array([t_bounds[0], t_bounds[1]])
        return WindField.constant(0.0, 0.0, x_coords, y_coords, t_coords)

    # Try loading with xarray
    try:
        import xarray as xr
        ds = xr.open_mfdataset(nc_files, combine="by_coords")
        # ERA5 variable names: u10 / v10 (or 10m_u_component_of_wind etc.)
        u_name = "u10" if "u10" in ds else "10m_u_component_of_wind"
        v_name = "v10" if "v10" in ds else "10m_v_component_of_wind"
        if u_name not in ds or v_name not in ds:
            log.warning("ERA5 files lack u10/v10 variables. Keys: %s. Using zero wind.", list(ds.data_vars))
            x_coords = np.linspace(0.0, dom.Lx, 3)
            y_coords = np.linspace(0.0, dom.Ly, 3)
            t_coords = np.array([t_bounds[0], t_bounds[1]])
            return WindField.constant(0.0, 0.0, x_coords, y_coords, t_coords)

        # ERA5 grids are in lat/lon; project to metric
        from bapinnsformer.data.align import project_to_metric
        era5_lons = ds.longitude.values if "longitude" in ds.coords else ds.lon.values
        era5_lats = ds.latitude.values if "latitude" in ds.coords else ds.lat.values
        x_coords, _ = project_to_metric(era5_lons, np.full_like(era5_lons, dom.lat0), dom.lon0, dom.lat0)
        _, y_coords = project_to_metric(np.full_like(era5_lats, dom.lon0), era5_lats, dom.lon0, dom.lat0)
        # Sort ascending (ERA5 lat is often descending)
        y_sort = np.argsort(y_coords)
        y_coords = y_coords[y_sort]

        epoch = pd.Timestamp("2015-01-01", tz="UTC")
        t_coords = (pd.to_datetime(ds.time.values, utc=True) - epoch).total_seconds().values

        u_data = ds[u_name].values  # (Nt, Nlat, Nlon)
        v_data = ds[v_name].values
        # Re-order lat axis if it was descending
        u_data = u_data[:, y_sort, :]
        v_data = v_data[:, y_sort, :]
        ds.close()

        log.info("Loaded ERA5 wind field: %d times x %d lats x %d lons", *u_data.shape)
        return WindField(u_data, v_data, x_coords, y_coords, t_coords)

    except Exception as e:
        log.warning("Failed to load ERA5 wind: %s. Using zero wind.", e)
        x_coords = np.linspace(0.0, dom.Lx, 3)
        y_coords = np.linspace(0.0, dom.Ly, 3)
        t_coords = np.array([t_bounds[0], t_bounds[1]])
        return WindField.constant(0.0, 0.0, x_coords, y_coords, t_coords)


def main(argv=None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=(__doc__ or "").encode("ascii", "ignore").decode("ascii")))
    parser.add_argument("--city", default=None)
    parser.add_argument("--season", default=None)
    parser.add_argument("--pollutant", default=None)
    parser.add_argument("--smoke-fit", action="store_true",
                        help="Run a tiny CPU fit on a synthetic batch + checkpoint")
    parser.add_argument("--smoke-steps", type=int, default=3)
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)
    for key in ("city", "season", "pollutant"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    log.info("fit request: %s / %s / %s", cfg.get("city"), cfg.get("season"), cfg.get("pollutant"))

    from bapinnsformer.utils.config import build_trainer_config, normalize_season
    from bapinnsformer.train.trainer import Trainer
    from bapinnsformer.data.domain import build_domain, xy_to_metric, boundary_points
    from bapinnsformer.models.normalizer import Normalizer
    from bapinnsformer.train.checkpoint import save_checkpoint

    cfg["season"] = normalize_season(cfg.get("season", "postmonsoon"))

    # ── 1. Load processed data ──────────────────────────────────────────
    processed_dir = os.path.join("data", "processed")
    try:
        cpcb = pd.read_parquet(os.path.join(processed_dir, "cpcb_data.parquet"))
    except Exception:
        log.error("Processed data not found. Please run E0 (tasks.py data) first.")
        return 1

    # Filter pollutant
    cpcb = cpcb[cpcb["pollutant"] == cfg.get("pollutant", "PM2.5")]
    if cpcb.empty:
        log.error("No data found for requested pollutant.")
        return 1

    # Filter season
    cpcb["month"] = pd.to_datetime(cpcb["time"]).dt.month
    season = cfg.get("season", "postmonsoon")
    if season == "postmonsoon":
        cpcb = cpcb[cpcb["month"].isin([10, 11])]
    elif season == "winter":
        cpcb = cpcb[cpcb["month"].isin([12, 1, 2])]

    # ── 2. Apply interior/perimeter split ───────────────────────────────
    split_path = os.path.join(processed_dir, "split_primary.json")
    split_dict = {}                       # safe default
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            split_dict = json.load(f)
        fit_ids = set(split_dict.get("fit_ids", []))
        if fit_ids:
            cpcb_fit = cpcb[cpcb["station_id"].isin(fit_ids)]
        else:
            cpcb_fit = cpcb
    else:
        log.warning("split_primary.json not found — using all stations.")
        cpcb_fit = cpcb

    if cpcb_fit.empty:
        log.error("No data found for requested season/pollutant in fit split.")
        return 1

    # ── 3. Map coordinates and time ─────────────────────────────────────
    d_cfg = cfg.get("domain", {})
    dom = build_domain(
        lon_min=d_cfg.get("lon_min", 76.40), lon_max=d_cfg.get("lon_max", 77.90),
        lat_min=d_cfg.get("lat_min", 27.80), lat_max=d_cfg.get("lat_max", 29.20),
    )

    x_obs = xy_to_metric(cpcb_fit["longitude"], cpcb_fit["latitude"], dom)[0]
    y_obs = xy_to_metric(cpcb_fit["longitude"], cpcb_fit["latitude"], dom)[1]
    epoch = pd.Timestamp("2015-01-01", tz="UTC")
    t_obs = (pd.to_datetime(cpcb_fit["time"], utc=True) - epoch).dt.total_seconds().values
    v_obs = cpcb_fit["value"].values

    # ── 4. Normalizer + wind field + Trainer ────────────────────────────
    t_bounds = (float(t_obs.min()), float(t_obs.max()))
    norm = Normalizer.fit_from_data(
        x=np.array([0.0, dom.Lx]), y=np.array([0.0, dom.Ly]),
        t=np.array(t_bounds), C=v_obs, s=np.array([0.0, dom.P]),
    )

    era5_dir = cfg.get("era5", {}).get("cache_dir", os.path.join("data", "raw", "era5"))
    wf = _load_wind_field(dom, t_bounds, era5_dir, log)

    tcfg = build_trainer_config(cfg)
    trainer = Trainer(config=tcfg, normalizer=norm, wind_field=wf)
    n_params = sum(p.numel() for p in trainer.parameters() if p.requires_grad)
    log.info("trainer built: device=%s params=%d", trainer.device, n_params)

    # ── 5. Checkpoint hook ──────────────────────────────────────────────
    ckpt_dir = os.path.join(rundir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    split_hash = split_dict.get("hash", "")

    def _hook(step, state):
        save_checkpoint(
            os.path.join(ckpt_dir, f"step_{int(step):06d}.pt"),
            trainer.C_net, trainer.Cb_net, trainer.S_net,
            trainer.phys, trainer.normalizer,
            tcfg, split_hash=split_hash, extra={"step": step},
        )

    trainer.checkpoint_hook = _hook

    # ── 6. Batch generator ──────────────────────────────────────────────
    def batch_gen(step: int) -> dict:
        B = int(tcfg.get("train", {}).get("batch_size", 1024))

        # Data points (random sample with replacement)
        idx = np.random.randint(0, len(x_obs), size=B)
        x_d, y_d, t_d, c_d = x_obs[idx], y_obs[idx], t_obs[idx], v_obs[idx]

        # PDE collocation points (uniform interior)
        x_p = np.random.uniform(0.0, dom.Lx, size=B)
        y_p = np.random.uniform(0.0, dom.Ly, size=B)
        t_p = np.random.uniform(t_bounds[0], t_bounds[1], size=B)

        # Boundary collocation points
        s_b, x_b, y_b, n_b = boundary_points(B, dom)
        t_b = np.random.uniform(t_bounds[0], t_bounds[1], size=B)

        return {
            "x_data": torch.tensor(x_d, dtype=torch.float32),
            "y_data": torch.tensor(y_d, dtype=torch.float32),
            "t_data": torch.tensor(t_d, dtype=torch.float32),
            "C_obs":  torch.tensor(c_d, dtype=torch.float32),
            "mask":   torch.ones(B, dtype=torch.float32),
            "x_pde":  torch.tensor(x_p, dtype=torch.float32),
            "y_pde":  torch.tensor(y_p, dtype=torch.float32),
            "t_pde":  torch.tensor(t_p, dtype=torch.float32),
            "x_bc":   torch.tensor(x_b, dtype=torch.float32),
            "y_bc":   torch.tensor(y_b, dtype=torch.float32),
            "s_bc":   torch.tensor(s_b, dtype=torch.float32),
            "t_bc":   torch.tensor(t_b, dtype=torch.float32),
            "nx":     torch.tensor(n_b[:, 0], dtype=torch.float32),
            "ny":     torch.tensor(n_b[:, 1], dtype=torch.float32),
        }

    # ── 7. Fit ──────────────────────────────────────────────────────────
    steps = int(args.smoke_steps) if args.smoke_fit else int(tcfg.get("train", {}).get("steps", 50))
    hist = trainer.fit(batch_gen, steps=steps)

    ckpts = sorted(os.listdir(ckpt_dir))
    final_loss = hist[-1]["total"] if hist else float("nan")
    metrics = {
        "instance": {k: cfg.get(k) for k in ("city", "season", "pollutant")},
        "n_params": n_params,
        "device": str(trainer.device),
        "status": "smoke-fit" if args.smoke_fit else "fit-complete",
        "final_total": final_loss,
        "checkpoints": ckpts,
    }
    log.info("fit done: total=%.4g checkpoints=%d", final_loss, len(ckpts))

    from scripts._common import save_results
    save_results(rundir, run_id, cfg, metrics, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
