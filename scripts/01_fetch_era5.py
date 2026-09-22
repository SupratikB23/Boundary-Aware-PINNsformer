"""01_fetch_era5.py — CDS retrieval of u10/v10/BLH/T/RH into data/raw/era5/.

PRD §4.4. Requires CDS API credentials in .env (never committed).
Offline stub: with --offline, verifies cache layout only.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from scripts._common import add_common_args, init_run  # noqa: E402


def main(argv=None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=(__doc__ or "").encode("ascii", "ignore").decode("ascii")))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)
    cache = os.path.join("data", "raw", "era5")
    os.makedirs(cache, exist_ok=True)
    metrics = {"cache_dir": cache, "offline": bool(args.offline)}
    if args.offline:
        import glob

        metrics["cached_files"] = sorted(glob.glob(os.path.join(cache, "*")))
    else:
        from bapinnsformer.data.era5_ingest import fetch_era5
        d_cfg = cfg.get("domain", {})
        box = (
            float(d_cfg.get("lon_min", 76.40)),
            float(d_cfg.get("lon_max", 77.90)),
            float(d_cfg.get("lat_min", 27.80)),
            float(d_cfg.get("lat_max", 29.20)),
        )
        e_cfg = cfg.get("era5", {})
        variables = e_cfg.get("variables", [
            "10m_u_component_of_wind", "10m_v_component_of_wind", "boundary_layer_height"
        ])
        
        # Determine years and months to fetch. For demo/Kaggle, default to 2019-2020 postmonsoon/winter.
        years = list(e_cfg.get("years", [2019, 2020]))
        months = list(e_cfg.get("months", [1, 2, 10, 11, 12]))
        
        log.info("Fetching ERA5 for box=%s, years=%s, months=%s", box, years, months)
        cached_files = []
        for y in years:
            for m in months:
                log.info("Fetching ERA5 %04d-%02d...", y, m)
                try:
                    path = fetch_era5(box, variables, year=y, month=m, cache_dir=cache)
                    cached_files.append(str(path))
                    log.info("-> %s", path)
                except Exception as e:
                    log.error("Fetch failed for %04d-%02d: %s", y, m, e)
                    
        metrics["cached_files"] = cached_files
        metrics["note"] = "Fetched via CDS API"
    from scripts._common import save_results

    save_results(rundir, run_id, cfg, metrics, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
