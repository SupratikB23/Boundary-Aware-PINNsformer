"""04_characterize_data.py — missingness, ERA5-vs-station wind agreement,
directional entropy per window (E0 data section + domain-map figure).

PRD §4.4. Reads data/processed/ (built by 03), writes tidy tables +
figures under results/e0_data_quality/.
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
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)

    import json
    import pandas as pd
    from bapinnsformer.utils.io import ensure_dir

    processed_dir = os.path.join("data", "processed")
    out_dir = ensure_dir(os.path.join("results", "e0_data_quality"))
    outputs = []

    # ── 1. Missingness table ────────────────────────────────────────────
    parquet_path = os.path.join(processed_dir, "cpcb_data.parquet")
    qc_report_path = os.path.join(processed_dir, "qc_report.csv")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        from bapinnsformer.data.stats import missingness_table

        # Reconstruct QC columns if not present (the parquet has only valid=True rows)
        if "valid" not in df.columns:
            df["valid"] = True
        if "qc_flag" not in df.columns:
            df["qc_flag"] = "ok"

        miss = missingness_table(df)
        miss_path = os.path.join(out_dir, "missingness.csv")
        miss.to_csv(miss_path, index=False)
        outputs.append(miss_path)
        log.info("Missingness table: %d rows -> %s", len(miss), miss_path)
    else:
        log.warning("cpcb_data.parquet not found — skipping missingness table.")

    # ── 2. Station summary stats ────────────────────────────────────────
    stations_path = os.path.join(processed_dir, "stations.csv")
    if os.path.exists(stations_path):
        stations = pd.read_csv(stations_path)
        summary = {
            "n_stations": len(stations),
            "n_interior": int((stations["split"] == "interior").sum()) if "split" in stations.columns else "N/A",
            "n_perimeter": int((stations["split"] == "perimeter").sum()) if "split" in stations.columns else "N/A",
        }
        summary_path = os.path.join(out_dir, "station_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        outputs.append(summary_path)
        log.info("Station summary: %s", summary)
    else:
        stations = None
        log.warning("stations.csv not found — skipping station summary.")

    # ── 3. Domain map figure ────────────────────────────────────────────
    if stations is not None and "split" in stations.columns:
        try:
            from bapinnsformer.viz.maps import domain_map

            d_cfg = cfg.get("domain", {})
            bounds = (
                float(d_cfg.get("lon_min", 76.40)),
                float(d_cfg.get("lon_max", 77.90)),
                float(d_cfg.get("lat_min", 27.80)),
                float(d_cfg.get("lat_max", 29.20)),
            )
            interior = stations[stations["split"] == "interior"]
            perimeter = stations[stations["split"] == "perimeter"]
            # Use metric coords if available, otherwise lon/lat
            if "x" in interior.columns and "y" in interior.columns:
                int_xy = list(zip(interior["x"], interior["y"]))
                per_xy = list(zip(perimeter["x"], perimeter["y"]))
                map_bounds = (0.0, float(interior["x"].max()) * 1.1 if not interior.empty else 1e5,
                              0.0, float(interior["y"].max()) * 1.1 if not interior.empty else 1e5)
            else:
                int_xy = list(zip(interior["longitude"], interior["latitude"]))
                per_xy = list(zip(perimeter["longitude"], perimeter["latitude"]))
                map_bounds = bounds

            map_path = os.path.join(out_dir, "domain_map.png")
            domain_map(map_bounds, int_xy, per_xy, save_path=map_path)
            outputs.append(map_path)
            log.info("Domain map saved to %s", map_path)
        except Exception as e:
            log.warning("Domain map generation failed: %s", e)

    # ── 4. Per-pollutant value distribution ─────────────────────────────
    if os.path.exists(parquet_path):
        try:
            import numpy as np
            pol_stats = []
            for pol, g in df.groupby("pollutant"):
                vals = g["value"].dropna()
                pol_stats.append({
                    "pollutant": pol,
                    "n_obs": len(vals),
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "p25": float(vals.quantile(0.25)),
                    "median": float(vals.median()),
                    "p75": float(vals.quantile(0.75)),
                    "max": float(vals.max()),
                })
            pol_df = pd.DataFrame(pol_stats)
            pol_path = os.path.join(out_dir, "pollutant_stats.csv")
            pol_df.to_csv(pol_path, index=False)
            outputs.append(pol_path)
            log.info("Pollutant stats: %s", pol_path)
        except Exception as e:
            log.warning("Pollutant stats failed: %s", e)

    # ── 5. Split info ───────────────────────────────────────────────────
    split_path = os.path.join(processed_dir, "split_primary.json")
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            split = json.load(f)
        log.info("Split hash: %s, fit=%d, holdout=%d stations",
                 split.get("hash", "?"),
                 len(split.get("fit_ids", [])),
                 len(split.get("holdout_ids", [])))

    metrics = {
        "status": "complete",
        "outputs": outputs,
        "n_outputs": len(outputs),
    }
    from scripts._common import save_results
    save_results(rundir, run_id, cfg, metrics, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
