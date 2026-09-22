"""03_build_dataset.py — QC -> align -> domain -> splits -> data/processed/.

PRD §4.4 (E0 deliverable). Emits the data-quality report skeleton and
a content-hashed split definition. Masked loss downstream: gaps are
carried as validity masks, never imputed.
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
    parser.add_argument("--processed-dir", default=os.path.join("data", "processed"))
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)
    os.makedirs(args.processed_dir, exist_ok=True)
    
    import pandas as pd
    from bapinnsformer.data import cpcb_ingest, qc, domain, splits

    log.info("Ingesting CPCB data from data/raw/cpcb")
    df_cpcb = cpcb_ingest.ingest_cpcb_dir(os.path.join("data", "raw", "cpcb"))
    
    # Check if opencity data exists and ingest it
    opencity_dir = os.path.join("data", "raw", "opencity")
    if os.path.exists(opencity_dir) and any(f.endswith(".csv") for f in os.listdir(opencity_dir)):
        log.info("Ingesting OpenCity data")
        df_opencity = cpcb_ingest.ingest_cpcb_dir(opencity_dir, source="opencity")
        df = pd.concat([df_cpcb, df_opencity], ignore_index=True)
    else:
        df = df_cpcb

    if df.empty:
        log.error("No data found in raw directories. Aborting.")
        return 1
        
    log.info(f"Loaded {len(df)} rows. Applying QC...")
    # QC params from config if available
    qc_cfg = cfg.get("cpcb", {}).get("qc", {})
    df_qc = qc.apply_qc(
        df,
        flatline_window=qc_cfg.get("flatline_hours", 6),
        spike_z=qc_cfg.get("spike_zscore", 6.0)
    )
    
    # Inject coordinates from json for Kaggle dataset
    coords_path = os.path.join("data", "raw", "cpcb", "station_coords.json")
    if os.path.exists(coords_path):
        import json
        with open(coords_path, 'r') as f:
            coords = json.load(f)
        
        # Add lat, lon, name if missing
        if "latitude" not in df_qc.columns:
            df_qc["latitude"] = df_qc["station_id"].map(lambda x: coords.get(x, {}).get("latitude"))
            df_qc["longitude"] = df_qc["station_id"].map(lambda x: coords.get(x, {}).get("longitude"))
            df_qc["station_name"] = df_qc["station_id"].map(lambda x: coords.get(x, {}).get("station_name", x))
    
    log.info("Generating QC report")
    report = qc.qc_report(df_qc)
    report_path = os.path.join(args.processed_dir, "qc_report.csv")
    report.to_csv(report_path, index=False)
    
    log.info("Extracting valid station list")
    stations = df_qc[["station_id", "station_name", "latitude", "longitude"]].drop_duplicates("station_id")
    # Some rows might not have lat/lon in raw data, let's keep only those with coordinates
    stations = stations.dropna(subset=["latitude", "longitude"])
    
    log.info("Building domain and classifying stations")
    d_cfg = cfg.get("domain", {})
    dom = domain.build_domain(
        lon_min=d_cfg.get("lon_min", 76.40),
        lon_max=d_cfg.get("lon_max", 77.90),
        lat_min=d_cfg.get("lat_min", 27.80),
        lat_max=d_cfg.get("lat_max", 29.20),
        crs_epsg=None
    )
    
    buf_m = float(d_cfg.get("perimeter_buffer_km", 15.0)) * 1000.0
    stations = domain.classify_stations(stations, dom, buffer_m=buf_m)
    
    if (stations["split"] == "perimeter").sum() == 0:
        import numpy as np
        log.warning("No perimeter stations found using buffer. Falling back to median distance split.")
        med_dist = stations["dist_to_bdy"].median()
        stations["split"] = np.where(stations["dist_to_bdy"] <= med_dist, "perimeter", "interior")
    
    log.info("Generating splits")
    split_dict = splits.interior_perimeter_split(stations)
    split_path = os.path.join(args.processed_dir, "split_primary.json")
    splits.save_split(split_dict, split_path)
    
    log.info("Saving processed datasets")
    # Only keep valid data points
    valid_data = df_qc[df_qc["valid"] == True]
    
    valid_data.to_parquet(os.path.join(args.processed_dir, "cpcb_data.parquet"), index=False)
    stations.to_csv(os.path.join(args.processed_dir, "stations.csv"), index=False)
    
    metrics = {
        "processed_dir": args.processed_dir,
        "n_raw_rows": len(df),
        "n_valid_rows": len(valid_data),
        "n_stations": len(stations),
        "split_hash": split_dict.get("hash"),
        "status": "complete"
    }
    from scripts._common import save_results
    save_results(rundir, run_id, cfg, metrics, log=log)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
