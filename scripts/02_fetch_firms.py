"""02_fetch_firms.py — FIRMS VIIRS/MODIS fire+FRP into data/heldout/ ONLY.

PRD §4.4 + leakage guard (§10): this is the ONLY script allowed to
touch `data/heldout/`. Nothing under src/bapinnsformer/{models,
physics, train} may import it or read its outputs (tests/test_leakage.py).
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

HELDOUT_DIR = os.path.join("data", "heldout", "firms")


def main(argv=None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=(__doc__ or "").encode("ascii", "ignore").decode("ascii")))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)
    os.makedirs(HELDOUT_DIR, exist_ok=True)
    metrics = {"heldout_dir": HELDOUT_DIR, "offline": bool(args.offline)}
    
    from scripts._common import save_results
    import urllib.request
    from urllib.error import HTTPError
    from bapinnsformer.data.firms_ingest import fetch_firms_archive
    
    if args.offline:
        import glob
        metrics["heldout_files"] = sorted(glob.glob(os.path.join(HELDOUT_DIR, "*")))
    else:
        # Get FIRMS API KEY
        api_key = os.environ.get("FIRMS_API_KEY")
        if not api_key and os.path.exists(".env"):
            with open(".env") as f:
                for line in f:
                    if line.startswith("FIRMS_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
                        
        if not api_key:
            log.error("FIRMS_API_KEY not found in .env")
            metrics["error"] = "FIRMS_API_KEY not found"
        else:
            log.info("Fetching NASA FIRMS data...")
            # Area coordinates: W,S,E,N - roughly Punjab and Haryana
            box = "73,28,78,32" 
            source = "VIIRS_SNPP_NRT"
            days = "10" # maximum allowed by FIRMS
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{box}/{days}"
            
            dest_file = os.path.join(HELDOUT_DIR, "firms_punjab_haryana.csv")
            try:
                out_path = fetch_firms_archive(url, dest_file)
                log.info(f"FIRMS data saved to {out_path}")
                metrics["success"] = True
                metrics["outfile"] = str(out_path)
            except Exception as e:
                log.error(f"Failed to fetch FIRMS data: {e}")
                metrics["error"] = str(e)

    save_results(rundir, run_id, cfg, metrics, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
