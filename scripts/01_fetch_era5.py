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
        log.info("fetch stub: wire cdsapi request here (variables per configs/data/era5.yaml)")
        metrics["note"] = "set CDS_API_KEY in .env; retrieval cached by (box, variable, month)"
    from scripts._common import save_results

    save_results(rundir, run_id, cfg, metrics, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
