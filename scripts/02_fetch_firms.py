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
    if args.offline:
        import glob

        metrics["heldout_files"] = sorted(glob.glob(os.path.join(HELDOUT_DIR, "*")))
    else:
        log.info("fetch stub: wire FIRMS API here; outputs stay under data/heldout/firms/")
        metrics["note"] = "set FIRMS_API_KEY in .env"
    from scripts._common import save_results

    save_results(rundir, run_id, cfg, metrics, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
