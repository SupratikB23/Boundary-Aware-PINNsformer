"""50_transfer_airshed.py — E5 second-airshed run (PRD §6 E5).

Applies the identical pipeline with only a domain/*.yaml override
(Kolkata/Chennai). Results reported honestly even if weaker.
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
    parser.add_argument("--domain", default=None, help="Domain yaml, e.g. configs/domain/kolkata.yaml")
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)
    cities = list(cfg.get("cities", ["kolkata"]))
    if args.domain:
        from bapinnsformer.utils.io import load_config

        dom = load_config(args.domain)
        cities = [dom.get("name", args.domain)]
    log.info("transfer airsheds: %s", cities)
    from scripts._common import save_results

    save_results(rundir, run_id, cfg, {"cities": cities, "status": "transfer-stub"}, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
