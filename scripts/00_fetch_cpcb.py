"""00_fetch_cpcb.py — pull/refresh CPCB + OpenCity raw data into data/raw/.

PRD §4.4. Network-enabled; writes immutable timestamped snapshots plus
a manifest with checksums. Offline stub: with --offline, writes the
manifest skeleton only.
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
    parser.add_argument("--offline", action="store_true", help="Skip network; manifest only")
    parser.add_argument("--mirror", default="both", choices=["ccr", "opencity", "both"])
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)
    manifest = {"mirror": args.mirror, "offline": bool(args.offline), "files": []}
    if args.offline:
        log.info("offline mode: no download attempted")
    else:
        log.info("fetch stub: wire CPCB CCR / OpenCity endpoints here (see data/README)")
        manifest["note"] = "live fetch not yet wired; drop portal exports into data/raw/cpcb + data/raw/opencity"
    from scripts._common import save_results

    save_results(rundir, run_id, cfg, {"manifest": manifest}, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
