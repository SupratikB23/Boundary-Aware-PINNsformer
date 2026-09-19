"""03_build_dataset.py — QC → align → domain → splits → data/processed/.

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
    metrics = {"processed_dir": args.processed_dir, "status": "stub"}
    log.info("build stub: wire cpcb_ingest → qc → align → domain → splits here")
    from scripts._common import save_results

    save_results(rundir, run_id, cfg, metrics, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
