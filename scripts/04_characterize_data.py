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
    metrics = {"status": "stub", "outputs": ["missingness table", "wind agreement", "entropy bins"]}
    log.info("characterize stub: wire data/stats.py summaries + viz/maps.py here")
    from scripts._common import save_results

    save_results(rundir, run_id, cfg, metrics, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
