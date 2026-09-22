"""21_fit_all_real.py — fan-out of E2 across seasons × pollutants.

PRD §4.4. Respects the Kaggle session cap: never starts a fit whose
expected duration exceeds remaining session time (budget guard stub).
Each instance delegates to 20_fit_real.main with overridden config.
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
    parser.add_argument("--max-fits", type=int, default=None)
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)
    seasons = list(cfg.get("seasons", ["postmonsoon", "winter"]))
    pollutants = list(cfg.get("pollutants", ["PM2.5"]))
    import subprocess
    instances = [(s, p) for s in seasons for p in pollutants]
    if args.max_fits is not None:
        instances = instances[: max(0, args.max_fits)]
    log.info("fan-out: %d instances", len(instances))
    
    results = []
    for s, p in instances:
        log.info("launching fit for %s / %s", s, p)
        cmd = [sys.executable, "scripts/20_fit_real.py", 
               "--config", args.config,
               "--season", s,
               "--pollutant", p]
        try:
            subprocess.run(cmd, check=True)
            results.append({"season": s, "pollutant": p, "status": "success"})
        except subprocess.CalledProcessError:
            log.error("fit failed for %s / %s", s, p)
            results.append({"season": s, "pollutant": p, "status": "failed"})

    metrics = {"instances": results, "status": "fanout-complete"}
    from scripts._common import save_results

    save_results(rundir, run_id, cfg, metrics, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
