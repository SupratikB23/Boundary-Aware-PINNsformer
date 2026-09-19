"""40_validate_fires.py — E4 recovered NW inflow vs withheld VIIRS FRP.

PRD §4.4/§6 E4 + leakage guard: reads ONLY results/ artifacts (recovered
inflow) and data/heldout/firms (fire). Never writes to training inputs.
Reports Spearman correlation with lag analysis + external divergence table.
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
    parser.add_argument("--inflow-csv", default=None, help="Recovered NW inflow daily series CSV")
    parser.add_argument("--frp-csv", default=None, help="Withheld FRP daily series CSV (data/heldout/...)")
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)
    metrics: dict = {"status": "no-inputs", "note": "pass --inflow-csv + --frp-csv for the correlation"}
    if args.inflow_csv and args.frp_csv:
        from bapinnsformer.eval.metrics import spearman_with_p
        from bapinnsformer.utils.io import read_csv

        def series(path, key):
            rows = read_csv(path)
            cols = rows[0].keys() if rows else []
            col = key if key in cols else (list(cols)[-1] if cols else None)
            return [float(r[col]) for r in rows]

        q = series(args.inflow_csv, "nw_inflow")
        f = series(args.frp_csv, "frp")
        n = min(len(q), len(f))
        lags = [int(v) for v in cfg.get("lags_days", [0])]
        best = {}
        for lag in lags:
            qq, ff = q[:n], f[:n]
            if lag:
                # Lagged correlation: inflow day i vs fire day i-lag.
                # Trim (no circular wrap — wrapping would leak future
                # fires into the past).
                lag = int(lag)
                if lag >= n:
                    continue
                qq, ff = q[lag:n], f[: n - lag]
            rho, p = spearman_with_p(qq, ff)
            best[str(lag)] = {"rho": rho, "p": p, "n": len(qq)}
        metrics = {"n_days": n, "lags": best}
        log.info("E4 lags: %s", best)
    from scripts._common import save_results

    save_results(rundir, run_id, cfg, metrics, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
