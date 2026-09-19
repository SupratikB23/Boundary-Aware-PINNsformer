"""91_make_tables.py — regenerate every paper table (LaTeX) from results/.

PRD §4.4. Writes paper/tables/*.tex. Missing inputs are skipped with
a warning (exit 0).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from scripts._common import add_common_args, init_run  # noqa: E402


def main(argv=None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=(__doc__ or "").encode("ascii", "ignore").decode("ascii")))
    parser.add_argument("--tables-dir", default=os.path.join("paper", "tables"))
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)
    os.makedirs(args.tables_dir, exist_ok=True)
    made, skipped = [], []

    from bapinnsformer.utils.io import read_csv

    for path in glob.glob(os.path.join("results", "e1_identifiability", "*.csv")):
        try:
            rows = read_csv(path)
            tex = ["\\begin{tabular}{lrrrr}", "\\toprule",
                   "cell & n & noise & rel\\_l2 & margin \\\\", "\\midrule"]
            for r in rows[:50]:
                tex.append(f"{r.get('cell','--')} & {r.get('n_stations','--')} & "
                           f"{r.get('noise','--')} & {r.get('rel_l2','--')} & {r.get('margin','--')} \\\\")
            tex += ["\\bottomrule", "\\end{tabular}"]
            out = os.path.join(args.tables_dir, "e1_sweep.tex")
            with open(out, "w", encoding="utf-8") as fh:
                fh.write("\n".join(tex) + "\n")
            made.append(out)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{path}: {exc}")
            log.warning("skip %s: %s", path, exc)

    log.info("tables made=%d skipped=%d", len(made), len(skipped))
    from scripts._common import save_results

    save_results(rundir, run_id, cfg, {"made": made, "skipped": skipped}, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
