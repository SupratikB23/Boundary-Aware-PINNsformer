"""90_make_figures.py — regenerate every paper figure from results/.

PRD §4.4. Derived artifacts only (never hand-edited). Reads tidy
CSVs under results/*/ and writes paper/figures/. Missing inputs are
Missing inputs are skipped with a warning (exit 0) so `python tasks.py figures` never blocks writing.
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
    parser.add_argument("--figures-dir", default=os.path.join("paper", "figures"))
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)
    os.makedirs(args.figures_dir, exist_ok=True)
    made, skipped = [], []

    import numpy as np

    from bapinnsformer.viz import ablation_plots, attention_plots, boundary_plots

    # E3: accuracy vs L from any ablation CSV present (rows with numeric
    # rmse), plus wall-clock/memory vs L (available from timing-only rows).
    for path in glob.glob(os.path.join("results", "e3_ablation", "*.csv")):
        try:
            from bapinnsformer.utils.io import read_csv

            rows = read_csv(path)
            if rows and "L" in rows[0]:
                for r in rows:
                    r["L"] = int(float(r["L"]))
                    for k in ("rmse", "wall_s", "peak_mb"):
                        if k in r and r[k] not in (None, ""):
                            r[k] = float(r[k])
                acc_rows = [r for r in rows if isinstance(r.get("rmse"), float)]
                if acc_rows:
                    made.append(ablation_plots.accuracy_vs_length(
                        acc_rows, os.path.join(args.figures_dir, "fig08_accuracy_vs_L.png")))
                eff_rows = [r for r in rows if isinstance(r.get("wall_s"), float)]
                if eff_rows:
                    try:
                        made.append(ablation_plots.efficiency_vs_length(
                            eff_rows, os.path.join(args.figures_dir, "fig08b_efficiency_vs_L.png")))
                    except (AttributeError, TypeError):
                        # Older viz without efficiency_vs_length: wall-clock
                        # table is still in the CSV; skip the figure only.
                        skipped.append(f"{path}: efficiency figure unavailable")
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{path}: {exc}")
            log.warning("skip %s: %s", path, exc)

    # Placeholder Hovmöller so the figure pipeline is exercised end-to-end.
    try:
        Cb = np.maximum(0, 40 + 20 * np.sin(np.linspace(0, 3.14, 64))[:, None]
                        + 5 * np.random.default_rng(0).normal(size=(64, 48)))
        made.append(boundary_plots.hovmoller(Cb, os.path.join(args.figures_dir, "fig06_hovmoller_demo.png")))
        made.append(attention_plots.attention_over_tokens(
            np.array([0.5, 0.2, 0.12, 0.1, 0.08]),
            [False, True, False, True, False],
            os.path.join(args.figures_dir, "fig09_attention_demo.png")))
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"demo figures: {exc}")

    log.info("figures made=%d skipped=%d", len(made), len(skipped))
    from scripts._common import save_results

    save_results(rundir, run_id, cfg, {"made": made, "skipped": skipped}, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
