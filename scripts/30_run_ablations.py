"""30_run_ablations.py — E3 pseudo-sequence / activation / physics ablations.

PRD §4.4/§6 E3. Variant × L grid at matched parameter counts with
wall-clock / peak-memory profiling (eval/profiling.py).
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from scripts._common import add_common_args, init_run  # noqa: E402


def main(argv=None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=(__doc__ or "").encode("ascii", "ignore").decode("ascii")))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)

    from bapinnsformer.eval.profiling import Timer, count_parameters, peak_memory_mb
    from bapinnsformer.models.pseudoseq import get_generator
    from bapinnsformer.utils.io import ensure_dir, write_csv
    from bapinnsformer.utils.registry import ACTIVATION, PSEUDOSEQ

    variants = list(cfg.get("variants", ["advection_backward"]))
    lengths = [3] if args.quick else list(cfg.get("sequence_lengths", [5]))
    out_dir = ensure_dir(os.path.join("results", str(cfg.get("results_subdir", "e3_ablation"))))
    rows = []
    wind = (2.0, 0.5)
    for variant, L in itertools.product(variants, lengths):
        PSEUDOSEQ.resolve(variant)  # fail fast on unknown variant names
        gen = get_generator(variant)
        with Timer() as t:
            tokens, hits = gen.generate(5000.0, 5000.0, 12.0, wind, L=int(L), dt=1.0,
                                        bounds=(0.0, 20000.0, 0.0, 20000.0))
        mem = peak_memory_mb()
        import numpy as _np

        tok = _np.asarray(tokens)
        # tokens are (B, L, 3): sequence length is axis 1, not len().
        n_tokens = int(tok.shape[1]) if tok.ndim == 3 else int(len(tok))
        peak = mem.get("cuda_peak_mb", mem.get("cpu_rss_peak_mb", float("nan")))
        try:
            peak = float(peak)
        except (TypeError, ValueError):
            peak = float("nan")
        import math as _math

        if not _math.isfinite(peak):
            # Windows has no `resource` module: fall back to tracemalloc peak.
            import tracemalloc as _tm

            if _tm.is_tracing():
                _, pk = _tm.get_traced_memory()
                peak = float(pk / (1024**2))
        rows.append({"variant": variant, "L": int(L), "wall_s": t.elapsed,
                      "peak_mb": peak, "n_tokens": n_tokens,
                      "rmse": "", "note": "timing-only; accuracy rows need a trained fit"})
    for act in cfg.get("activations", ["wavelet"]):
        ACTIVATION.resolve(act)  # fail fast on unknown activation names
    write_csv(os.path.join(out_dir, f"ablation_{run_id}.csv"), rows)
    log.info("ablation sweep: %d rows", len(rows))
    from scripts._common import save_results

    save_results(rundir, run_id, cfg, {"n_rows": len(rows)}, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
