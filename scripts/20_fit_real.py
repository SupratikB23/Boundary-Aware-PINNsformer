"""20_fit_real.py — E2 single-instance fit (city, season, pollutant).

PRD §4.4. Builds the full Trainer (C_net + Cb_net + S_net + phys) from the
composed config via utils.config.build_trainer_config — this validates the
YAML→constructor key mapping end-to-end. Without --smoke-fit it records the
composed run request + split hash (real data arrives in E0). With
--smoke-fit it runs a tiny CPU fit on a synthetic batch and writes a
checkpoint to results/runs/<run_id>/checkpoints/ (exercises the checkpoint
hook wiring). Fit on interior Delhi stations only; perimeter ring withheld.
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
    parser.add_argument("--city", default=None)
    parser.add_argument("--season", default=None)
    parser.add_argument("--pollutant", default=None)
    parser.add_argument("--smoke-fit", action="store_true",
                        help="Run a tiny CPU fit on a synthetic batch + checkpoint")
    parser.add_argument("--smoke-steps", type=int, default=3)
    args = parser.parse_args(argv)
    cfg, run_id, rundir, log = init_run(args)
    for key in ("city", "season", "pollutant"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    log.info("fit request: %s / %s / %s", cfg.get("city"), cfg.get("season"), cfg.get("pollutant"))

    from bapinnsformer.utils.config import build_trainer_config, normalize_season
    from bapinnsformer.train.trainer import Trainer

    cfg["season"] = normalize_season(cfg.get("season", "postmonsoon"))
    tcfg = build_trainer_config(cfg)
    trainer = Trainer(config=tcfg)
    n_params = sum(p.numel() for p in trainer.parameters() if p.requires_grad)
    log.info("trainer built: device=%s params=%d", trainer.device, n_params)
    metrics = {"instance": {k: cfg.get(k) for k in ("city", "season", "pollutant")},
               "n_params": n_params, "device": str(trainer.device),
               "status": "request-composed"}

    if args.smoke_fit:
        import torch

        from bapinnsformer.train.checkpoint import save_checkpoint

        ckpt_dir = os.path.join(rundir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        def _hook(step, state):
            save_checkpoint(os.path.join(ckpt_dir, f"step_{int(step):06d}.pt"),
                            trainer.C_net, trainer.Cb_net, trainer.S_net,
                            trainer.phys, trainer.normalizer,
                            tcfg, split_hash="", extra={"step": step})

        trainer.checkpoint_hook = _hook
        tcfg.setdefault("train", {})["checkpoint_every"] = 1
        torch.manual_seed(0)
        B = 8
        batch = {
            "x_data": torch.rand(B) * 1000, "y_data": torch.rand(B) * 1000,
            "t_data": torch.rand(B) * 3600, "C_obs": torch.rand(B) * 100 + 20,
            "mask": torch.ones(B),
            "x_pde": torch.rand(B) * 1000, "y_pde": torch.rand(B) * 1000,
            "t_pde": torch.rand(B) * 3600,
            "x_bc": torch.zeros(B), "y_bc": torch.rand(B) * 1000,
            "s_bc": torch.rand(B) * 100, "t_bc": torch.rand(B) * 3600,
            "nx": torch.full((B,), -1.0), "ny": torch.zeros(B),
        }
        hist = trainer.fit(batch, steps=int(args.smoke_steps))
        ckpts = sorted(os.listdir(ckpt_dir))
        metrics.update({"status": "smoke-fit", "final_total": hist[-1]["total"],
                        "checkpoints": ckpts})
        log.info("smoke fit done: total=%.4g checkpoints=%d", hist[-1]["total"], len(ckpts))

    from scripts._common import save_results

    save_results(rundir, run_id, cfg, metrics, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
