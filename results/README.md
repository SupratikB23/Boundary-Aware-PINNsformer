# results/ — provenance note (PRD §4.7, §10)

```
results/
├── e0_data_quality/
├── e1_identifiability/
├── e2_real/
├── e3_ablation/
├── e4_validation/
├── e5_transfer/
└── runs/<run_id>/  metrics.json (record + embedded provenance), config_snapshot.json, events.jsonl, stdout.log, checkpoints/ (when the trainer hook fires)
```

## Provenance

- Every results record carries the provenance block produced by `src/bapinnsformer/utils/provenance.py`: git commit, composed config hash, split hash, raw-data checksums, package versions, hardware, wall-clock, peak memory.
- Splits are serialized and referenced by hash; an experiment must never silently use a different split.
- Per-(city, season, pollutant) fits are independent (Kaggle 12 h session cap; no checkpoint-resume engineering required).

## Derived artifacts

- Figures (`paper/figures/`) and tables (`paper/tables/`) are generated from `results/` by `scripts/90_make_figures.py` / `91_make_tables.py`. Never hand-edit them.
- Negative results (especially E1 identifiability limits) are first-class outputs.
