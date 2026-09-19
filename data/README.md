# data/ — layout, immutability, and held-out guard (PRD §4.6, §10)

```
data/
├── raw/          cpcb/  opencity/  era5/   (immutable, checksummed, gitignored)
├── heldout/      firms/                    (E4 only; training code must not touch)
├── interim/      qc reports, per-station masks
├── processed/    aligned station parquet, era5 netcdf, domain + split json
└── synthetic/    E1 generated fields + ground truth, keyed by sweep cell
```

## Immutability

- `raw/` is write-once. Fetch scripts (`scripts/00_fetch_cpcb.py`, `01_fetch_era5.py`) write here and record checksums; no later stage modifies raw files.
- `processed/` is rebuilt only by `scripts/03_build_dataset.py` (QC -> align -> domain -> splits). `python tasks.py data` must rebuild end-to-end from raw with one command.

## Held-out guard (E4 validity)

- FIRMS VIIRS/MODIS fire + FRP lives ONLY in `heldout/firms/`, written ONLY by `scripts/02_fetch_firms.py` / `src/bapinnsformer/data/firms_ingest.py`.
- Nothing under `src/bapinnsformer/models/`, `physics/`, or `train/` may import `firms_ingest` or read `data/heldout/`. Enforced by `tests/test_leakage.py`. Any violation is a blocking bug (PRD §8.3 rule 4).
- E4 (`scripts/40_validate_fires.py`) is the sole consumer of held-out data, and it only reads recovered inflow + FRP for correlation — never for fitting.

## Other rules

- Masked loss over CPCB gaps — never impute (see `qc.py` validity mask).
- Study period Oct 2019 - Feb 2024; focus Oct-Nov (post-monsoon) and Dec-Feb (winter).
