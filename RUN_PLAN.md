# RUN_PLAN — Boundary-Aware PINNsformer

What to run, in what order, with what data. Root-only file per request.
PRD.md is authoritative; CLAUDE.md is the short summary. This file is the operator manual.

Status: full scaffold built, `pytest` 62/62 green on CPU (2026-09-12).
No real data has been downloaded yet — everything below starts from zero data.

---

## 0. Environment (do once)

```powershell
# Windows PowerShell, from repo root "BBM Research PAPER/"
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .            # installs src/bapinnsformer
copy .env.example .env      # then fill keys (never commit .env)
pytest -q                   # expect: 62 passed
```

Notes:
- `pyproject.toml` pins `torch==2.3.1` (+ CUDA build on workstation/Kaggle).
  CPU-only dev machines can use whatever torch is installed — the suite passes on torch 2.7 CPU.
- Pinned siblings (`numpy==1.26.4`, `pandas==2.2.2`, `polars`, `xarray`, `netCDF4`, `cdsapi`,
  `geopandas`) may NOT match your dev env (suite is green on numpy 2.x / pandas 3.x too).
  Do NOT `pip install -r requirements.txt` blindly over a working env: it will downgrade
  torch/numpy/pandas. Before GPU runs, create a FRESH venv and install pinned versions there.
- Heavy optionals (`cdsapi`, `xarray`, `netCDF4`, `geopandas`/`pyproj`, `psutil`) are lazily imported:
  missing ones only break the fetch script that needs them (with a clear ImportError), never
  `pytest` or training. `peak_memory_mb` falls back to tracemalloc on Windows without psutil.
- GPU: RTX 3050 (dev) / Kaggle dual T4. Keep `<0.5M` params, `L=5` default → `<3GB`.
  Run **two independent fits concurrently** on Kaggle, never data-parallel (PRD §9).

## 1. Datasets required RIGHT NOW

| # | Dataset | Why / used in | Where it lands | How to get it (verified 2026-09-12) |
|---|---------|---------------|----------------|--------------------------------------|
| D1a | **CPCB CCR portal** hourly station data — the PRIMARY source | E0/E2 fit + validation. Hourly PM2.5, PM10 (PM1 where available) + station met (wind speed/dir, temp, RH for ERA5 cross-check), Delhi + NCR ring, Oct 2019–Feb 2024 | `data/raw/cpcb/` (immutable snapshots + manifest) | https://airquality.cpcb.gov.in/ccr/#/ → Historical Data → state/city/station, parameter, date range → download CSV per station, drop into `data/raw/cpcb/`; `scripts/00_fetch_cpcb.py --mirror ccr` normalizes. Manual per-station export (no bulk API); ~40 Delhi + ~15 NCR stations. Portal may need login → creds in `.env`. |
| D1b | **Kaggle "Air Quality Data in India (2015–2020)"** (rohanrao) — station-wise HOURLY CPCB mirror | Fast-track D1 for Kaggle: attach as Kaggle input dataset, zero download. Same schema family as CCR (station, city, datetime, PM2.5/PM10/gases/met, AQI). Covers Oct 2019–Dec 2020 of our window | `data/raw/cpcb/` (or `/kaggle/input/...` on Kaggle, see notebook §2) | Kaggle → Add Input → search "Air Quality Data in India". LIMIT: ends 2020 — use for pipeline dev + pilot fits; backfill 2021–2024 from D1a. |
| D1c | **GitHub cp099/India-Air-Quality-Dataset** — ⚠️ NOT usable for training | Daily city-level AQI with REVERSE-ENGINEERED (estimated, not measured) pollutant values; no stations, no hourly, no met. Verdict: EDA sanity-check only, never training/eval. | Nowhere in `data/` (keep out of the pipeline) | Skip unless you want a quick plotting toy. |
| D2 | **OpenCity (data.opencity.in)** bulk mirror | Backfill/2nd source where available; `source` column keeps provenance. Bengaluru-leaning catalogue — Delhi coverage not guaranteed; treat as optional | `data/raw/opencity/` | Browse data.opencity.in → download city archive → `--mirror opencity`. No key. |
| D3 | **ERA5** hourly `u10`/`v10`, BLH, 2m T, RH over Delhi-NCR box | Known wind `u=(u,v)` for PDE + characteristics; BLH for K | `data/raw/era5/*.nc` (cache key = box×var×month) | CDS API (cds.climate.copernicus.eu account): `CDS_API_KEY/CDS_API_URL` in `.env`, box in `configs/data/era5.yaml`, run `01_fetch_era5.py`. 0.25°, hourly. KAGGLE SHORTCUT: download once anywhere, upload as a private Kaggle dataset, mount it (notebook §3) — avoids flaky multi-GB fetches inside sessions. |
| D4 | **NASA FIRMS VIIRS (+MODIS) fire + FRP**, Punjab/Haryana/adjacent, Oct–Nov seasons | **E4 ONLY, strictly held out.** Never enters training; `tests/test_leakage.py` enforces | `data/heldout/firms/` ONLY | FIRMS API (`FIRMS_API_KEY`, firms.modaps.eosdis.nasa.gov) area query → `scripts/02_fetch_firms.py`. Sole writer; training code cannot import it. Fetch early to exercise the guardrail; do not open before E4. |
| D5 | **Published DSS / WRF-Chem seasonal shares** (numbers from bulletins/papers) | External comparison table only (not a baseline) | Entered manually in `configs/experiment/e4_validation.yaml` (`comparison_targets`) | Copy seasonal share numbers + citations into the config; `compare_external.py` builds the divergence table. No download. |

Station list / domain box: `configs/domain/delhi_ncr.yaml` (E5: `kolkata.yaml`/`chennai.yaml`).
QC rules + inclusion threshold: `configs/data/cpcb.yaml`. Nothing to download for these — curate them.

Minimum to start E0: (D1a or D1b) + D3. D1c stays OUT of the pipeline. D4 can wait until E4 but fetching early proves the leakage guard.

## 2. Run order (commands)

All scripts are `python scripts/<name>.py --config <yaml> [--quick]` and log run-ids to `results/runs/`.
`python tasks.py <target>` wraps them (cross-platform; no `make` needed).

```
# 0) sanity (no data needed, laptop CPU is fine)
python tasks.py smoke                      # E1 --quick + E2 smoke-fit + checkpoint (~1 min)
# or: pytest -q                            # 77 passed = green light

# E0) data pipeline — needs D1+D3 on disk (laptop CPU is fine)
python tasks.py data
#  = 03_build_dataset.py → 04_characterize_data.py
#  out: data/processed/ (aligned parquet, era5 nc, domain+split json) + results/e0_data_quality/
#  gate: rebuilds end-to-end with one command; missingness table + ERA5-vs-station wind agreement reported

# E1) synthetic identifiability — CPU/small GPU, no real data needed (can run BEFORE E0 finishes)
python tasks.py e1                         # configs/experiment/e1_identifiability.yaml
#  sweep: stations {7,15,25,40} × {clustered,uniform,perimeter-biased} × entropy bins × noise × replicates
#  out: results/e1_identifiability/ (tidy rows) → identifiability surface (Fig 4/5)
#  THIS IS THE PUBLISHABLE FALLBACK — finish it first. Runs on Kaggle CPU or laptop.

# E2) real Delhi-NCR recovery — needs E0 output, GPU (Kaggle)
python tasks.py e2                         # 21_fit_all_real.py fans out per (season, pollutant)
#  single debug fit: python scripts/20_fit_real.py --config configs/experiment/e2_real.yaml --city delhi_ncr
#  protocol: fit interior Delhi stations only, withhold NCR perimeter ring; LOISO secondary
#  out: results/e2_real/ + checkpoints in results/runs/<run_id>/checkpoints/

# E3) ablations — needs E0 (+E2 config), GPU (Kaggle)
python tasks.py e3                         # 30_run_ablations.py, configs/experiment/e3_ablation.yaml
#  4 pseudo-seq variants × L sweep at matched params + Wavelet on/off + physics-term off; profiles time/mem

# E4) fire validation — needs E2 inflow + D4 (held out), CPU is fine
python tasks.py e4                         # 40_validate_fires.py; Spearman NW-inflow vs FRP + lag; DSS comparison
#  audit: pytest tests/test_leakage.py -q must pass; cite in paper

# E5) transfer — needs D1–D3 equivalents for second airshed, GPU (Kaggle)
python tasks.py e5                         # 50_transfer_airshed.py --config configs/experiment/e5_transfer.yaml
#  only change: configs/domain/kolkata.yaml (or chennai.yaml)

# Figures / tables / paper (derived artifacts only — never hand-edit; laptop CPU fine)
python tasks.py figures                    # 90_make_figures.py from results/ → paper/figures/
python tasks.py tables                     # 91_make_tables.py → paper/tables/*.tex  (figures implies tables)
python tasks.py paper                      # latexmk paper/main.tex (needs TeX install)
```

Per-fit session rule (PRD §9): one (city, season, pollutant) per run, must finish inside a 12h Kaggle
session; `results/runs/quota.json` tracks weekly quota. Never start a fit longer than remaining session time.

## 3. Config surface (what to edit vs never touch)

- Edit freely: `configs/pseudoseq/*.yaml` (variant, L, dt, rk2/rk4, jitter), `configs/model/*.yaml`
  (widths/depths, wavelet/tanh/gelu), `configs/train/default.yaml` (Adam/L-BFGS steps, balancing period,
  curriculum, collocation counts), `configs/experiment/*.yaml` (sweep grids, seasons, pollutants).
- Registry names resolve via `utils/registry.py` — `test_config.py` fails fast on typos.
- Never hand-edit: units/CRS/normalization (owned by `models/normalizer.py`), boundary direction
  (CCW from SW corner, `data/domain.py`), split definitions (hash-pinned in `data/splits.py`),
  anything under `results/` or `paper/figures|tables/` (regenerated by scripts 90/91).

## 4. Verify-after-each-step checklist

- [ ] `pytest -q` green (esp. `test_pde_residual` manufactured solution, `test_leakage`, `test_masking`, `test_splits`).
- [ ] Every `results/runs/<run_id>/` has `metrics.json (record + embedded provenance) + config_snapshot.json + events.jsonl + stdout.log` (+ `checkpoints/` when the hook fires).
- [ ] E1: error ↓ with stations + wind entropy (tomography hypothesis) — or document the negative result.
- [ ] E2: beat zero/climatology/MLP-PINN/uniform-PSF on withheld perimeter RMSE/MAE/R².
- [ ] E3: advection variants win at smaller L (accuracy-vs-L + time/mem curves).
- [ ] E4: Spearman NW-inflow vs FRP significant; leakage test green.
- [ ] `paper/notes/claims_ledger.md`: every manuscript claim → results file + figure.

## 5. Review fixes applied (audit trail, updated 2026-09-12 — full recheck, 22 items)

First pass: `models/pseudoseq.py::_finalize_np` dead code removed; `baselines.py`
docstring decontaminated for the leakage audit; smoke artifacts cleaned. `pytest` 62/62.

Cleanup pass: `Makefile` → `tasks.py`; `Papers/` stub deleted (literature lives in
gitignored `Reference Papers/`); `__pycache__`/`.pytest_cache` purged; `.gitignore`
extended (results E0–E5, paper figures/tables, data stages, binaries, Kaggle secrets).

Second pass (independent review, 22 findings → all fixed, `pytest` 77/77):
- **Blocking**
  1. `pseudoseq` tokens/flags/randn allocated on CPU → now on input `device` (CUDA-ready).
  2. YAML keys ≠ constructor kwargs → aliases accepted in constructors
     (`n_heads/d_ff/depth/width/output_transform/L/dt_hours/out_of_domain`) plus
     strict adapters in new `utils/config.py`
     (`pinnsformer_kwargs/cb_kwargs/s_kwargs/pseudoseq_kwargs/train_kwargs/build_trainer_config`).
  3. `scripts/_common.py` ignored `defaults:` → new `utils/io.compose_config`
     (sections nested as model/cb_net/s_net/cpcb/era5/domain/pseudoseq/train; root wins).
  4. `device: auto` crashed (`torch.device("auto")`) → `utils/io.resolve_device`
     (auto→cuda/cpu); `Trainer` uses it.
  5. `synthetic.run_forward` used backward differences for both wind signs
     (overshoot 195 vs 100 on the −5 m/s probe; double-counted `u` via flux form) →
     proper upwind advective stencil matching `SyntheticSolver.step`.
  6. Sweep seeds used salted `hash()` → sha256 cell seeds (reproducible).
- **High (silent wrong numbers)**
  7. `create_generator("advection-backward-jitter")` KeyError → normalized lookup.
  8. `ClimatologicalInflow.boundary_value` returned stale `c0` after `fit()` →
     reads the live `value` param (numpy + torch agree).
  9. Trainer counted the hysteresis buffer as outflow → now exactly
     `physics/boundary.py::partition` (buffer contributes to neither).
  10. `bc_loss(...,None,None)` silently double-counted → both-None raises;
      single-None = empty side.
  11. Losses forced float32 → dtype-preserving (`_float_tensor`; trajectories stay uniform).
  12. E3 `n_tokens=len(tokens)` always 1 → `shape[1]`; `peak_mb` NaN on Windows →
      psutil→tracemalloc fallbacks (verified: real RSS numbers).
  13. `PhysParams.K_max` stored `lam_min` → fixed.
- **Medium**
  14. Docs claimed `config.yaml/provenance.json/logs.jsonl` → corrected to actual
      `metrics.json (+embedded provenance)/config_snapshot.json/events.jsonl/stdout.log`
      (+`checkpoints/` via hook); `20_fit_real.py --smoke-fit` wires trainer→checkpoint end-to-end.
  15. Pinned-vs-installed env drift → RUN_PLAN warns against blind downgrades; optionals documented lazy.
  16. `postmonsoon` vs `post-monsoon` → `normalize_season` accepts both; canonical `postmonsoon`.
  17. Fig. 8 never emitted (guarded on missing `rmse`) → timing rows carry `rmse:""`,
      new `viz.efficiency_vs_length` always plots cost-vs-L (verified: 3 figures incl. `fig08b`).
  18. E4 lag wrapped circularly → trim (`q[lag:n]` vs `f[:n-lag]`).
  19. `_maybe_numpy` dead/wrong-shape code → corrected to plain pass-through.
- **Low**
  20. Kolkata/Chennai reused Delhi CRS → UTM 45N (`EPSG:32645`) / 44N (`EPSG:32644`).
  21. `physics.boundary_loss` detached to float → returns grad-carrying tensor on torch path.
- **Bonus found during fixing**: global float64 defaults broke the trainer
  (nets inherit float64, inputs hardcoded float32) → `Trainer.dtype` adoption,
  dtype-preserving `_prep`/tokens/forwards (`utils/dtype.py`), loss family
  preserved; new `tests/conftest.py` fixture resets default dtype after each test;
  verified a full `train_step` under `set_default_dtype(float64)`.
- Pinned by `tests/test_review_fixes.py` (15 tests: aliases, composition, upwind probe,
  seed stability, bc semantics, clamps, climatology consistency, buffer exclusion).

## 6. Known limitations / next actions for the operator

- No data yet: D1–D4 downloads are the critical path (start CDS + CPCB requests now; they are slow).
- `pyproject.toml` pins `torch==2.3.1`; dev CPU with torch 2.7 works but reinstall pinned versions before GPU runs.
- `02_fetch_firms.py` + ERA5 fetch need real API keys (`.env`).
- `paper/` is a skeleton (sections carry PRD-tied bullets, `refs.bib` entries marked verify-before-submission).
- `notebooks/01_eda.ipynb` is a stub for exploration only — never part of the pipeline.
