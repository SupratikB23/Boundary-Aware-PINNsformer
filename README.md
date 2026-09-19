# Boundary-Aware PINNsformer

Observation-driven recovery of transboundary pollutant inflow for the Delhi-NCR airshed using a physics-informed spatiotemporal inverse solver. Given sparse interior CPCB station observations and a known ERA5 wind field, the framework jointly recovers the interior concentration field `C(x,y,t)`, the time-varying direction-resolved boundary inflow `C_b(s,t)` (primary output), the interior source field `S(x,y,t)`, and physical parameters eddy diffusivity `K` and deposition rate `lam`, subject to the advection-diffusion-deposition PDE `dC/dt + div(uC) = div(K grad C) + S - lam*C` as a soft constraint. Its methodological novelty is an advection-aligned pseudo-sequence generator that samples PINNsformer pseudo-tokens along backward wind characteristics instead of uniform forward time offsets. See `PRD.md` (source of truth) and `CLAUDE.md` (short summary).

## Install

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Requires Python >= 3.10. Copy `.env.example` to `.env` and fill in API keys (never commit `.env`). Pinned versions in `pyproject.toml` / `requirements.txt` are authoritative.

## Reproduce

```bash
python tasks.py data      # E0: QC, align, domain, splits -> data/processed (+ data-quality report)
python tasks.py e1        # E1: synthetic identifiability sweep (stations x config x wind-entropy x noise)
python tasks.py e2        # E2: real Delhi-NCR fits, interior fit / NCR perimeter validation
python tasks.py e3        # E3: pseudo-sequence / activation / physics-term ablations + profiling
python tasks.py e4        # E4: recovered NW inflow vs withheld VIIRS FRP + DSS/WRF-Chem comparison
python tasks.py e5        # E5: transfer to second airshed (Kolkata/Chennai domain override)
python tasks.py figures   # regenerate paper/figures + paper/tables from results/
python tasks.py paper     # build paper/main.pdf (requires a LaTeX toolchain)
python tasks.py test      # pytest suite incl. manufactured-solution and leakage guards
python tasks.py smoke     # fastest CPU sanity check (~1 min, no data needed)
```

No `make` needed: `tasks.py` (stdlib only) is the cross-platform runner and works
identically on Windows/PowerShell, Linux and Kaggle. (`make` is absent on Windows
and was the only reason for a Makefile; PRD §4.1 allowed `tasks.py`.)

Each `tasks.py` target maps to a thin `scripts/*.py` entry point plus a `configs/**/*.yaml` composition (see `tasks.py` header and PRD §4.2/§4.4). Every run writes `metrics.json` (record + embedded provenance block), `config_snapshot.json`, `events.jsonl` and `stdout.log` under `results/runs/<run_id>/`, plus `checkpoints/` when the trainer checkpoint hook fires (`scripts/20_fit_real.py --smoke-fit` exercises it). Figures and tables are derived artifacts, never hand-edited.

## Data access

- CPCB CCR portal exports + OpenCity bulk mirror -> `data/raw/cpcb/`, `data/raw/opencity/` (immutable, checksummed, gitignored).
- ERA5 via CDS API (u10/v10, BLH, T, RH) -> `data/raw/era5/`. Set `CDS_API_KEY` in `.env`.
- NASA FIRMS VIIRS/MODIS fire + FRP -> `data/heldout/firms/` ONLY. Held out of all training; asserted by `tests/test_leakage.py`. Set `FIRMS_API_KEY` in `.env`.
- Processed, analysis-ready objects live in `data/processed/`; QC masks in `data/interim/`; E1 ground truth in `data/synthetic/`. See `data/README.md`.
- Study period Oct 2019 - Feb 2024 (focus Oct-Nov post-monsoon, Dec-Feb winter). Masked loss over gaps; never impute.

## Citation

```bibtex
@unpublished{bapinnsformer2026,
  title  = {Boundary-Aware PINNsformer: Observation-Driven Recovery of Transboundary
            Pollutant Inflow using Advection-Aligned Pseudo-Sequences},
  author = {BBM Research},
  year   = {2026},
  note   = {Manuscript in preparation. Code and processed data release forthcoming.}
}
```
