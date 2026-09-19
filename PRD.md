# PRD — Boundary-Aware PINNsformer

**Product Requirements Document / Master Build Specification**

Project: *Boundary-Aware PINNsformer: Observation-Driven Recovery of Transboundary Pollutant Inflow using Advection-Aligned Pseudo-Sequences*
Domain: Delhi-NCR airshed (transfer target: Kolkata or Chennai)
Deliverables: (a) a reproducible research codebase, (b) a submitted manuscript, (c) a released processed dataset.
Status of this document: **authoritative**. `CLAUDE.md` is the short summary; this file is the in-depth plan. All code generation and paper writing later in the project must be driven from here.

---

## 0. How to read this document

| Section | Answers |
|---|---|
| 1 | What problem we are solving and what "done" means |
| 2 | The mathematical object being built (contracts, not code) |
| 3 | System architecture — data plane, model plane, experiment plane |
| 4 | Full repository file tree and the responsibility of **every** file |
| 5 | Configuration system and what is sweepable |
| 6 | Experiment specifications E0–E5 with acceptance criteria |
| 7 | Baselines and evaluation metrics |
| 8 | AI agent architecture — the agents/tools that build and run this project |
| 9 | Compute budget and scheduling |
| 10 | Reproducibility, provenance and data-leakage guards |
| 11 | Paper production plan (files, figures, tables) |
| 12 | Milestones, risks, definition of done |

---

## 1. Problem statement and success definition

### 1.1 What is being built

An **inverse solver**. Inputs: sparse hourly pollutant observations from interior CPCB stations, plus a known ERA5 wind field over a rectangular domain enclosing Delhi-NCR. Outputs, recovered jointly:

- `C(x,y,t)` — continuous interior concentration field
- `C_b(s,t)` — **the primary scientific output**: time-varying, direction-resolved pollutant inflow along the domain boundary, parameterized by boundary arc-length `s`
- `S(x,y,t)` — interior emission source field
- `K`, `λ` — eddy diffusivity and deposition/removal rate

subject to `∂C/∂t + ∇·(uC) = ∇·(K∇C) + S − λC` enforced as a soft PDE residual by automatic differentiation.

From `C_b` we derive the headline number: the **emission-inventory-free transboundary contribution share** to Delhi-NCR particulate burden, per season and per pollutant.

### 1.2 Novelty being claimed (must be defensible in the paper)

1. First formulation of atmospheric transboundary inflow as a physics-informed **inverse boundary problem**.
2. First application of **PINNsformer to real observational data** of any kind.
3. **Advection-aligned pseudo-sequence generator** — pseudo-tokens on backward wind characteristics, replacing arbitrary uniform forward offsets.
4. **Identifiability characterization** of the boundary-vs-source decomposition, including the *tomographic* interpretation of wind rotation (angular diversity ⇒ invertibility).
5. An observation-only transboundary estimate validated against withheld perimeter stations and withheld satellite fire data.

### 1.3 Success criteria

| Level | Criterion |
|---|---|
| **Minimum (project is publishable)** | E1 complete: identifiability boundary mapped on synthetic ground truth as a function of station count, station configuration, wind directional entropy and noise. |
| **Target** | E2 recovers `C_b` with RMSE/R² against **withheld** NCR perimeter stations better than zero-inflow, climatological-inflow, MLP-PINN and back-trajectory baselines; derived transboundary share is within, or explicably divergent from, the DSS/WRF-Chem range. |
| **Strong** | E3 shows advection-aligned sequences reach target accuracy at **shorter** sequence length (lower wall-clock/memory) than uniform offsets; E4 shows significant Spearman correlation between recovered NW inflow and withheld VIIRS FRP; E5 reproduces the pipeline on a second airshed. |

### 1.4 Explicit non-goals

- Not a forecasting model. We do not claim next-hour/next-day prediction skill.
- Not a chemistry model. Single-species transport with first-order loss; no gas-phase mechanism, no aerosol microphysics.
- Not 3-D. Vertically integrated 2-D field; the vertical-mixing assumption is an acknowledged limitation modulated by boundary-layer height.
- Not an emission inventory replacement. `S` is a recovered nuisance field with a sparsity prior, not a validated inventory.

---

## 2. Mathematical contracts

These are the interfaces every module must respect. Written as contracts so code and paper stay consistent.

### 2.1 Domain

- Rectangular lat/lon box enclosing Delhi + the NCR ring, projected to a local metric CRS (metres) so that advection and diffusion carry physical units.
- Non-dimensionalization: all networks see normalized `(x̂, ŷ, t̂) ∈ [-1,1]³`; a single `Normalizer` object owns the forward/inverse maps and is serialized with every checkpoint. **No module may hardcode scaling.**
- Boundary parameterized by arc-length `s ∈ [0, P)` traversed counter-clockwise from the SW corner, with outward normal `n̂(s)` available analytically.

### 2.2 Networks

| Net | Signature | Notes |
|---|---|---|
| `C_net` | `(x̂, ŷ, t̂) → C` | PINNsformer; consumes a pseudo-sequence of length `L`, returns the value at the query token |
| `Cb_net` | `(ŝ, t̂) → C_b` | Small MLP/SIREN; positive output (softplus); temporal smoothness regularized |
| `S_net` | `(x̂, ŷ, t̂) → S` | Small MLP; non-negative output; L1/TV sparsity regularized |
| `K` | scalar or `(BLH, stability) → K` | Positive; log-parameterized |
| `λ` | scalar per pollutant | Positive; log-parameterized; ordering constraint across PM fractions |

### 2.3 Pseudo-sequence generator (the core novelty)

A generator takes a query collocation point `(x, y, t)` and the wind field and returns `L` tokens. Four variants, all behind one interface:

| Variant | Token `k` |
|---|---|
| `uniform_forward` (PINNsformer original) | `(x, y, t + kΔt)` |
| `uniform_backward` | `(x, y, t − kΔt)` |
| `advection_backward` (**ours**) | `(x_k, y_k, t − kΔt)` where `(x_k,y_k)` is obtained by integrating `−u` from `(x,y,t)` over `[t−kΔt, t]` |
| `advection_jitter` (**ours**) | as above, plus isotropic Gaussian jitter of scale `√(2KΔt)` representing diffusive spread |

Backward integration uses RK2/RK4 on a bilinearly-interpolated ERA5 wind field, with reflection or clamping when a characteristic exits the domain. **A token whose characteristic terminates on the boundary is flagged**; this flag is exposed because such tokens are exactly the ones carrying boundary information, and the flag is used both in the boundary loss and in an attention-attribution figure for the paper.

### 2.4 Boundary condition switching

At each boundary collocation point `(s, t)`:
- `u·n̂ < 0` (inflow) ⇒ Dirichlet residual `C(s,t) − C_b(s,t)`
- `u·n̂ ≥ 0` (outflow) ⇒ natural residual `∂C/∂n (s,t)`

The switch is determined entirely by observed wind; no tunable threshold beyond a small hysteresis band around `u·n̂ = 0` to avoid gradient chatter.

### 2.5 Objective

```
L_total = w_data·L_data + w_pde·L_pde + w_bc·L_bc + w_reg·L_reg
```

- `L_data` — **masked** MSE at station locations/times where a valid observation exists. Mask, never impute.
- `L_pde` — MSE of the PDE residual at interior collocation points, autodiff.
- `L_bc` — inflow Dirichlet + outflow natural residuals as above.
- `L_reg` — L1 or total-variation on `S` (spatial sparsity) + temporal smoothness (first/second difference) on `C_b`, + positivity penalties.

Weights `w_*` are set by **gradient-norm balancing** (learning-rate-annealing style), recomputed every `N` steps, not fixed constants. Training: Adam with cosine schedule → L-BFGS refinement. **Temporal curriculum**: fit an expanding time window, so the model sees `[t0, t0+Δ]` before `[t0, t0+2Δ]`, etc.

---

## 3. System architecture

Three planes. Each plane is independently runnable and independently testable.

```
┌─────────────────────────── DATA PLANE (CPU) ───────────────────────────┐
│ CPCB CCR / OpenCity ─┐                                                 │
│ ERA5 (CDS API)     ──┼─► ingest ─► QC/mask ─► align to grid+clock ──┐  │
│ FIRMS VIIRS (held) ──┘                                              │  │
│                                        domain def + station split ──┼─►│ processed/*.parquet + *.nc
└─────────────────────────────────────────────────────────────────────┴──┘
                                        │
┌─────────────────────────── MODEL PLANE (GPU) ──────────────────────────┐
│  Normalizer                                                            │
│     │                                                                  │
│  PseudoSequenceGenerator ──(L tokens, boundary-hit flags)──┐           │
│     │                                                      ▼           │
│  C_net (PINNsformer: enc/dec attention + Wavelet act.)  ─► C           │
│  Cb_net ─► C_b        S_net ─► S        K, λ (log-params)              │
│     │                                                                  │
│  Residuals: PDE (autodiff) | BC (wind-switched) | Data (masked) | Reg  │
│     │                                                                  │
│  GradNorm balancer ─► Adam(+curriculum) ─► L-BFGS ─► checkpoint        │
└────────────────────────────────────────────────────────────────────────┘
                                        │
┌────────────────────── EXPERIMENT PLANE (orchestration) ────────────────┐
│ configs/*.yaml ─► runner ─► per-(city,season,pollutant) fit            │
│ synthetic FD solver (E1 ground truth)                                  │
│ baselines: zero / climatological / MLP-PINN / uniform-PSF / trajectory │
│ metrics ─► results/*.json ─► figures ─► paper/                         │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Key architectural decisions and their rationale

| Decision | Rationale |
|---|---|
| Separate small nets for `C_b` and `S` rather than multi-head on `C_net` | Different input domains (1-D+t vs 2-D+t), different priors, and they must be independently ablatable/freezable to probe identifiability. |
| Pseudo-sequence generator is a **pluggable strategy object** | E3 ablates four variants at matched parameter counts; the swap must be a config line, not a code edit. |
| Wind field is a precomputed interpolator object, not re-read per step | Backward characteristic integration is called for every token of every collocation point; this is the hot loop. Precompute to a GPU tensor grid + `grid_sample`-style bilinear lookup. |
| Independent per-instance fits, not one joint model | Maps onto the 12 h Kaggle session cap without checkpoint-resume engineering; gives per-season/per-pollutant error bars for free. |
| Masked loss, never imputation | CPCB missingness is substantial and non-uniform; imputation would inject a prior into the very field we invert. |
| FIRMS lives in a separate `heldout/` directory the training code cannot import | Hard structural guard against the circularity that would invalidate E4. |

---

## 4. Repository layout — every file and what belongs in it

> Descriptions state **responsibilities and contents**, not code.

```
BBM Research PAPER/
├── CLAUDE.md
├── PRD.md                       ← this file
├── README.md
├── pyproject.toml / requirements.txt
├── tasks.py                     (cross-platform runner; replaces Makefile)
├── .env.example
│
├── configs/
├── src/bapinnsformer/
├── scripts/
├── notebooks/
├── tests/
├── data/
├── results/
├── paper/
├── agents/
└── Reference Papers/            (existing literature PDFs, gitignored)
```

### 4.1 Root files

| File | Contents |
|---|---|
| `README.md` | One-paragraph project description, install steps, "reproduce Figure N" commands, data-access instructions and licensing, citation block. Written last, kept honest. |
| `pyproject.toml` | Package metadata, pinned dependency versions (torch, numpy, scipy, xarray, cdsapi, pandas/polars, geopandas/pyproj, netCDF4, hydra/omegaconf, matplotlib, pytest). Pinned — reproducibility is a deliverable. |
| `tasks.py` | Named entry points: `python tasks.py data/e1/e2/e3/e4/e5/figures/paper/test/smoke`. Each maps to a script + config. Stdlib only; works on Windows, Linux, Kaggle (no `make` dependency). |
| `.env.example` | Placeholder names for CDS API key, CPCB portal credentials if any, FIRMS API key. **Never** commit real keys. |

### 4.2 `configs/` — Hydra/OmegaConf tree

| File | Contents |
|---|---|
| `config.yaml` | Root composition: defaults list, global seed, device, output dir, logging level. |
| `domain/delhi_ncr.yaml` | Lat/lon box, target CRS, boundary discretization (number of `s` points), interior/perimeter station split rule, buffer distances. |
| `domain/kolkata.yaml`, `domain/chennai.yaml` | Same schema for E5. |
| `data/cpcb.yaml` | Station whitelist/blacklist, pollutant list, missingness threshold for station inclusion, unit conversions, QC rule parameters (flatline length, spike z-score, negative/zero handling). |
| `data/era5.yaml` | CDS variable names, pressure/single levels, grid resolution, time range, cache path. |
| `model/pinnsformer.yaml` | Embedding dim, number of encoder/decoder layers, heads, feed-forward width, activation (`wavelet` / `tanh` / `gelu`), dropout, parameter-count target. |
| `model/cb_net.yaml`, `model/s_net.yaml` | Depth, width, activation, output transform (softplus), Fourier-feature settings. |
| `pseudoseq/{uniform_forward,uniform_backward,advection_backward,advection_jitter}.yaml` | Variant name, sequence length `L`, `Δt`, integrator (rk2/rk4), substeps, jitter scale rule, out-of-domain policy. |
| `train/default.yaml` | Optimizer settings, Adam steps, L-BFGS steps, gradient-balancing period, curriculum schedule, collocation-point counts and resampling policy, early-stop rules. |
| `experiment/e1_identifiability.yaml` | Sweep grids: station counts {7,15,25,40}, configuration archetypes, wind-entropy bins, noise levels; number of synthetic replicates per cell. |
| `experiment/e2_real.yaml` | Seasons, pollutants, fit windows, station split, validation protocol. |
| `experiment/e3_ablation.yaml` | Variant × sequence-length grid, matched-parameter-count constraint, what to profile. |
| `experiment/e4_validation.yaml` | FIRMS aggregation radius/sector, daily resampling, correlation tests, comparison targets. |
| `experiment/e5_transfer.yaml` | Second-airshed overrides. |

### 4.3 `src/bapinnsformer/` — the package

#### `data/`

| File | Responsibility |
|---|---|
| `cpcb_ingest.py` | Fetch/parse CPCB CCR exports and the OpenCity bulk CSV archive; normalize station IDs, names, coordinates; harmonize column naming across vintages; emit a tidy long-format table (station, time, pollutant, value, source). |
| `era5_ingest.py` | CDS API retrieval of hourly 10 m `u10`/`v10`, boundary-layer height, temperature, humidity over the domain box; caching by (box, variable, month); write to NetCDF. |
| `firms_ingest.py` | VIIRS/MODIS active-fire and FRP retrieval for Punjab/Haryana and adjacent states. **Writes only to `data/heldout/`.** Module carries an explicit banner that it must never be imported by anything under `models/` or `train/`. |
| `qc.py` | Per-station quality control: range checks, flatline detection, spike detection, duplicate timestamps, timezone normalization (IST), unit checks. Produces a per-station QC report and a boolean validity mask — **no imputation anywhere**. |
| `align.py` | Build the aligned analysis object: station table joined to ERA5 fields at station locations and at collocation grid; consistent hourly clock; projection to metric CRS; normalization statistics. |
| `domain.py` | Domain polygon/rectangle construction, arc-length parameterization of the boundary, outward-normal computation, interior-vs-perimeter station classification, buffer logic, plotting helpers for the domain map figure. |
| `splits.py` | The validation protocols: interior-fit/perimeter-hold-out split (E2 primary), leave-one-interior-station-out, temporal blocks. Split definitions are serialized to disk so every experiment references the *same* split by hash. |
| `wind_field.py` | Wraps the ERA5 wind into a differentiable, GPU-resident interpolator with a `sample(x, y, t)` contract; used by the PDE residual *and* by the pseudo-sequence generator. Owns the performance-critical path. |
| `synthetic.py` | Finite-difference forward solver for E1: generates concentration fields from prescribed `S` and `C_b` under real ERA5-derived wind sequences; configurable grid, CFL-safe timestep, optional noise model; returns ground truth for every recoverable quantity. |
| `stats.py` | Missingness characterization, station-vs-ERA5 wind agreement metrics, **wind directional entropy** computation per fitting window (this quantity is central to RQ2 and appears in E1, E2 stratification and E5). |

#### `models/`

| File | Responsibility |
|---|---|
| `normalizer.py` | Forward/inverse scaling for space, time, concentration; serialized with checkpoints; single source of truth for units. |
| `activations.py` | Wavelet activation (PINNsformer original) plus tanh/GELU/SIREN alternatives for the E3 ablation. |
| `embeddings.py` | Spatiotemporal input embedding, Fourier features, positional/temporal encodings for the pseudo-sequence. |
| `pseudoseq.py` | The four generator variants behind one interface; backward characteristic integration (RK2/RK4) against `wind_field`; out-of-domain handling; **boundary-hit flags**; diffusive jitter. The headline methodological file. |
| `pinnsformer.py` | The `C_net`: encoder–decoder attention stack over the pseudo-sequence, configurable depth/width/heads, optional encoder-free mode (the literature argues the encoder is redundant — keep it as a flag), attention-weight extraction hook for the interpretability figure. |
| `boundary_net.py` | `Cb_net`: maps (arc-length, time) → non-negative inflow concentration; smoothness-friendly parameterization; optional per-pollutant heads. |
| `source_net.py` | `S_net`: non-negative interior source field with sparsity-friendly parameterization. |
| `params.py` | Learnable physical scalars `K`, `λ` (log-parameterized, positive), optional `K` as a function of BLH/stability; enforces the PM1<PM2.5<PM10 deposition ordering when fitting jointly. |
| `baselines.py` | Zero-inflow and climatological-inflow boundary assumptions; vanilla MLP-PINN with the identical inverse formulation; back-trajectory + regression statistical attribution baseline (HYSPLIT or an equivalent trajectory engine). |

#### `physics/`

| File | Responsibility |
|---|---|
| `operators.py` | Autodiff gradient/divergence/Laplacian helpers over the network outputs; the advection–diffusion–deposition residual assembled from them. |
| `boundary.py` | Wind-sign partition of the boundary, Dirichlet residual on inflow segments, natural (zero-gradient) residual on outflow segments, hysteresis band. |
| `collocation.py` | Sampling of interior/boundary/initial collocation points; adaptive or residual-weighted resampling; curriculum-aware time windows. |
| `mass_balance.py` | Domain mass-balance diagnostic: integrated inflow, outflow, source, deposition and storage change; used as a physical-consistency metric and as a figure. |

#### `train/`

| File | Responsibility |
|---|---|
| `losses.py` | Masked data loss, PDE loss, boundary loss, regularizers (L1/TV on `S`, temporal smoothness on `C_b`, positivity); each returns both value and gradient-norm for the balancer. |
| `balancing.py` | Gradient-norm loss balancing — periodic recomputation of `w_*` from per-term gradient norms; logs the weight trajectory (a diagnostic figure in the paper). |
| `curriculum.py` | Expanding time-window schedule; interacts with collocation sampling and with early stopping. |
| `optim.py` | Adam schedule then L-BFGS refinement stage; handles the closure-based L-BFGS loop and divergence guards. |
| `trainer.py` | The fit loop for one instance (city, season, pollutant): assembles nets, losses, balancer, curriculum, optimizers; logs; checkpoints; returns a fitted-instance object. |
| `checkpoint.py` | Save/load of all nets + scalars + normalizer + config hash + git commit + split hash. A checkpoint must be sufficient to regenerate every reported number. |

#### `eval/`

| File | Responsibility |
|---|---|
| `metrics.py` | Relative L2 on recovered `C_b` and `S` (E1); RMSE/MAE/R² against withheld perimeter and held-out interior stations (E2); Spearman correlation utilities (E4); identifiability margin definition. |
| `attribution.py` | Converts recovered `C_b` into the **transboundary contribution share** per season/pollutant, with the accounting rules written down explicitly (this is the headline number and must be auditable); sector decomposition of the boundary (NW/N/E/S/W). |
| `compare_external.py` | Comparison against published DSS and WRF-Chem seasonal shares; produces the divergence table and the narrative of where and why estimates differ. |
| `profiling.py` | Wall-clock, peak memory and parameter-count measurement per pseudo-sequence variant and sequence length (E3 efficiency claim). |
| `identifiability.py` | E1 analysis: recovery error as a function of station count, configuration, wind directional entropy and noise; fits the boundary surface and reports the operating regime. |

#### `viz/`

| File | Responsibility |
|---|---|
| `maps.py` | Domain map with interior/perimeter stations, boundary arc-length parameterization, example wind field. |
| `fields.py` | Recovered `C`, `S` field snapshots and animations; ground-truth-vs-recovered panels for E1. |
| `boundary_plots.py` | `C_b(s,t)` Hovmöller plots; sector time series; inflow-vs-fire overlay for E4. |
| `ablation_plots.py` | Accuracy-vs-sequence-length, accuracy-vs-wall-clock, accuracy-vs-memory; variant comparison. |
| `attention_plots.py` | Attention weights over pseudo-tokens, highlighting boundary-hitting tokens — the figure that visually justifies advection alignment. |
| `style.py` | Single matplotlib style/rcParams module so every figure is publication-consistent (fonts, sizes, colourmaps, colour-blind-safe palette). |

#### `utils/`

| File | Responsibility |
|---|---|
| `seeding.py` | Global seed control across numpy/torch/python; determinism flags. |
| `logging.py` | Structured run logging (console + JSONL); run IDs. |
| `registry.py` | Name→class registries for pseudo-sequence variants, activations, baselines, so configs can select by string. |
| `io.py` | Path conventions, parquet/NetCDF/JSON readers-writers, results schema validation. |
| `provenance.py` | Captures git commit, config hash, split hash, data-file checksums, package versions into every results record. |

### 4.4 `scripts/` — thin CLI entry points (argument parsing + config + call into `src/`)

| Script | Purpose |
|---|---|
| `00_fetch_cpcb.py` | Pull/refresh CPCB + OpenCity raw data into `data/raw/`. |
| `01_fetch_era5.py` | CDS retrieval into `data/raw/era5/`. |
| `02_fetch_firms.py` | FIRMS retrieval into `data/heldout/`. Separate, deliberately. |
| `03_build_dataset.py` | QC → align → domain → splits → `data/processed/`. Emits the data-quality report (E0 deliverable). |
| `04_characterize_data.py` | Missingness stats, ERA5-vs-station wind agreement, directional entropy per window; figures + tables for the data section. |
| `10_run_synthetic_sweep.py` | E1 sweep driver. |
| `20_fit_real.py` | E2 single-instance fit (city, season, pollutant). |
| `21_fit_all_real.py` | Fan-out of E2 across seasons/pollutants, respecting the session cap. |
| `30_run_ablations.py` | E3 pseudo-sequence / activation / physics-term ablations with profiling. |
| `40_validate_fires.py` | E4 recovered-inflow vs FRP correlation and external comparison. |
| `50_transfer_airshed.py` | E5 second-airshed run. |
| `90_make_figures.py` | Regenerate every paper figure from `results/`. |
| `91_make_tables.py` | Regenerate every paper table (LaTeX) from `results/`. |

### 4.5 `tests/`

| File | What it guards |
|---|---|
| `test_normalizer.py` | Round-trip scaling exactness. |
| `test_operators.py` | Autodiff derivatives vs analytic derivatives on manufactured functions. |
| `test_pde_residual.py` | **Method of manufactured solutions** — residual ≈ 0 for an analytic `C` satisfying the PDE. Non-negotiable. |
| `test_pseudoseq.py` | Backward characteristics recover straight lines under uniform wind; boundary-hit flags fire correctly; out-of-domain policy; jitter statistics. |
| `test_boundary_switch.py` | Inflow/outflow partition matches wind sign on synthetic wind fields; hysteresis behaviour. |
| `test_synthetic_solver.py` | FD solver conserves mass with zero source/loss; CFL guard; convergence under grid refinement. |
| `test_masking.py` | Missing observations contribute exactly zero gradient; no silent imputation anywhere in the pipeline. |
| `test_leakage.py` | **Import-graph assertion**: nothing under `models/`, `physics/`, `train/` transitively imports `firms_ingest` or reads `data/heldout/`. Protects E4's validity. |
| `test_splits.py` | Perimeter stations never appear in the fit set; split hashes are stable. |
| `test_config.py` | Every config composes; every registry name resolves. |
| `test_reproducibility.py` | Same seed + config ⇒ same loss trajectory to tolerance. |

### 4.6 `data/`

```
data/
├── raw/          cpcb/  opencity/  era5/           (immutable, checksummed, gitignored)
├── heldout/      firms/                            (E4 only; training code must not touch)
├── interim/      qc reports, per-station masks
├── processed/    aligned station parquet, era5 netcdf, domain + split json
└── synthetic/    E1 generated fields + ground truth, keyed by sweep cell
```

### 4.7 `results/`

```
results/
├── e0_data_quality/   ├── e1_identifiability/   ├── e2_real/
├── e3_ablation/       ├── e4_validation/        ├── e5_transfer/
└── runs/<run_id>/     config.yaml, provenance.json, metrics.json, checkpoints/, logs.jsonl
```

Every results record carries the provenance block from §4.3/`provenance.py`. Figures and tables are **derived artifacts** — never hand-edited.

### 4.8 `paper/`

| File | Contents |
|---|---|
| `main.tex` | Manuscript (journal template swapped per target venue). |
| `sections/*.tex` | `intro`, `related`, `method`, `data`, `experiments`, `results`, `discussion`, `limitations`, `conclusion`. |
| `figures/`, `tables/` | Auto-generated from `results/` by scripts 90/91. |
| `refs.bib` | Bibliography, seeded from `Reference Papers/` plus the PINNsformer line, AirPhyNet, SPIN, CoNOAir, OmniAir, DSS, WRF-Chem attribution studies. |
| `notes/gap_matrix.md` | The living related-work table (approach → what it leaves open) that becomes §2 of the paper. |
| `notes/claims_ledger.md` | **Every claim in the paper mapped to the results file and figure that supports it.** Reviewed before submission; nothing unsupported ships. |

### 4.9 `agents/` — see §8.

---

## 5. Configuration and sweepable surface

- Hydra-style composition; a run is fully described by its composed config + seed + split hash.
- **Sweepable**: pseudo-sequence variant, sequence length `L`, `Δt`, network widths/depths, activation, loss-weighting mode, curriculum schedule, collocation counts, station count/configuration (synthetic), noise level, wind-entropy bin, season, pollutant, city.
- **Fixed by contract**: units, CRS, normalization scheme, boundary parameterization direction, split definitions.
- Every sweep writes one row per cell to a tidy results table; analysis code never re-reads raw logs.

---

## 6. Experiment specifications

### E0 — Data pipeline and characterization
**Do**: retrieve CPCB (CCR + OpenCity mirror) and ERA5; QC; align; define the domain and boundary; classify interior vs perimeter stations; characterize missingness; quantify station-wind vs ERA5-wind agreement; compute directional entropy per candidate fitting window.
**Deliverable**: reproducible pipeline + data-quality section + domain map figure + missingness table.
**Acceptance**: dataset rebuilds end-to-end from raw with one command; station-inclusion protocol documented and applied; ERA5-vs-station wind agreement reported per station and per season.

### E1 — Synthetic identifiability study *(the insurance policy; runs on CPU/small GPU)*
**Do**: generate synthetic `C` from known `S` and `C_b` under real ERA5-derived Delhi wind sequences; run the full inversion; measure recovery error against ground truth. Sweep station count {7, 15, 25, 40} × configuration archetypes (clustered / uniform / perimeter-biased) × wind directional-entropy bins × noise levels, with replicates.
**Answers**: RQ2.
**Acceptance**: a mapped identifiability boundary — recovery error as a function of the four sweep axes — plus a test of whether error is predicted by directional entropy (the tomography hypothesis). **This result is reportable on its own, including as a negative result.**

### E2 — Boundary recovery on real Delhi-NCR data *(primary quantitative result)*
**Do**: fit using **interior Delhi stations only**; withhold the ring of NCR perimeter stations entirely; evaluate recovered `C_b` against those withheld observations; report per-season and per-pollutant accuracy; derive the transboundary share; compare to published DSS/WRF-Chem figures.
**Answers**: RQ1.
**Acceptance**: RMSE/MAE/R² against withheld perimeter stations reported with the baselines of §7.1; leave-one-interior-station-out as a secondary protocol; results stratified by wind directional entropy per the E1 finding.

### E3 — Architecture ablations
**Do**: four pseudo-sequence variants at **matched parameter counts**; sweep `L`; record accuracy vs wall-clock and peak memory; ablate the Wavelet activation; ablate the physics term entirely.
**Answers**: RQ3.
**Acceptance**: an accuracy-vs-`L` curve per variant showing whether advection alignment reaches target accuracy at smaller `L`; an explicit statement of whether the Wavelet benefit reported on analytical PDEs transfers to noisy real signals.

### E4 — Independent validation
**Do**: correlate daily recovered north-westerly sector inflow (Oct–Nov) against upwind VIIRS active-fire counts and FRP; compare seasonal mean recovered share against published inventory-based estimates; characterize divergence.
**Answers**: RQ4.
**Acceptance**: Spearman correlation with significance and lag analysis; an explicit audit trail showing no fire data entered training (`tests/test_leakage.py` passing is cited in the paper).

### E5 — Transferability
**Do**: apply the identical pipeline to Kolkata or Chennai (sparser network, coastal/sea-breeze meteorology); test the E1 wind-diversity prediction in a second real setting.
**Acceptance**: full pipeline runs with only a `domain/*.yaml` change; results reported honestly even if weaker.

---

## 7. Baselines and metrics

### 7.1 Baselines
1. **Zero-inflow** boundary assumption (implicit treatment in existing physics-informed AQ work).
2. **Climatological-inflow** boundary assumption.
3. **Vanilla MLP-PINN** with the identical inverse formulation → isolates the transformer's contribution.
4. **PINNsformer with original uniform pseudo-sequence** → isolates advection alignment.
5. **Back-trajectory statistical attribution** (HYSPLIT or equivalent + regression) → the standard non-mechanistic approach.
6. **Published inventory-based attribution** (DSS, WRF-Chem) as an external reference point, not a trainable baseline.

### 7.2 Metrics

| Category | Metrics |
|---|---|
| Synthetic recovery (E1) | Relative L2 on `C_b` and on `S`; identifiability margin vs station count and wind directional entropy |
| Real-data validation (E2) | RMSE, MAE, R² vs withheld perimeter stations; interior field error at held-out interior stations |
| Physical consistency | Domain mass-balance closure; recovered `K`, `λ` within published ranges; deposition ordering PM1 < PM2.5 < PM10 |
| Independent validation (E4) | Spearman correlation of recovered NW inflow vs upwind FRP; agreement/divergence vs published seasonal shares |
| Efficiency (E3) | Accuracy vs wall-clock, peak memory, parameter count — per variant and per `L` |

---

## 8. AI agent architecture

This project is built with Claude Code as the engineering surface. The agent layer is part of the product: it must be documented in `agents/` so the workflow is reproducible and so the paper's methods section can state honestly how the code was produced.

### 8.1 Agent roster

| Agent | Mandate | Primary tools | Outputs |
|---|---|---|---|
| **Orchestrator** (main session) | Owns `PRD.md`, decomposes work, enforces acceptance criteria, decides when a milestone is met. Never writes large code itself when a specialist fits. | Read, Write, Edit, Bash, Task | Milestone status, task assignments |
| **Data Engineer** | E0: ingestion, QC, alignment, domain/split construction, data-quality report | Bash, Read/Write/Edit, WebFetch (CPCB/OpenCity/CDS/FIRMS docs) | `src/bapinnsformer/data/*`, `scripts/0*`, data-quality report |
| **Physics/Numerics** | PDE residual, boundary switching, collocation, FD synthetic solver, manufactured-solution tests | Read/Write/Edit, Bash (pytest) | `physics/*`, `data/synthetic.py`, `tests/test_pde_residual.py` |
| **Model Engineer** | PINNsformer, pseudo-sequence variants, `Cb_net`, `S_net`, physical parameters, baselines | Read/Write/Edit, Bash | `models/*` |
| **Training Engineer** | Losses, gradient-norm balancing, curriculum, Adam→L-BFGS, checkpointing, stability triage | Read/Write/Edit, Bash, Monitor (long runs) | `train/*` |
| **Experiment Runner** | Executes E1–E5 sweeps, manages the Kaggle session cap, collects tidy results | Bash (background runs), Read/Write | `results/*` |
| **Evaluator** | Metrics, attribution accounting, external comparison, profiling, identifiability analysis | Read/Write/Edit, Bash | `eval/*`, results tables |
| **Literature Agent** | Maintains `gap_matrix.md` and `refs.bib`; verifies every related-work claim against primary sources; watches for pre-emption | alphaXiv / paper-search MCP tools, `literature-review` & `alpha-research` skills, WebSearch, WebFetch | `paper/notes/gap_matrix.md`, `paper/refs.bib` |
| **Paper Writer** | Drafts and revises manuscript sections from `results/` only; maintains `claims_ledger.md` | `paper-writing` skill, Read/Write/Edit | `paper/sections/*` |
| **Reviewer / Red Team** | Adversarial pass: attacks identifiability claims, leakage, overclaiming, figure honesty, reproducibility | `peer-review` skill, `code-review`, Read, Bash | Review reports, blocking issues |
| **Figure Agent** | Publication figures and tables regenerated from results | `dataviz` skill, matplotlib via Bash | `paper/figures`, `paper/tables` |

### 8.2 Available tool surface (mapped to this project)

| Tool / skill | Used for |
|---|---|
| `Read` / `Write` / `Edit` / `Glob` / `Grep` | All code and manuscript work |
| `Bash` / `PowerShell` | Environment setup, pytest, training launches, profiling, background sweeps |
| `Agent` (subagents) | Parallel specialist work — **only on explicit request**, per repo working style |
| `Monitor` / background Bash | Long training runs on the workstation |
| **alphaXiv MCP** (`search`, `get_paper_content`, `answer_pdf_queries`, library folders) | Literature agent: PINNsformer line, AirPhyNet/SPIN/CoNOAir/OmniAir, PINN inverse-boundary work in acoustics/haemodynamics |
| **Firecrawl MCP** (`firecrawl_search`, `firecrawl_scrape`, `research_*`) | CPCB/OpenCity portal structure, DSS bulletins, WRF-Chem attribution reports, ERA5/FIRMS docs |
| `WebSearch` / `WebFetch` | Policy figures, court references, venue formatting requirements |
| `literature-review`, `alpha-research`, `source-comparison` skills | Building and maintaining the gap matrix |
| `paper-writing`, `peer-review`, `humanizer` skills | Manuscript drafting and adversarial review |
| `dataviz` skill | Every chart in the paper |
| `docx` / `pdf` / `preview` skills | Proposal/manuscript conversion and review copies |
| `graphify` skill | Knowledge graph over the codebase + `Reference Papers/` for cross-referencing method and literature |
| `session-log` skill | Durable per-session research diary under `agents/logs/` |
| `Artifact` | Interactive dashboards for sweep results when a static figure is insufficient |

### 8.3 Agent operating rules

1. **PRD-first.** Any agent writing code reads the relevant §4 entry first; file responsibilities are not renegotiated ad hoc.
2. **One concern per file.** If a file starts doing two things in §4's table, split it and update this PRD.
3. **Tests before claims.** No experimental number is reported until the tests guarding that path (§4.5) pass.
4. **Leakage is a blocking bug.** Any path from FIRMS data into training halts the project until fixed.
5. **Results are generated, never typed.** Figures/tables come from `results/` via scripts 90/91.
6. **Every claim gets a ledger row.** `claims_ledger.md` maps claim → results file → figure/table.
7. **Honest reporting.** Negative results (especially E1) are first-class outputs, per §13 of the proposal.
8. **Session logs.** Each working session ends with a `session-log` entry: what was done, what broke, what's next.

---

## 9. Compute budget and scheduling

| Resource | Use |
|---|---|
| RTX 3050 workstation | Development, debugging, unit tests, small E1 cells, all CPU preprocessing |
| Kaggle dual T4 (30 h/week, 12 h/session) | E2/E3/E5 fits — **two independent fits concurrently**, not data-parallel |

- Model size target: **< 0.5 M parameters**; peak memory **< 3 GB** at `L = 5` (matches reported PINNsformer footprint).
- Each instance fit = one (city, season, pollutant) and must complete **within a single session** — no checkpoint-resume engineering required, though `checkpoint.py` exists for safety.
- Total accelerator estimate including failures and reruns: **≈ 150 GPU-hours ≈ 5 weeks of quota** against a 16-week schedule.
- Synthetic generation (E1) and all reanalysis preprocessing are **CPU**.
- Scheduling rule: never start a fit whose expected duration exceeds the remaining session time; the Experiment Runner tracks weekly quota consumption in `results/runs/quota.json`.

---

## 10. Reproducibility, provenance and guards

- Pinned dependencies; single-command environment creation.
- Global seeding; determinism flags where they don't cost unacceptable speed; `test_reproducibility.py` enforces loss-trajectory stability.
- Every run records: git commit, composed config hash, split hash, raw-data checksums, package versions, hardware, wall-clock, peak memory.
- Splits are serialized and referenced by hash — an experiment cannot silently use a different split.
- **Data-leakage guards**: FIRMS isolated in `data/heldout/`; import-graph test; perimeter stations structurally excluded from the fit set.
- Public release: code + processed dataset + configs sufficient to reproduce every figure.

---

## 11. Paper production plan

**Structure**: Introduction (policy problem + the inventory-free framing) → Related work (the three-literature gap matrix) → Method (inverse formulation, advection-aligned pseudo-sequences, objective, identifiability argument) → Data → Experiments E0–E5 → Results → Discussion (what the transboundary number means, where it diverges from DSS/WRF-Chem and why) → Limitations (2-D vertical integration, winter inversions, wind error, network sparsity) → Conclusion.

**Planned figures**
1. Domain map: boundary arc-length, interior vs perimeter stations, example wind.
2. Data quality: missingness heatmap; ERA5-vs-station wind agreement.
3. Method schematic: uniform vs advection-aligned pseudo-sequences, with boundary-hitting tokens highlighted.
4. E1 identifiability surface: recovery error vs station count × wind directional entropy.
5. E1 ground truth vs recovered `C_b` and `S` panels.
6. E2 recovered `C_b(s,t)` Hovmöller, by season.
7. E2 scatter/time series vs withheld perimeter stations, with baselines.
8. E3 accuracy vs sequence length, and accuracy vs wall-clock/memory, per variant.
9. Attention attribution over pseudo-tokens.
10. E4 recovered NW inflow vs withheld VIIRS FRP, daily, Oct–Nov.
11. Transboundary share: ours vs DSS vs WRF-Chem, per season.
12. Mass-balance closure diagnostic.

**Planned tables**: related-work gap matrix · station inclusion and missingness · E1 sweep results · E2 metrics per season/pollutant with baselines · E3 ablation with efficiency · recovered physical parameters vs published ranges · external attribution comparison.

**Venues**: *Environmental Modelling & Software* (primary) → *Atmospheric Environment* → *Geoscientific Model Development* → SIGSPATIAL / AAAI AI for Social Impact → *Scientific Reports* / *IEEE Access* (fallback).

---

## 12. Milestones, risks, definition of done

### 12.1 Milestones (16 weeks)

| Weeks | Activity | Milestone |
|---|---|---|
| 1–2 | CPCB + ERA5 retrieval, QC, domain definition, interior/perimeter split (E0) | Aligned dataset + data-quality report |
| 3–5 | FD synthetic solver; identifiability sweeps (E1) | Identifiability boundary characterized |
| 6–8 | Model implementation; advection-aligned pseudo-sequence; first single-season single-pollutant Delhi fit | First real-data boundary recovery |
| 9–10 | Full E2 across seasons and pollutants; perimeter validation | Primary quantitative result |
| 11 | E3 ablations + efficiency | Architecture contribution established |
| 12–13 | E4 fire correlation + external comparison | Independent validation |
| 14 | E5 second airshed | Generality demonstrated |
| 15–16 | Figures, writing, code + data release | Manuscript submitted |

### 12.2 Risk register

| Risk | Consequence | Mitigation / fallback |
|---|---|---|
| `C_b` and `S` not separately identifiable | Recovered share is meaningless | Caught in **E1 before any real-data work**; regularization asymmetry; stratify by directional entropy. **Fallback: report the identifiability boundary itself — a publishable negative result.** |
| Wind-field error | Advection misdirects the inversion | Cross-validate ERA5 vs station wind; perturbed-wind sensitivity analysis; restrict primary claims to good-agreement periods |
| Vertical-mixing assumption fails in winter inversions | Model least reliable when pollution is worst | BLH modulates effective diffusivity and loss; report inversion-period performance separately; document as explicit limitation |
| CPCB gaps and outages | Reduced effective sample, biased fits | Masked loss (never impute); documented inclusion threshold; report effective observation counts per fit |
| PINN training instability | Non-convergence or physics-term collapse | Gradient-norm balancing from the outset; temporal curriculum; L-BFGS refinement; **vanilla-PINN control first** to confirm the problem is solvable before adding architecture |
| Too few / poorly placed perimeter stations | Weak E2 validation | Supplement with leave-one-interior-station-out; use E1 to establish the expected error floor at that station density |
| Pre-emption by a competing paper | Novelty claim weakened | Literature Agent re-runs the gap check at weeks 1, 8 and 14 |

### 12.3 Definition of done

- [ ] `python tasks.py data` rebuilds the processed dataset from raw with a data-quality report.
- [ ] All tests in §4.5 pass, including the manufactured-solution and leakage tests.
- [ ] E1 sweep complete; identifiability boundary mapped and written up.
- [ ] E2 complete across seasons and pollutants with all §7.1 baselines.
- [ ] E3 ablations complete with efficiency measurements.
- [ ] E4 correlation computed with a documented no-leakage audit trail.
- [ ] E5 run on a second airshed.
- [ ] Every figure and table regenerable by `python tasks.py figures` / `python tasks.py tables`.
- [ ] `claims_ledger.md` has a supporting artifact for every claim in the manuscript.
- [ ] Code + processed dataset released; README reproduces at least Figures 4, 6, 8 and 10 from scratch.
- [ ] Manuscript submitted to the primary venue.
