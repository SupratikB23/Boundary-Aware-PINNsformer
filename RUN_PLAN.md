# RUN_PLAN — Boundary-Aware PINNsformer

The operator manual: what the method is, what is already built and verified, what code still has to be written, what to download, what to run in what order, and how to tell whether a result is real.

- `PRD.md` is authoritative for architecture. **Where this file and PRD.md disagree, this file wins** (data sources revised 2026-10-03 to India-only government data; maths and evaluation revised 2026-10-03, §4).
- Status (2026-10-03): `main` has the verified mathematical core (§4), `pytest` 97/97 green. Real-data pipeline wiring exists on branch `hkv1`. **No real experiment has been run. There are no results.** This file contains no results by design.

---

## 0. Brief

**What we are doing.** We estimate how much of the PM measured in Delhi entered the Delhi-NCR airshed across its boundary. We use only Indian government monitoring-station data and the advection–diffusion–deposition equation, with no emission inventory. A physics-informed transformer (PINNsformer) jointly learns three things:

- the concentration field `C`;
- the local sources `S`;
- the unknown boundary inflow `C_b`.

The boundary inflow, and the share of Delhi's concentration it explains, are the scientific outputs.

**Why it is publishable (the novelty, §4).**

1. **C1 — Characteristic (Feynman–Kac) loss.** Pseudo-sequence tokens are placed along backward wind characteristics. The loss also enforces the exact integral solution of the PDE along them, and it couples interior points to the boundary inflow at the exact exit point.
2. **C2 — Computable identifiability theory.** The PDE is linear, so observations are `y = A_b c_b + A_s s + A_0 c_0`. Principal angles, a confounding coefficient `ρ` and boundary observability then say **in advance** when the boundary-vs-local split can be recovered from a given station network and wind window.
3. **C3 — Receptor share by exact superposition, with calibrated uncertainty.** "Transboundary share" is defined as the fraction of concentration *at Delhi receptors* due to boundary inflow. It is computed exactly by linear superposition, has a posterior 90% interval, and is the quantity directly comparable with IITM's DSS.

**Rule zero.** A number may go into a report, slide or paper only if:

- a completed run on real CPCB data produced it;
- every gate in §7 passed for it.

A fallback, dummy, smoke, quick or synthetic stand-in is never a result.

---

## 1. How to work through this file

Work strictly in order: §2 → §5 (code tasks) → §6 (runs) → §7 (gates) → §8 (reports). Every task has an ID (`M*`, `F*`, `G*`, `H*`) and a **proof** column. A task is done only when its proof exists.

When using Claude Code, start every session with:

> Read CLAUDE.md, PRD.md and RUN_PLAN.md. We are on task <ID> in RUN_PLAN §5. Read every file the task names before changing anything. Implement exactly the task, add its proof test, run `pytest -q`, and show me the diff. Do not tick anything in PRD.md or RUN_PLAN.md.

Rules for every task:

- One task per commit. The commit message starts with the task ID, e.g. `H3: windowed E2 fits`.
- `pytest -q` must be green before every commit. Never delete or weaken a test to make it pass.
- Never mark a task or experiment as done in `PRD.md` / `RUN_PLAN.md`. Done-ness lives in `results/<exp>/REPORT.md` (§8).
- When something fails or looks odd, write it in the step's `REPORT.md`. Do not tune silently until it passes.

---

## 2. Environment (do once)

```powershell
# Windows PowerShell, from the repo root "BBM Research PAPER/"
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .            # installs src/bapinnsformer
copy .env.example .env      # no API keys are needed for the India-only data plan
pytest -q                   # expect all green (97 on main as of 2026-10-03)
```

- Add `pdfplumber` to `requirements.txt` (needed for CREAMS, D3). `cdsapi` is no longer needed.
- `pyproject.toml` pins `torch==2.3.1`. Never `pip install -r requirements.txt` over a working environment: it downgrades torch, numpy and pandas. For GPU work, create a **fresh** venv.
- GPU: RTX 3050 (dev) and Kaggle dual T4 (**compute only, never a data source**). Keep models under 0.5 M params with `L = 5`. On Kaggle, run **two independent fits concurrently**, one per GPU, never data-parallel.
- Moving data to Kaggle: upload our own processed CPCB files as a **private** Kaggle dataset only to mount them. The data remains CPCB data. Cite CPCB, never Kaggle.

---

## 3. Datasets — India, government, public

All sources below were researched on 2026-10-03.
- **Login:** CPCB, CREAMS, IMD and IITM need no account. IMDAA (D2b) is the only exception and is optional.
- **Access caveat:** the CPCB CCR portal blocks automated access from outside India, so it could not be machine-checked from here. Its public, no-login status is established by its use as the stated open data source in recent peer-reviewed Delhi studies, e.g. Nandi et al., *npj Clean Air* 2026, [doi:10.1038/s44407-026-00065-6](https://www.nature.com/articles/s44407-026-00065-6).
- **First action:** open each link once in a browser from India and record in `data/README.md` what you saw: date, any login prompt, any captcha.

### 3.1 Required

| # | Dataset (owner) | Link | Login | What to take | Lands in | Used for |
|---|---|---|---|---|---|---|
| **D1** | **CPCB CCR — Continuous Ambient Air Quality Monitoring Stations (CAAQMS)**. Central Pollution Control Board, MoEFCC | Data repository: https://airquality.cpcb.gov.in/ccr/#/caaqm-dashboard-all/caaqm-landing/caaqm-data-repository  · Advanced search (per-station export): https://airquality.cpcb.gov.in/ccr/#/caaqm-dashboard-all/caaqm-landing/data  · Older mirror host: https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqm-landing | None | **Hourly** averages, **Oct 2019 – Feb 2024**. Parameters: **PM2.5, PM10, PM1** (where installed), NO2, CO, SO2, O3, **WS, WD, AT, RH**. Stations: every Delhi CAAQMS (CPCB / DPCC / IMD / IITM) **plus** the NCR ring: Gurugram, Faridabad, Sonipat, Panipat, Rohtak, Bahadurgarh, Jhajjar (HSPCB); Noida, Greater Noida, Ghaziabad, Meerut, Baghpat, Bulandshahr, Hapur (UPPCB); Bhiwadi, Alwar (RSPCB). Keep every station inside `configs/domain/delhi_ncr.yaml` plus a 50 km buffer. | `data/raw/cpcb/`. Immutable, one file per station × year, plus `MANIFEST.csv` (filename, station, period, sha256, download date). | E0, E1 (real wind), E2, E3, E5 |
| **D1-meta** | **CPCB station metadata** (name, operator, lat/lon) | Station details on the CCR pages above | None | Name, ID, operating agency, latitude, longitude, commissioning date | `data/raw/cpcb/station_coords.json`. Hand-built; record the source of each coordinate. | Domain split, maps, receptors |
| **D2** | **Wind field from CPCB station anemometers** (WS/WD from D1) | Same as D1 | None | No extra download | Built into `data/processed/wind_field.nc` (G2) and `data/processed/wind_hourly.npz` (H2) | The known wind `u = (u, v)` in the PDE, the characteristics and E1 |
| **D3** | **ICAR-IARI CREAMS — daily paddy-residue burning bulletins.** Indian Agricultural Research Institute (ICAR), New Delhi | Index: https://creams.iari.res.in/?page_id=1123  · Home: https://creams.iari.res.in/  · Example URL: https://creams.iari.res.in/Creams_website_bulletin/Fire_bulletin_2020/30.RiceResidueFireBulletin_30Oct_2020_ICAR.pdf | None | One **Rice** bulletin PDF per day, seasons **2019–2023**: daily burning-event counts per state (Punjab, Haryana, UP) and per district. **Gap:** 2022 rice bulletins are not on the index page; look under "List of All Bulletins". If they cannot be found, E4 uses the 4 available seasons and the paper says so. | `data/heldout/creams/pdf/` → `data/heldout/creams/daily_counts.csv` | **E4 only. Held out: never enters training.** |
| **D4** | **IITM Pune Decision Support System (DSS v1.0)**, MoES | Paper: https://gmd.copernicus.org/articles/17/2617/2024/ (Govardhan et al., *GMD* 17, 2617–2640, 2024)  · Live: https://ews.tropmet.res.in/dss/ | None | Seasonal Delhi PM2.5 shares. Post-monsoon: Delhi ≈34.4%, NCR ≈31%, stubble ≈7.3%, other ≈27.3%. Winter: Delhi 33.4%, NCR 40.2%, stubble 0.1%, other 26.4%. Copy the numbers, the years they cover, and the citation exactly. | `configs/experiment/e4_validation.yaml` → `comparison_targets` | External comparison only (§7.4). Not a baseline, not a training signal. |
| **D5** | **CPCB CCR, second airshed** (E5) | Same as D1 | None | Same parameters and period for **Kolkata** (WBPCB) **or** **Chennai** (TNPCB) | `data/raw/cpcb_<city>/` | E5 |

### 3.2 Optional

| # | Dataset | Link | Login | Why optional |
|---|---|---|---|---|
| D2b | **IMDAA regional reanalysis** (NCMRWF, MoES): 12 km hourly 10 m wind and boundary-layer height | https://rds.ncmrwf.gov.in/datasets | **Free account** | Covers only 1979–2020. Use it only as an independent check of the D2 wind for Oct 2019 – Dec 2020. Skip it if the project must stay strictly no-login. |
| — | **IMD gridded daily rainfall** (0.25°) | https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html | None | Rain-day flag for interpreting `λ`. Diagnostic only. |

### 3.3 Checked and NOT usable (record why, so nobody re-proposes them)

| Source | Why not |
|---|---|
| OGD "Real time Air Quality Index" (https://www.data.gov.in/resource/real-time-air-quality-index-various-locations) | Current-hour snapshot only, no 2019–2024 history. At most a lat/lon cross-check. |
| AIKosh real-time AQI and NAMP (https://aikosh.indiaai.gov.in/) | Re-hosts the same snapshot. The NAMP data are manual, twice-weekly annual summaries. |
| CPCB NAMP manual network (https://cpcb.nic.in/namp-data/) | 24-h samples about twice a week: far too sparse for an hourly inversion. |
| NITI Aayog ICED air quality (https://iced.niti.gov.in/climate-and-environment/environment/air-quality) | Annual district/state values. One sentence of context in the introduction, nothing else. |
| Kaggle CPCB mirrors, OpenCity, GitHub re-uploads | Not government-published. Removed. |
| ERA5 (ECMWF), NASA FIRMS | Not Indian. Removed. The code stays in the repo but is disabled. |

**Leakage rule for D2.** The **wind** at perimeter stations may be used, because wind is a known input. Their **pollutant** values must never enter training: they are the E2 validation target. The split holds out pollutant channels only (G6).

---

## 4. The method and what already exists on `main` (verified)

Read this section fully before writing code: every later task uses these pieces. File references are clickable in VS Code.

### 4.1 Governing equation and boundary condition

`∂C/∂t + u·∇C = K∇²C + S − λC` (advective form, `configs/config.yaml → physics.form: advective`).

On the box boundary:

- **inflow** where `u·n < 0`: `C = C_b(s, t)`, with `s` the arc-length counter-clockwise from the SW corner (`data/domain.py`);
- **outflow** where `u·n > 0`: zero normal gradient.

The PDE is **linear** in `(C_b, S, C(t=0))` for fixed wind, `K` and `λ`. Contributions C2 and C3 rest on this.

### 4.2 C1 — characteristic / Feynman–Kac loss ([physics/characteristic.py](src/bapinnsformer/physics/characteristic.py), [train/trainer.py](src/bapinnsformer/train/trainer.py))

Along a backward wind characteristic `X(τ)` ending at `(x, t)`, the exact solution of the advection–loss part is

`C(x,t) = e^{−λτ} C(X(t−τ), t−τ) + ∫₀^τ e^{−λσ} S(X(t−σ), t−σ) dσ`.

The pseudo-sequence tokens of the `advection_backward` generator lie exactly on this curve. The loss `w_char · mean((residual · T_ref/C_ref)²)` enforces the identity on the tokens:

- **Boundary coupling.** If the characteristic leaves the domain through an **inflow** edge, the exit value is `C_b(s_exit, t_exit)` at the refined exit point. This puts a direct gradient path from interior observations to the boundary net, which is the key novelty. On an outflow edge the exit value is `C` there.
- **Stochastic variant.** With `advection_jitter` the tokens are Euler–Maruyama samples of the backward SDE `dX = −u dt + √(2K) dW`, so the residual is a Monte-Carlo Feynman–Kac estimate that also carries diffusion (`char_n_mc` replicas).
- **Diagnostic.** `boundary_reach` = fraction of collocation points whose characteristic reaches the boundary inside the token window. If it is ~0, C1 cannot inform `C_b`: lengthen `seq_len·dt` or check the wind.

### 4.3 C2 — identifiability theory ([physics/greens.py](src/bapinnsformer/physics/greens.py), [eval/identifiability.py](src/bapinnsformer/eval/identifiability.py))

`build_greens` discretises the unknowns into a basis and runs the linear finite-difference model once per basis function. The basis is:

- boundary: `n_bseg` arc segments × `bwin_h`-hour windows;
- sources: `src_patches` × `swin_h`-hour windows;
- initial field: `ic_patches`.

This gives the exact linear map `y = A_b c_b + A_s s + A_0 c_0` from unknowns to station observations. On it:

| Quantity | Function | Meaning |
|---|---|---|
| Posterior mean and covariance | `bayes_linear_inversion` | Closed-form linear-Gaussian inversion: the **oracle** for any estimator given the same basis and priors |
| Principal angles | `principal_angles`, `whitened_blocks` | Angles between the prior/noise-whitened boundary range and the local range. A small minimum angle means some boundary pattern looks like a local one. |
| Confounding coefficient `ρ` | `confounding_coefficient` | Posterior correlation between the boundary contribution and the local contribution at the receptors. `|ρ| → 1` means the split is decided by the prior, not the data. |
| Boundary observability | `identifiability_report` | Fraction of **inflow-active** boundary coefficients whose variance the data reduce by ≥ 50%. Outflow coefficients are excluded: they are unidentifiable by construction. |
| Share posterior | `share_posterior` | Monte-Carlo posterior of the share (ratio of linear forms): mean, median, 5% and 95% quantiles |

**Key behaviour, seen in a smoke check (not a result):**

- **15 clustered stations, mixed wind:** the null-boundary test fails, with the estimate staying near the prior and `|ρ| ≈ 0.99`.
- **40 perimeter-biased stations, rotating wind:** the null-boundary test passes, with `|ρ| ≈ 0.73`.

E1 turns this into the RQ2 answer: **`|ρ|` and the minimum angle, computable from geometry and wind alone, predict whether the split can be trusted.**

### 4.4 C3 — receptor share by exact superposition ([eval/attribution.py](src/bapinnsformer/eval/attribution.py) `receptor_share_superposition`)

Given recovered `C_b`, `S` and `C0`, the fitted `K, λ` and the wind, the linear model runs four members: boundary-only, source-only, IC-only and full.

- `share = mean C_bnd / mean C` at receptor points (the Delhi interior stations), after `spinup_h`.
- `closure_rel` checks `C_bnd + C_src + C_ic = C_full`. It must be < 1e-10.
- `share_ic` is the part still explained by the initial field. It must be small after spin-up (§7.3).
- `sectors` splits the boundary part by compass sector of the inflow edge.

The older `transboundary_share` (inflow **mass flux** through the box) counts air that crosses the box without reaching Delhi. It is kept only as a secondary budget diagnostic named `flux_share`.

### 4.5 E1 benchmark ([eval/e1_benchmark.py](src/bapinnsformer/eval/e1_benchmark.py), [scripts/10_run_synthetic_sweep.py](scripts/10_run_synthetic_sweep.py))

The previous sweep added noise to the truth and called it an error, so no inversion happened. It has been replaced. Per cell:

1. **Truth** on a 2× finer grid: smooth background `C_b` plus a north-westerly pulse, Gaussian sources with a diurnal cycle, and a smooth initial field. The coarse inversion basis cannot represent this exactly: **no inverse crime**.
2. **Observations** at station points, hourly, with noise `noise × mean` and 20% random missingness.
3. **Inversions** with data-derived priors (never truth-derived):
   - an oracle with the true `K, λ`;
   - a misspecified oracle with `K×2, λ×0.5`.
4. **Truth share** by exact superposition on the **fine** model, so `*_share_err_vs_fine` includes discretisation error.
5. **Null tests** in every sweep: `null_boundary` (truth `C_b = 0`, so the share must be ≈0) and `null_source` (truth `S = 0`, so the share must be ≈1).
6. `prior_share` and `data_shift` columns show whether the data, not the prior, moved the answer.
7. Optional `--pinn-cells N` fits the PINNsformer on the same data and scores it with the **same** share rule (`run_pinn_cell`).

### 4.6 Other fixes already on `main`

| File | Change |
|---|---|
| `train/trainer.py` | Dimensionless losses (`C_ref`, `T_ref`, `L_ref`; `scales.mode: auto` sets `C_ref` from the observations); characteristic loss; GradNorm balancing excludes `reg`; L1 source sparsity and a `C_b` temporal-smoothness prior inside `reg`. The constructor signature is unchanged. |
| `data/synthetic.py` | Boundary condition switched by wind sign: Dirichlet on inflow, zero-gradient on outflow. Before, every edge was Dirichlet. |
| `models/baselines.py` | `physical_output = True` on `ZeroInflow` and `ClimatologicalInflow` |
| `utils/config.py`, `configs/*` | New train keys (`w_char`, `char_n_mc`, `reg_*`), `physics`, `scales` |
| `train/checkpoint.py` | Loads with `torch.load(weights_only=True)`. A full unpickle only happens with `trusted=True`, for files you created. |
| `data/qc.py` | A spike is flagged only if it is a **single-hour** glitch. Multi-hour episodes (Diwali, stubble-smoke pulses) are kept: they are the RQ4 signal. |
| `configs/data/cpcb.yaml` | `missingness_threshold: 0.25` (was 0.40) |
| `tests/test_novelty_math.py` | 18 tests: Green's linearity, principal-angle limits, oracle coverage, superposition closure, the characteristic identity, outflow boundary, checkpoint safety, the E1 null test and QC episodes |

---

## 5. Code tasks, in order

### 5.0 Merge first (blocking)

| # | Task | Proof |
|---|---|---|
| **M1** | `git checkout hkv1 && git merge main`. On conflict, apply the rules below. | `pytest -q` green on `hkv1` after the merge; `git log` shows the merge commit |
| **M2** | After the merge, `grep -rn "transboundary_share" scripts/`: every **headline** share must come from `receptor_share_superposition` (H4). The flux share may remain only under the name `flux_share`. | grep output pasted in the M2 commit message |

Conflict rules for M1:

- **Take `main`'s version** of everything under `src/bapinnsformer/physics/`, `src/bapinnsformer/eval/`, `train/trainer.py`, `train/checkpoint.py`, `data/synthetic.py`, `data/qc.py`, `models/baselines.py`, `utils/config.py`, `configs/train/`, `configs/model/`, `configs/config.yaml`, `configs/experiment/e1_identifiability.yaml`, `scripts/10_run_synthetic_sweep.py`, `tests/test_novelty_math.py`, `RUN_PLAN.md`, `PRD.md` and `CLAUDE.md`.
- **Keep `hkv1`'s version** of `scripts/01–04, 20, 21, 40, 50` and the ingestion modules, then apply F1–F9.

### 5.1 Fixes from the `hkv1` review (blocking)

| # | Change | Proof |
|---|---|---|
| F1 | Revert the false checkboxes. `PRD.md` §12.3 must not mark E0–E5 complete. | `git diff main -- PRD.md` shows no ticked experiment boxes |
| F2 | Delete the E4 dummy-data fallback in `scripts/40_validate_fires.py`. With no inputs it must exit non-zero and write nothing. | New test: no inputs → exit code ≠ 0 and no CSV |
| F3 | Zero-wind fallback in `scripts/20_fit_real.py::_load_wind_field` becomes a hard error. | New test: missing `wind_field.nc` → exception |
| F4 | Median-distance perimeter fallback in `scripts/03_build_dataset.py` becomes a hard error. Fix the buffer or domain in config instead. | New test: empty perimeter → exception |
| F5 | E5 must be a real transfer. `50_transfer_airshed.py` loads `configs/domain/<city>.yaml` **and** `data/raw/cpcb_<city>/`, runs the full schedule, and fails if the stations it loaded are Delhi's. | Run log shows the city's bounds, station IDs and n_obs |
| F6 | Training schedule comes from `configs/train/default.yaml` (`adam_steps`, `lbfgs_steps`, curriculum, balancing, `w_char`). | `config_snapshot.json` of a real run shows the full schedule |
| F7 | Sample data points stratified by station, so dense Delhi clusters do not dominate the data loss. | Per-station data-loss contribution logged |
| F8 | Call `xy_to_metric` once. | — |
| F9 | Rewrite `hkv1.readme.md` to describe what exists, not what was "confirmed complete". | — |

### 5.2 India-only data plan

| # | Change | Proof |
|---|---|---|
| G1 | **`cpcb_ingest.py`** parses CCR exports (repository files and Advanced-search exports), including WS, WD, AT, RH. Normalise to IST and then UTC; keep `source="cpcb_ccr"`; write `MANIFEST.csv` checksums. **Never impute.** | `tests/test_masking.py` green; NaN count identical before and after ingest |
| G2 | **New `data/station_wind.py`**: the four steps below, writing `data/processed/wind_field.nc` in the format `data/wind_field.py::WindField` expects. | New `tests/test_station_wind.py`: uniform wind at all stations → the same uniform field; a direction sign test (wind FROM 315° gives u > 0, v < 0) |
| G3 | **`scripts/01_build_wind_field.py`** replaces `01_fetch_era5.py` in `tasks.py data`. | `python tasks.py data` never imports `era5_ingest` |
| G4 | **`data/creams_ingest.py` + `scripts/02_fetch_creams.py`** replace FIRMS. Download the Rice bulletins 2019–2023 into `data/heldout/creams/pdf/`, parse with `pdfplumber`, write `daily_counts.csv`. | New `tests/test_creams_parse.py`: 30-Oct-2020 gives **Punjab 4266, Haryana 155, UP 51, total 4472** |
| G5 | Extend the leakage guard: `tests/test_leakage.py` forbids `creams_ingest` imports and reads of `data/heldout/` from `models/`, `physics/`, `train/`, `data/align.py`, `data/station_wind.py`, `eval/e1_benchmark.py`. | Test green, and a planted bad import makes it fail (do once, then remove) |
| G6 | Split = pollutant channels only. Perimeter WS/WD may feed D2; perimeter PM must never reach the fit set. | `tests/test_splits.py` case for exactly this |
| G7 | E1 uses real Delhi wind windows via `--wind-npz data/processed/wind_hourly.npz` (from H2). | E1 rows show `wind_source = station_interp` |
| G8 | No boundary-layer height, so `K` is a learned scalar. State this as a limitation. A BLH-modulated `K` is an ablation only if D2b is used. | Config shows the scalar `K` |
| G9 | Remove the Kaggle/OpenCity ingestion branches from `03_build_dataset.py`, and `CDS_API_*` / `FIRMS_API_KEY` from `.env.example`. | `grep -ri "opencity\|kaggle\|firms_api" scripts src` is empty except comments explaining the removal |
| G10 | Replace the in-body ERA5/FIRMS references in `PRD.md` (§1.1, §2.3, §3 diagram, §4.3, §6 E0/E1/E4, §12.2) with CPCB station wind and CREAMS. `CLAUDE.md` and the PRD novelty section are already updated. | `grep -n "ERA5\|FIRMS" PRD.md CLAUDE.md` only shows "replaced by" notes |

G2 steps:

- **(a) Wind QC.** Drop WS flatlines ≥ 6 h, WS > 25 m/s, WD outside 0–360, and stations with > 50% calm hours.
- **(b) Convert to components.** WD is the direction the wind blows **from**, so `u = −WS·sin(WD)` and `v = −WS·cos(WD)`.
- **(c) Interpolate** hourly to the model grid, inverse-distance or thin-plate RBF on **u and v separately**, then mild smoothing.
- **(d) Write** `data/processed/wind_field.nc`.

### 5.3 New tasks for the revised method

| # | Change | Proof |
|---|---|---|
| **H1** | **Timestamp convention.** CPCB hourly values are averages over `[t, t+1 h)`. Ingest must record which end the file's label marks (check one known day against the portal) and place each observation at the **mid-point** `t + 30 min` in model time. Write the convention in `data/README.md`. | Test: a label of 10:00 for the 10–11 h average maps to model time 10:30 |
| **H2** | **`scripts/05_export_wind_npz.py`**: read `wind_field.nc`, write `data/processed/wind_hourly.npz` with float arrays `u`, `v` of shape `(T, ny, nx)` (hourly, east/north components, row 0 = south) and `t0_utc`. Refuse to write if any value is NaN. | Test: the round trip reproduces `wind_field.nc` values |
| **H3** | **Windowed E2 fits.** `20_fit_real.py` gets `--window-start <ISO date> --window-hours 96 --spinup-h 24`. Each fit covers one 96-h window; consecutive windows start every 72 h, so they overlap by the 24-h spin-up. The curriculum and collocation time bounds come from the window. **Reason:** one fit per season gives the boundary net thousands of hours and too few parameters; 96 h matches the E1 setting, so E1's identifiability conclusions transfer. | Log shows the window bounds, n_obs and stations; two adjacent windows share 24 h |
| **H4** | **Receptor share in E2.** After each fit, build `GridTransport` on the domain grid (nx = ny = 33 by default, about 4.7 km) with the **fitted** `K, λ`. Evaluate the trained nets hourly to get `C_b(s, t)`, `S(x, y, t)` and `C(x, y, 0)`, and call `receptor_share_superposition` with the **interior Delhi stations as receptors** and `spinup_h = 24`. Write `share`, `share_src`, `share_ic`, `closure_rel`, `sectors` and `flux_share` (old function) to `metrics.json`. The reference implementation is the second half of `e1_benchmark.run_pinn_cell`. | `metrics.json` has all fields; `closure_rel < 1e-10` |
| **H5** | **Linear Bayesian inversion on real data** (new `scripts/22_linear_inversion.py`; baseline + real-network RQ2 diagnostic). Per window: `build_greens` on the real wind grid and real station coordinates, with the E1 basis sizes. Priors: data-derived as in `e1_benchmark.run_cell` (copy that block; never use model outputs as priors). Fit stations only. Outputs per window: `identifiability_report` (angles, `ρ`, observability, share posterior). Run on the 3×3 grid `K ∈ {0.5, 1, 2} × K_fit` and `λ ∈ {0.5, 1, 2} × λ_fit`. | One CSV row per (window, K, λ) |
| **H6** | **Season aggregation** (new `eval/aggregate.py`). Season share = obs-weighted mean of window shares (post-spin-up hours only). Uncertainty: report the across-seed spread and the across-window spread separately; for H5 also report the mean of the window posteriors' 5–95% ranges. Never pool seeds into "more data". | Unit test on a toy table |
| **H7** | **E3 ablation grid.** Variants `uniform_forward`, `uniform_backward`, `advection_backward`, `advection_jitter` × `w_char ∈ {0, 1}`. `w_char > 0` is only meaningful for the two advection variants: the trainer needs an advection generator, so for the uniform variants run `w_char = 0` only and say so. Add `L ∈ {3, 5, 9}` and `activation ∈ {wavelet, tanh}` on `advection_backward`, and a physics-off ablation (`w_pde = w_char = 0`). | `configs/experiment/e3_ablation.yaml` lists exactly this grid |
| **H8** | **Perimeter validation target.** For each withheld perimeter station, compare observed PM with the model's `C` at that point, hourly and as daily means. Where the station sits within 10 km of an **inflow** edge, also compare with `C_b` at the nearest boundary point. That is a direct check of the recovered inflow. | `metrics.json` has `perimeter_*` fields per station |
| **H9** | **Checkpoint safety in scripts.** Every `load_checkpoint` call in `scripts/` uses the default `trusted=False`. | `grep -rn "trusted=True" scripts` is empty |

---

## 6. Run order

All scripts take `--config <yaml>`, log a run-id under `results/runs/`, and are wrapped by `python tasks.py <target>`. `results/` is git-ignored: never commit it.

```powershell
# 0) Sanity, no data (laptop)
pytest -q                                   # all green, incl. the new F, G and H tests
python tasks.py smoke                       # plumbing only. NOT a result.
python scripts/10_run_synthetic_sweep.py --config configs/experiment/e1_identifiability.yaml --quick --results-root scratch_results
#   quick E1 check: about 1 min on CPU; ignore its numbers

# E0) Data (laptop CPU)
#   manual: download D1 (+ D1-meta) into data/raw/cpcb/, then fill MANIFEST.csv
python tasks.py data                        # ingest → QC → station wind field → domain → split
python scripts/05_export_wind_npz.py        # H2
python scripts/04_characterize_data.py      # missingness, wind QC, entropy per window, domain map
python scripts/02_fetch_creams.py           # D3 → data/heldout/creams/ ; do NOT open the CSV until E4

# E1) Synthetic identifiability. CPU for the oracle grid, GPU for the PINN cells. FINISH FIRST: it is the publishable fallback.
python scripts/10_run_synthetic_sweep.py --config configs/experiment/e1_identifiability.yaml --wind-npz data/processed/wind_hourly.npz
python scripts/10_run_synthetic_sweep.py --config configs/experiment/e1_identifiability.yaml --wind-npz data/processed/wind_hourly.npz --pinn-cells 24 --pinn-steps 5000 --device cuda --run-id e1_pinn
#   then §6.2 (w_reg calibration), on synthetic cells only

# E2) Real Delhi-NCR (Kaggle GPU). Pilot first: one window, PM2.5, post-monsoon 2020, 3 seeds
python scripts/20_fit_real.py --config configs/experiment/e2_real.yaml --season postmonsoon --pollutant PM2.5 --window-start 2020-11-01 --window-hours 96 --spinup-h 24 --seed 0
python scripts/22_linear_inversion.py --config configs/experiment/e2_real.yaml --window-start 2020-11-01      # H5, CPU
#   pilot must pass §7.3 before the fan-out:
python tasks.py e2

# E3) Ablations (Kaggle GPU) on the pilot windows only
python tasks.py e3

# E4) Held-out fire validation (CPU)
pytest tests/test_leakage.py -q             # must pass immediately before E4; paste the output into the REPORT
python tasks.py e4

# E5) Transfer (Kaggle GPU)
python tasks.py e5

# Paper artifacts (laptop)
python tasks.py figures ; python tasks.py tables
```

### 6.1 E1 grid (default `configs/experiment/e1_identifiability.yaml`)

- **Grid:** 4 station counts × 3 archetypes × 3 entropy bins × 2 windows per bin × 4 noise levels × 3 replicates × 3 kinds (full + 2 null) = **2592 oracle cells**.
- **Cost:** each cell takes a few seconds on CPU at `n_coarse = 17`. Measure 10 cells first, then run the grid on the laptop overnight; split by `--seed` if needed.
- **PINN cells:** 24, chosen to cover both identifiable and non-identifiable regions. Pick them **after** the oracle grid, half from cells where both null tests pass and half where they fail, and record the list in the REPORT.

### 6.2 Calibrating the prior weight `w_reg` (E1 only, then freeze)

Run the 24 PINN cells for each `w_reg ∈ {1e-3, 1e-2, 1e-1}`, then pick the value with the lowest **median share error across the full and null cells together**.

- Write the chosen value into `configs/train/default.yaml` with a comment pointing to the E1 run-id.
- It is **frozen** for E2–E5. Calibrating on real data is forbidden: it would leak the answer into the method.

### 6.3 Compute budget (≈150 GPU-h total)

Measure the pilot fit time `T_fit` (one window, one seed, the full schedule on one T4) before planning the fan-out.

| Experiment | GPU-h | How it is spent |
|---|---|---|
| E1 PINN cells + `w_reg` calibration | 15 | 24 cells × 3 `w_reg` values |
| E2 | 70 | **PM2.5 first:** all post-monsoon windows of every year (≈ 20 windows/season-year) × 1 seed, plus 3 seeds on 6 fixed windows. Then winter. PM10 and PM1 get 6 windows each, for the λ ordering only. |
| E3 | 35 | Pilot windows only, 3 seeds per variant |
| E5 | 15 | One season, PM2.5 |
| Reserve | 15 | Reruns after failures |

If `T_fit × planned fits` exceeds a row, cut windows from the **end of the priority list**: PM1, then PM10, then winter years, then post-monsoon years, in that order. Never cut seeds on the 6 fixed windows. Track usage in `results/runs/quota.json`.

---

## 7. How to know a result is good

Every gate below is pass/fail. A step is **done** only when all its gates pass and `results/<exp>/REPORT.md` exists (§8). If a gate fails, the step is not done: report the failure; never tune until it passes silently.

### 7.0 Red flags: stop and report immediately

- In any E2–E5 run, a log line containing `fallback`, `dummy`, `constant(0.0`, `smoke`, `quick` or `synthetic`.
- Perimeter R² > 0.95, or any suspiciously perfect metric. Suspect leakage first.
- Recovered `C_b` constant in space and time, or `S` with all its mass in one or two cells.
- NaN or Inf in the loss history, or a balancing weight on `pde` or `char` driven to ~0 (physics switched off).
- `boundary_reach < 0.05` in a run with `w_char > 0`: C1 is not touching the boundary.
- `closure_rel > 1e-10`, or `share_ic > 0.05` after spin-up.
- Any FIRMS, ERA5, Kaggle-dataset or OpenCity file under `data/` used by a run.

### 7.1 E0 — data

| Gate | Pass condition |
|---|---|
| Provenance | Every raw file is in `MANIFEST.csv` with a sha256. Rebuilding from raw reproduces `data/processed/` with the same split hash. |
| No imputation | Missing hourly PM count per station is identical in raw and processed. |
| Station inclusion | A station enters a window fit only if it has ≥ 75% valid hours in that window (`missingness_threshold: 0.25`). List included and excluded stations with their coverage. |
| Enough stations | ≥ 15 interior stations pass for PM2.5 in each season. ≥ 6 perimeter stations pass, on ≥ 3 of 4 sides, **including the NW**. Otherwise, say E2's primary validation is weak and lean on LOISO. |
| Physical sanity | PM2.5 ≤ PM10 in ≥ 95% of co-located valid hours; units µg/m³. |
| QC audit | The fraction of hours flagged `spike` is reported per station and must be < 1%. Plot the 5 largest flagged spikes and confirm by eye they are single-hour glitches. |
| Wind quality | Leave-one-station-out interpolation check per season: direction MAE < 45°, speed RMSE < 1.5 m/s. Report the calm-hour %. The dominant post-monsoon direction must be **from the north-west**; if not, a sign convention is wrong. |
| Wind entropy | Distribution of 96-h window directional entropy per season (it places real windows on the E1 surface). |
| Map | `domain_map.png` shows interior and perimeter stations correctly; check visually against a real NCR map. |

### 7.2 E1 — synthetic identifiability (truth known)

| Gate | Pass condition |
|---|---|
| Unit tests | `tests/test_novelty_math.py` green: Green's linearity, closure, the characteristic identity and oracle coverage are all proven there. |
| Real wind | `wind_source = station_interp` in every reported row. Synthetic-wind rows may appear only in an appendix, labelled. |
| Replicates | Report median and IQR per cell over the 3 replicates × 2 windows, never one run. |
| Oracle calibration | Over all `full` cells with true `K, λ`, the fraction with `oracle_share_covered_90 = True` is in **[0.80, 0.97]**. If it is lower, the posterior is over-confident: report it and do not use the intervals in E2. The `*_vs_fine` coverage is reported next to it; it is expected to be lower, because it includes representation error. |
| Null tests define the identifiable region | Per (n, archetype, entropy bin), the pass rate of `null_boundary` and `null_source` at `null_tol = 0.10` (from `summary_<run>.json`). A cell is **identifiable** if both pass in ≥ 80% of replicates. This map is the RQ2 result. |
| Diagnostic predicts identifiability | ROC-AUC of `|oracle_confounding_rho|` (and separately `oracle_min_angle_deg`) for predicting null-test failure is ≥ 0.80. This is the C2 claim: if it fails, the a-priori diagnostic is not useful; report that honestly. |
| Prior vs data | In identifiable cells the median `oracle_data_shift` is ≥ 0.10. In non-identifiable cells, report it: values near 0 show the prior decided the answer. |
| Trends | Report the median share error vs station count per archetype and vs noise. Report the Spearman ρ between window entropy and share error with its p-value (`summary` has the correlation; compute the p-value). A non-negative result is still reportable. |
| Misspecification | Report the `misspec_*` share-error increase over the oracle. This is the honest cost of wrong `K, λ`. |
| PINN vs oracle | On the 24 PINN cells: the median of `pinn_share_err_vs_fine / oracle_share_err_vs_fine`. "The PINNsformer approaches the oracle" may be claimed only if this is ≤ 1.5 in identifiable cells. Also run the PINN null tests (2 cells each) and report them. |
| Wind-shuffle control | On 6 identifiable cells, shuffle the wind's hour order. The oracle share error must be > 1.5× the unshuffled error; if not, the inversion is not using transport. |

### 7.3 E2 — real Delhi-NCR

| Gate | Pass condition |
|---|---|
| Real inputs only | `config_snapshot.json` and the log show the real `wind_field.nc`, the real split hash, n_obs, the fit-station list and the window bounds. No fallbacks. |
| Convergence | Total, PDE and char losses plateau. The final normalised PDE residual is ≤ 1e-2. The L-BFGS stage ran. `boundary_reach` is reported. |
| Superposition | `closure_rel < 1e-10` and `share_ic < 0.05` in every window. If `share_ic` is larger, raise `spinup_h` to 36 and rerun. Never quote a window that fails. |
| Held-out skill | RMSE, MAE and R² against the **withheld perimeter stations** (hourly and daily), for our model **and every baseline on the same split**: zero-inflow, climatological inflow, MLP-PINN, uniform-pseudo-sequence PINNsformer, back-trajectory regression, **and the H5 linear Bayesian inversion**. Include the H8 `C_b` check at inflow-side stations. |
| Beat baselines | Day-block bootstrap (1000 resamples) of the RMSE difference vs zero-inflow and vs climatological inflow: the 95% CI is entirely below 0. If not, report it: the boundary recovery adds no information. |
| LOISO | Leave-one-interior-station-out error reported. |
| Seeds | On the 6 fixed windows, 3 seeds each. The share spread is ≤ 5 percentage points; if larger, the decomposition is not stable enough to quote. |
| Identifiability of the real network | For every window, H5 gives `ρ` and the minimum angle. Place each window on the E1 map. A window share is **quotable** only if its (n, geometry, entropy) falls in the E1-identifiable region **and** its `|ρ|` is below the E1 ROC threshold. Report the fraction of quotable windows per season; the season share uses quotable windows only. |
| PINN vs linear inversion | Per quotable window, the PINN share lies inside the H5 posterior 5–95% range at the fitted `K, λ`, in ≥ 80% of windows. If not, explain (nonlinearity, basis resolution) before quoting either. |
| Physical plausibility | λ ordering PM1 < PM2.5 < PM10 holds. `K > 0` and within the literature range for horizontal eddy diffusivity at 1–10 km scale (cite the source). `flux_share` and the domain mass balance close within ±10%. |
| K–λ sensitivity | From H5's 3×3 grid, report how far the share moves. Quote the share with this range next to it. |
| Wind sensitivity | Pilot window rerun with the wind rotated ±15° and scaled ±20%. Report the share movement; the share is quotable only if not dominated by it. |

### 7.4 Comparison with IITM DSS (D4): do it correctly

- **Why "outside-NCR".** Our domain box (76.40–77.90 E, 27.80–29.20 N) contains Delhi and most of NCR. Our "boundary inflow" therefore corresponds to DSS's **outside-NCR** contributions, i.e. "other" + "stubble":
  - **post-monsoon ≈ 27.3 + 7.3 = 34.6%**;
  - **winter ≈ 26.4 + 0.1 = 26.5%**.
- **Do not compare with DSS's "non-Delhi" total** (Delhi complement ≈ 66% / 67%). NCR districts are mostly **inside** our box, so they are part of our local `S`.
- **Compare like with like:**
  - our season share uses quotable windows only, at the **Delhi interior receptors**, PM2.5;
  - restrict to the same years DSS covers, if the paper states them;
  - our `share_src + share_ic` is compared with DSS Delhi + NCR (≈65.4% / 73.6%).
- **Partial overlap.** The box does not match the NCR boundary exactly: parts of NCR (e.g. Alwar, far Meerut districts) lie outside it. Write this in the REPORT, with the area fraction of NCR inside the box.
- **Agreement is not required.** A divergence > 15 percentage points needs a written explanation in the REPORT before the number is shown to anyone.

### 7.5 E3 — ablations

| Gate | Pass condition |
|---|---|
| Fair comparison | All variants: parameter counts within ±5%, same windows, same split, same steps, 3 seeds. |
| A win means a win | "Advection-aligned is better" is claimed only where the mean improvement exceeds 2× the seed SD at the same `L`. The "shorter L suffices" claim must come from the accuracy-vs-L curve, with time and peak-memory curves alongside. |
| C1 effect | `advection_backward` with `w_char = 1` vs `w_char = 0`: report the share, the perimeter RMSE and `boundary_reach`. This is the direct test of C1. |
| Physics-off | Removing the PDE and char terms must change the recovered `C_b` materially; if not, report that the physics is not doing anything. |

### 7.6 E4 — held-out fire validation (CREAMS)

| Gate | Pass condition |
|---|---|
| Leakage | `pytest tests/test_leakage.py -q` passes immediately before the run; output pasted in the REPORT. |
| Parse correctness | `test_creams_parse.py` green; days per season reported. |
| Pre-registration | Before opening `daily_counts.csv`, commit `e4_validation.yaml` with lag 1 day as primary and the NW sector (from `receptor_share_superposition(...)["sectors"]`) as the predictor. The commit hash goes in the REPORT. |
| Primary test | Spearman ρ between the daily NW-sector boundary contribution at Delhi receptors and the daily Punjab + Haryana CREAMS count, Oct–Nov, lags 0–3: ρ, p and n per lag, with lag 1 primary. |
| Placebo | The same test with the SE sector should be clearly weaker. If it is as strong, the signal is seasonal co-variation, not transport. |
| Meteorology control | Repeat the primary test on NW-wind days only. |

### 7.7 E5 — transfer

| Gate | Pass condition |
|---|---|
| Really another city | The log shows the second city's bounds, station IDs and data files. |
| Same gates | §7.1 and §7.3 are re-applied and reported, even if weaker. E1-style oracle diagnostics (H5) are reported for the new network. |

---

## 8. Deliverables and the REPORT template

| Step | Must exist before the step counts as done |
|---|---|
| E0 | `data/README.md` (sources, dates, login/captcha seen, timestamp convention H1) · `MANIFEST.csv` · `results/e0_data_quality/REPORT.md` |
| E1 | `results/e1_identifiability/sweep_*.csv` + `summary_*.json` · identifiable-region map figure · ROC figure for `|ρ|` · `REPORT.md` |
| E2 | Per window: `metrics.json` (H4 fields, perimeter skill, baselines) · H5 CSV · season aggregation (H6) · DSS table (§7.4) · `REPORT.md` |
| E3 | Accuracy / time / memory vs L per variant, C1 effect table · `REPORT.md` |
| E4 | Pre-registration commit hash, then ρ/p/n table, placebo, NW-only · `REPORT.md` with the leakage-test output |
| E5 | Same as E0 + E2 for the second city |
| Always | `paper/notes/claims_ledger.md`: every claim → results file → figure/table |

Every `REPORT.md` has these headings, in this order:

1. **What was run:** command lines, run-ids, git commit, config hash, split hash.
2. **Inputs:** data files, stations, windows.
3. **Gates:** a table with each gate from §7, its value, and PASS or FAIL.
4. **Numbers:** median and IQR, or mean ± SD with n. Every number has its run-id.
5. **What failed or looked odd.**
6. **What this does and does not support.**

---

## 9. Integrity and security rules

- **Held-out data.** CREAMS stays under `data/heldout/` and is read only by `scripts/40_validate_fires.py`. `tests/test_leakage.py` enforces this. Never move it.
- **Checkpoints.** Load only with `load_checkpoint(...)` and the default `trusted=False`, which uses `weights_only=True`. Never load a `.pt` file you did not create. A pickle can execute code.
- **Configs.** YAML is read with `yaml.safe_load` only. Never switch to `yaml.load`.
- **Subprocess.** `tasks.py` calls subprocesses with argument lists, never `shell=True` with interpolated strings. Keep it that way.
- **No secrets.** The India-only plan needs no API keys. Do not put tokens in notebooks or configs, and never commit `.env` or `kaggle.json`.
- **Data provenance.** Only files listed in `MANIFEST.csv` with a sha256 may be used. Data received by chat, email or a third-party upload is not used until it is re-downloaded from the official link in §3.
- **Results.** `results/` is git-ignored and regenerated by scripts. Nobody edits numbers by hand in `results/`, `paper/figures/` or `paper/tables/`.
- **Seeds.** Every run is reproducible from its config and seed. E1 cell seeds are sha256-derived (`_cell_seed`).

---

## 10. Known limitations (state these in the paper)

- **Wind from station anemometers**, not a reanalysis. Urban siting and calm hours add error. Mitigated by the wind QC, the leave-one-station-out check (§7.1) and the wind-sensitivity runs (§7.3).
- **No boundary-layer height**, so `K` is a scalar and the model is 2-D, vertically integrated. Winter inversions are where this is weakest: report inversion periods separately.
- **Linear transport.** The share is defined by superposition for the fitted linear model. Secondary aerosol chemistry is not represented; it is absorbed into `S` and `λ`.
- **Basis resolution.** H5 and E1 use a coarse basis (12 boundary segments × 6 h, 4×4 source patches × 24 h). E1's `*_vs_fine` errors quantify what this costs.
- **CREAMS gives event counts, not fire radiative power.** E4 correlates against counts.
- **2022 rice bulletins** may be missing (§3, D3).
- **The domain box ≠ the NCR boundary** (§7.4).
- **Uneven CPCB network:** dense in Delhi, sparse in the outer NCR. E1 maps what that density can and cannot identify.

---

## 11. Audit trail

- **2026-09-12:** scaffold, two review passes (22 findings fixed), `pytest` 77/77.
- **2026-10-03, `hkv1` review:** ingestion, E0 build, characterisation and `20_fit_real` wiring are sound. Blocking issues are F1–F9 (§5.1).
- **2026-10-03, maths and correctness review on `main`:**
  - found: unscaled loss terms; a fabricated E1 (noise added to the truth, with no inversion); an all-Dirichlet synthetic boundary; a share definition counting pass-through air; a meaningless boundary smoothness term; `torch.load(weights_only=False)`; a spike QC that could delete real smoke episodes; a misread DSS comparison (non-Delhi instead of outside-NCR);
  - fixed: all of the above;
  - added: C1, C2 and C3 (§4);
  - `pytest` 97/97.
- **Security review, same date:** no high-confidence vulnerabilities. subprocess calls use argument lists, YAML uses `safe_load`, and no secrets are logged. The only finding (unsafe checkpoint deserialisation, low severity) is fixed.
