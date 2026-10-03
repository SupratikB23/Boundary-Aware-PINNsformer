# RUN_PLAN — Boundary-Aware PINNsformer

The operator manual: what to download, what to fix, what to run, in what order, and how to tell whether a result is real.
PRD.md is authoritative for architecture. **Where this file and PRD.md disagree about data sources, this file wins** (revised 2026-10-03: India-only government data).

Status (2026-10-03): scaffold complete, `pytest` green (79 on `hkv1`). Pipeline wiring exists on branch `hkv1`.
**No real experiment has been run yet. There are no results.**

---

## 0. Brief

**What we are doing.** We estimate how much of Delhi-NCR's particulate pollution enters the region from outside, using only government monitoring-station measurements and the laws of physics (advection–diffusion), with no emission inventory. A physics-informed transformer (PINNsformer) learns three things together: the pollution field, the local emission sources, and the unknown inflow along the domain boundary. The boundary inflow is the scientific output.

**What changed in this revision.**
1. **Every dataset is now an Indian government source.** No Kaggle datasets, no ERA5 (European), no NASA FIRMS (US), no OpenCity. Kaggle is still used for **GPU compute only**, never as a data source.
2. **Wind field:** ERA5 is replaced by **hourly wind measured at the CPCB stations themselves**, interpolated to a grid (§2, D2).
3. **Held-out fire validation:** NASA FIRMS is replaced by **ICAR-IARI CREAMS daily crop-residue-burning bulletins** (§2, D3).
4. The `hkv1` review findings are now mandatory fixes (§3) that must land before any number is reported.

**Rule zero.** Only numbers produced by a completed run on real CPCB data, with all of §5's checks passed, may go into a report, a slide, or the paper. A fallback, dummy, smoke or synthetic stand-in is never a result.

---

## 1. Environment (do once)

```powershell
# Windows PowerShell, from repo root "BBM Research PAPER/"
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .            # installs src/bapinnsformer
copy .env.example .env      # no API keys are needed for the India-only data plan
pytest -q                   # expect all green
```

- Add `pdfplumber` to `requirements.txt` (needed to parse the CREAMS bulletins, D3). `cdsapi` is no longer needed.
- `pyproject.toml` pins `torch==2.3.1`. Do NOT `pip install -r requirements.txt` over a working environment, because it downgrades torch, numpy and pandas. Before GPU runs, create a FRESH venv.
- GPU: RTX 3050 (dev) / Kaggle dual T4 (compute only). Keep the model under 0.5 M params with `L=5` (under 3 GB). On Kaggle, run two independent fits concurrently, never data-parallel (PRD §9).
- Moving data to Kaggle: upload our own processed CPCB files as a **private** Kaggle dataset purely to mount them in the notebook. The data is still CPCB data. Cite CPCB, never Kaggle.

---

## 2. Datasets — India, government, public

All sources below were researched on 2026-10-03.
- **Login:** CPCB, CREAMS, IMD and IITM need no account. IMDAA (D2b) is the only exception and is optional.
- **Access caveat:** the CPCB CCR portal blocks automated access from outside India, so it could not be machine-checked from here. Its public, no-login status is established by its use as the stated open data source in recent peer-reviewed Delhi studies, e.g. Nandi et al., *npj Clean Air* 2026 (IIT Delhi), [doi:10.1038/s44407-026-00065-6](https://www.nature.com/articles/s44407-026-00065-6).
- **First action:** open each link once in a browser from India and record in `data/README.md` what you saw: date, any login prompt, any captcha.

### 2.1 Required

| # | Dataset (owner) | Link | Login | What to take | Lands in | Used for |
|---|---|---|---|---|---|---|
| **D1** | **CPCB CCR — Continuous Ambient Air Quality Monitoring Stations (CAAQMS)**. Central Pollution Control Board, MoEFCC | Data repository: https://airquality.cpcb.gov.in/ccr/#/caaqm-dashboard-all/caaqm-landing/caaqm-data-repository  · Advanced search (per-station export): https://airquality.cpcb.gov.in/ccr/#/caaqm-dashboard-all/caaqm-landing/data  · Mirror host used in older papers: https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqm-landing | None | **Hourly** averages, **Oct 2019 – Feb 2024**. Parameters: **PM2.5, PM10, PM1** (where installed), NO2, CO, SO2, O3, **WS, WD, AT, RH**. Stations: every Delhi CAAQMS (CPCB / DPCC / IMD / IITM-operated) **plus** the NCR ring: Gurugram, Faridabad, Sonipat, Panipat, Rohtak, Bahadurgarh, Jhajjar (HSPCB); Noida, Greater Noida, Ghaziabad, Meerut, Baghpat, Bulandshahr, Hapur (UPPCB); Bhiwadi, Alwar (RSPCB). Keep all that fall inside `configs/domain/delhi_ncr.yaml` plus a 50 km buffer. | `data/raw/cpcb/` (immutable, one file per station × year, plus `MANIFEST.csv` with filename, station, period, sha256, download date) | E0, E1 (real wind sequences), E2, E3, E5 |
| **D1-meta** | **CPCB station metadata** (name, operator, lat/lon) | Station details on the CCR dashboard pages above | None | Name, ID, operating agency, latitude, longitude, commissioning date | `data/raw/cpcb/station_coords.json` (hand-built and checked; record the source of each coordinate) | Domain split, maps |
| **D2** | **Wind field from CPCB station anemometers (WS/WD from D1)** | Same as D1 | None | No extra download: WS and WD come in the D1 exports | Built by the pipeline into `data/processed/wind_field.nc` | The known `u=(u,v)` in the PDE and in the backward characteristics |
| **D3** | **ICAR-IARI CREAMS — daily paddy-residue burning bulletins.** Consortium for Research on Agroecosystem Monitoring and Modeling from Space, Indian Agricultural Research Institute (ICAR), New Delhi | Index: https://creams.iari.res.in/?page_id=1123  · Home: https://creams.iari.res.in/  · URL pattern, e.g. https://creams.iari.res.in/Creams_website_bulletin/Fire_bulletin_2020/30.RiceResidueFireBulletin_30Oct_2020_ICAR.pdf | None (direct PDF links) | One PDF per day, **Rice** (paddy) bulletins only, seasons **2019, 2020, 2021, 2022, 2023**. Each bulletin gives the satellite-detected burning-event count for the day **per state** (Punjab, Haryana, UP) and **per district**, plus cumulative totals. **Gap:** 2022 rice bulletins are not on the index page (2019: 58, 2020: 65, 2021: 79, 2023: 76 rice bulletins listed). Look under "List of All Bulletins" on the same page. If 2022 cannot be found, E4 runs on the 4 available seasons and the paper says so. | `data/heldout/creams/pdf/` (raw) → `data/heldout/creams/daily_counts.csv` (date, state, district, events) | **E4 only. Held out: never enters training.** |
| **D4** | **IITM Pune Decision Support System (DSS v1.0)** — published Delhi source shares (MoES) | Paper: https://gmd.copernicus.org/articles/17/2617/2024/ (Govardhan et al., *GMD* 17, 2617–2640, 2024)  · Live system: https://ews.tropmet.res.in/dss/ | None | Seasonal shares: post-monsoon Delhi ≈34.4%, NCR ≈31%, stubble ≈7.3%, other ≈27.3%; winter 33.4 / 40.2 / 0.1 / 26.4%. Copy the numbers and citation exactly. | `configs/experiment/e4_validation.yaml` → `comparison_targets` | External comparison table only. Not a baseline, not a training signal |
| **D5** | **CPCB CCR, second airshed** (E5) | Same portal as D1 | None | Same parameters and period for **Kolkata** (WBPCB stations) **or** **Chennai** (TNPCB stations) | `data/raw/cpcb_<city>/` | E5 |

### 2.2 Optional (only if needed and allowed)

| # | Dataset | Link | Login | Why it is optional |
|---|---|---|---|---|
| D2b | **IMDAA regional reanalysis** (NCMRWF, MoES): 12 km, **hourly**, 10 m wind and boundary-layer height | https://rds.ncmrwf.gov.in/  · Datasets: https://rds.ncmrwf.gov.in/datasets | **Free account required** (https://rds.ncmrwf.gov.in/login) | Covers **only 1979–2020**, so just Oct 2019 – Dec 2020 of our window. It is the only Indian gridded wind + boundary-layer-height product. Use it only as an independent check on the D2 station-wind field over 2019–2020, never as the main wind source. Skip it if the project must stay strictly no-login. |
| — | **IMD gridded daily rainfall** (0.25°) | https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html | None | Rain flag for wet-scavenging days (diagnostic for λ). Daily only, so not a model input. |

### 2.3 Checked and NOT usable for modelling (record why, so nobody re-proposes them)

| Source | Link | Why not |
|---|---|---|
| Open Government Data (OGD) — "Real time Air Quality Index from various locations" | https://www.data.gov.in/resource/real-time-air-quality-index-various-locations | **Current-hour snapshot only** (min/max/avg per pollutant per station), served through an API. There is **no history for 2019–2024**. Use it at most to cross-check station lat/lon. |
| AIKosh (IndiaAI) — same real-time AQI feed + NAMP | https://aikosh.indiaai.gov.in/home/datasets/details/real_time_air_quality_index_from_various_locations.html | It re-hosts the same real-time snapshot. The NAMP datasets are **manual, twice-weekly, annual-summary** values (e.g. PM10 2009). There is no hourly history in our window. |
| CPCB NAMP (manual network) | https://cpcb.nic.in/namp-data/ | Manual 24-h samples, about twice a week, so far too sparse for an hourly PDE inversion. Context only. |
| NITI Aayog — India Climate & Energy Dashboard, air quality | https://iced.niti.gov.in/climate-and-environment/environment/air-quality | **District/state annual** concentrations sourced from CPCB. Useful for one sentence of context in the paper's introduction, nothing else. |
| Kaggle CPCB mirrors, OpenCity, GitHub re-uploads | — | Not government-published. **Removed from the pipeline.** |
| ERA5 (ECMWF), NASA FIRMS | — | Not Indian. **Removed.** Their code paths stay in the repo, disabled (§3). |

**Leakage note for D2:** using the **wind** measured at perimeter stations is allowed, because wind is a known input. Using their **pollutant** values in training is forbidden: those are the E2 validation target. The split must hold out pollutant channels only, and §5 checks this.

---

## 3. Mandatory code changes before any real run

Work on `hkv1` and merge `main` into it first, so this RUN_PLAN is present. Each item ends with a test or a check that proves it is done.

### 3.1 Fixes from the `hkv1` review (blocking)

| # | Change | Proof it is done |
|---|---|---|
| F1 | **Revert the false checkboxes.** `PRD.md` §12.3 and `RUN_PLAN.md` must not mark E0–E5 as complete; nothing has run. | `git diff main -- PRD.md` shows no ticked experiment boxes |
| F2 | **Delete the E4 dummy-data fallback** in `scripts/40_validate_fires.py`. With no inputs it must exit non-zero with a clear message. A generated "inflow = 0.5 × fire" correlation is fabricated validation. | New test: running without inputs → exit code ≠ 0, and no CSV written |
| F3 | **Zero-wind fallback → hard error** in `scripts/20_fit_real.py::_load_wind_field`. Missing or unreadable wind must stop the run. Zero wind silently turns the model into pure diffusion. | New test: missing `wind_field.nc` → exception |
| F4 | **Median-distance perimeter fallback → hard error** in `scripts/03_build_dataset.py`. If no stations fall in the perimeter buffer, fix the buffer or domain in config. Never relabel half of Delhi as "perimeter". | New test: empty perimeter → exception |
| F5 | **E5 must be a real transfer.** `50_transfer_airshed.py` must load `configs/domain/<city>.yaml` **and** `data/raw/cpcb_<city>/`, run the full fit (not `--smoke-steps 1`), and fail if the data it loaded is Delhi's. | Run log shows the city's domain bounds, station IDs and n_obs |
| F6 | **Training settings come from config:** Adam steps, then L-BFGS, the curriculum and gradient balancing are all wired from `configs/train/default.yaml`. The current 50-step default is a smoke value only. | `config_snapshot.json` of a real run shows the full schedule |
| F7 | **Sample data points stratified by station**, so dense Delhi clusters do not dominate the data loss. | Per-station contribution to the data loss is logged |
| F8 | Call `xy_to_metric` once (it is currently called twice). | — |
| F9 | Rewrite `hkv1.readme.md` so it describes what exists, not what was "confirmed complete". | — |

### 3.2 Changes for the India-only data plan

| # | Change | Proof it is done |
|---|---|---|
| G1 | **`cpcb_ingest.py`**: parse CCR exports (both the repository files and Advanced-search exports) including **WS, WD, AT, RH**. Normalize to IST and then to UTC, keep a `source="cpcb_ccr"` column, write `MANIFEST.csv` checksums. **Never impute.** | `tests/test_masking.py` still green; the row count of NaNs is identical before and after ingest |
| G2 | **New `src/bapinnsformer/data/station_wind.py`** with these steps: (a) QC the winds: drop WS flatlines ≥ 6 h, WS > 25 m/s, WD outside 0–360, stations with > 50% calm hours; (b) convert WS/WD to u/v. Direction convention: WD is the direction the wind blows **from**, so u = −WS·sin(WD), v = −WS·cos(WD); (c) interpolate hourly to the model grid using inverse-distance or thin-plate RBF on **u and v separately**, then a mild spatial smoothing; (d) write `data/processed/wind_field.nc` with the **same variables and coordinates the existing `WindField` loader expects**, so the model code does not change. | New `tests/test_station_wind.py`: a known uniform wind at all stations → the same uniform field everywhere; a wind-direction sign test |
| G3 | **New script `scripts/01_build_wind_field.py`** replaces `01_fetch_era5.py` in `tasks.py data`. The ERA5 code stays in the repo but is not called. | `python tasks.py data` never imports `era5_ingest` |
| G4 | **New `src/bapinnsformer/data/creams_ingest.py` + `scripts/02_fetch_creams.py`** replace FIRMS. Download the Rice bulletins for 2019–2023 (Oct–Nov plus late Sep) into `data/heldout/creams/pdf/`, parse the state and district daily counts with `pdfplumber`, write `daily_counts.csv`. Spot-check: 30-Oct-2020 must give **Punjab 4266, Haryana 155, UP 51, total 4472** (read from that day's bulletin on 2026-10-03). | New `tests/test_creams_parse.py` asserts those four numbers |
| G5 | **Extend the leakage guard:** `tests/test_leakage.py` must also forbid any import of `creams_ingest` or any read of `data/heldout/creams/` from `models/`, `physics/`, `train/`, `data/align.py`, `data/station_wind.py`. | Test green, and a deliberately planted bad import makes it fail (do this once, then remove it) |
| G6 | **Split = pollutant channels only.** Perimeter stations' WS/WD may feed D2. Their PM values must never reach the fit set. | `tests/test_splits.py` gets a case for exactly this |
| G7 | **E1 synthetic winds** are drawn from the D2 station-wind field (real Delhi wind sequences), not from ERA5. | E1 config points to `wind_field.nc` |
| G8 | **No boundary-layer height** is available without IMDAA, so `K` is a learned scalar (`configs/model` and `params.py`). Write this up as a limitation. If D2b is used, BLH-modulated `K` becomes an ablation over 2019–2020 only. | Config shows `K: scalar` |
| G9 | Remove the Kaggle/OpenCity ingestion branches from `03_build_dataset.py`. Remove `CDS_API_*` and `FIRMS_API_KEY` from `.env.example`. | `grep -ri "opencity\|kaggle\|firms_api" scripts src` is empty (except comments explaining removal) |
| G10 | Update the data rows in `PRD.md` (§1.1, §2.3, §3 diagram, §4.3 `align/wind_field/stats/synthetic`, §6 E0/E1/E4, §12.2 wind risk) and the Data line in `CLAUDE.md` so they say *CPCB station wind* and *CREAMS*, not ERA5 and FIRMS. | `grep -n "ERA5\|FIRMS" PRD.md CLAUDE.md` only shows "replaced by" notes |

---

## 4. Run order

All scripts take `--config <yaml>`, log a run-id to `results/runs/`, and are wrapped by `python tasks.py <target>`.

```
# 0) Sanity, no data (laptop)
pytest -q                                   # all green, incl. the new F2–F4, G2, G4–G6 tests
python tasks.py smoke                       # plumbing only. NOT a result.

# E0) Data (laptop CPU)
#   manual: download D1 (+D1-meta) into data/raw/cpcb/   → fill MANIFEST.csv
python tasks.py data                        # ingest → QC → station wind field → domain → split
python scripts/04_characterize_data.py      # missingness, wind QC, entropy per window, domain map
python scripts/02_fetch_creams.py           # D3 → data/heldout/creams/ ; do NOT open the CSV until E4

# E1) Synthetic identifiability (CPU or small GPU). FINISH THIS FIRST: it is the publishable fallback.
python tasks.py e1

# E2) Real Delhi-NCR recovery (Kaggle GPU), one (season, pollutant) per session
python scripts/20_fit_real.py --config configs/experiment/e2_real.yaml --season postmonsoon --pollutant PM2.5   # pilot
python tasks.py e2                          # full fan-out, after the pilot passes §5.3

# E3) Ablations (Kaggle GPU)
python tasks.py e3

# E4) Held-out fire validation (CPU)
pytest tests/test_leakage.py -q             # must pass immediately before E4
python tasks.py e4                          # recovered NW-sector inflow vs CREAMS daily counts

# E5) Transfer (Kaggle GPU)
python tasks.py e5                          # needs D5 + configs/domain/kolkata.yaml (or chennai.yaml)

# Paper artifacts (laptop)
python tasks.py figures ; python tasks.py tables
```

Session rule (PRD §9): one (city, season, pollutant) fit per run, which must finish inside a 12 h Kaggle session. Track usage in `results/runs/quota.json`.

---

## 5. How to know a result is good

Every gate below is pass/fail. A step is **done** only when all of its gates pass and its `results/<exp>/REPORT.md` exists. That report holds: the numbers, the run-ids, the git commit, the config hash, the split hash, which gates passed, and anything that failed or looked odd. If a gate fails, the step is not done. Report the failure; never tune until it passes silently.

### 5.0 Red flags: stop and report immediately

- Any log line containing `fallback`, `dummy`, `constant(0.0`, `smoke`, or `synthetic` inside an E2–E5 real run.
- Perimeter R² > 0.95, or any metric that is suspiciously perfect. Suspect leakage first.
- Recovered `C_b` constant in space and time, or `S` with all its mass in one or two grid cells.
- NaN or Inf anywhere in the loss history, or a PDE-loss weight driven to ~0 by balancing (physics switched off).
- Results that change materially when only the seed changes (see 5.3).
- Any FIRMS, ERA5, Kaggle-dataset or OpenCity file under `data/` used by a run.

### 5.1 E0 — data

| Gate | Pass condition |
|---|---|
| Provenance | Every raw file is in `MANIFEST.csv` with a sha256. Rebuilding from raw reproduces `data/processed/` byte-identically (same split hash). |
| No imputation | The count of missing hourly PM values per station is identical in raw and processed. |
| Station inclusion | A station enters a (season, pollutant) fit only if it has ≥ 75% valid hours in that season. List included and excluded stations with their coverage. |
| Enough stations | ≥ 15 interior stations pass inclusion for PM2.5 in each season. ≥ 6 perimeter stations pass, covering **at least 3 of the 4 sides**, and the **NW side must be covered**. Otherwise E2's primary validation is weak: say so, and lean on leave-one-interior-station-out (LOISO). |
| Physical sanity | PM2.5 ≤ PM10 in ≥ 95% of co-located valid hours. Units are µg/m³ throughout. |
| Wind quality | Leave-one-station-out check of the station-wind interpolation: direction MAE < 45° and speed RMSE < 1.5 m/s, per season. Report the % of calm hours. The dominant post-monsoon direction must come out **north-westerly**, matching the documented climatology; if it doesn't, a sign convention is wrong. |
| Wind entropy | The directional-entropy distribution across fitting windows is reported (it drives RQ2). |
| Map | `domain_map.png` shows the interior and perimeter stations at the right places; check visually against a real map of NCR. |

### 5.2 E1 — synthetic identifiability (ground truth is known)

| Gate | Pass condition |
|---|---|
| Solver correctness | `test_synthetic_solver.py` green: mass conserved with zero source and zero loss; the error shrinks under grid refinement. |
| Replicates | ≥ 5 replicates per sweep cell. Report median and IQR, never a single run. |
| Best-case recovery | At 40 stations, high wind entropy and the lowest noise: relative L2 error on `C_b` ≤ 0.25 and on `S` ≤ 0.35. If this fails, the inverse problem is not solved even in the easy case. Stop and report; E2 must not start. |
| Monotone trends | Median `C_b` error falls as station count rises (7 → 15 → 25 → 40) and rises with noise. |
| Tomography hypothesis (RQ2) | Spearman ρ between window wind entropy and `C_b` error is negative with p < 0.05. If not, write it up as a negative result; it is still reportable. |
| **Null-boundary test** | Truth `C_b = 0` → recovered boundary-inflow share < 10%. This is the false-positive guard: the model must not invent transboundary pollution. |
| **Null-source test** | Truth `S = 0` → recovered interior-source share < 10%. |
| Wind-shuffle negative control | With the wind time order shuffled, recovery is clearly worse (> 1.5× error). If not, the model is not using the physics. |

### 5.3 E2 — real Delhi-NCR

| Gate | Pass condition |
|---|---|
| Real inputs only | `config_snapshot.json` + log show real `wind_field.nc`, the real split hash, the real n_obs and the list of fit stations, with no fallbacks. |
| Convergence | Total and PDE loss plateau. The normalized PDE residual is ≤ 1e-2 at the end. The L-BFGS stage ran. |
| Held-out skill | RMSE, MAE and R² against the **withheld perimeter stations** (hourly and daily-mean) are reported for our model **and every baseline on the same split**: zero-inflow, climatological inflow, MLP-PINN, uniform-pseudo-sequence PINNsformer, back-trajectory regression. |
| Beat baselines | Day-block bootstrap (1000 resamples) of the RMSE difference against zero-inflow and against climatological inflow: the 95% CI lies entirely below 0. If we don't beat these two, the boundary recovery is not adding information; report it honestly. |
| LOISO | Leave-one-interior-station-out error is reported, as the secondary check. |
| Seeds | 3 seeds per (season, pollutant). The spread of the recovered transboundary share must be ≤ 5 percentage points; if larger, the decomposition isn't stable enough to quote. |
| Physical plausibility | λ ordering PM1 < PM2.5 < PM10 holds when fitted jointly. `K` is positive and within the literature range for horizontal eddy diffusivity at these scales; record the reference used. Domain mass balance (inflow − outflow + source − deposition − storage change) closes within ±10%. |
| Wind sensitivity | Rerun the pilot instance with the wind rotated ±15° and scaled ±20%. Report how much the transboundary share moves. The share is only quotable if it is not dominated by this sensitivity. |
| Sanity vs published | Compare the season-mean "non-local" share with DSS (D4: roughly 58–66% non-Delhi). It does **not** have to match. Any divergence > 20 percentage points needs a written explanation in the REPORT before the number is shown to anyone. |

### 5.4 E3 — ablations

| Gate | Pass condition |
|---|---|
| Fair comparison | All pseudo-sequence variants have parameter counts within ±5% of each other, use the same split, same steps and 3 seeds each. |
| A win means a win | "Advection-aligned is better" may be claimed only where the mean improvement exceeds 2× the seed standard deviation, at the same L. The "needs a shorter L" claim must come from the accuracy-vs-L curve, with time and peak-memory curves next to it. |
| Physics-off ablation | Removing the PDE term must change the recovered `C_b` materially. If it doesn't, the physics isn't doing anything; report that. |

### 5.5 E4 — held-out fire validation (CREAMS)

| Gate | Pass condition |
|---|---|
| Leakage | `pytest tests/test_leakage.py -q` passes immediately before the run, and the output is pasted into the REPORT. |
| Parse correctness | `test_creams_parse.py` green (the 30-Oct-2020 numbers). Day count per season is reported. |
| Primary test | Spearman ρ between the daily recovered **NW-sector** boundary inflow and the daily **Punjab + Haryana** CREAMS event count, Oct–Nov, at lags 0–3 days: report ρ, p and n per lag, with no cherry-picking: pre-register lag 1 as primary in `e4_validation.yaml` **before** opening the CREAMS CSV. |
| Placebo | The same correlation with the **SE-sector** inflow should be clearly weaker. If it is just as strong, the signal is seasonal co-variation, not transport. |
| Meteorology control | Repeat the primary test on days with NW wind only. This is the physically meaningful subset. |

### 5.6 E5 — transfer

| Gate | Pass condition |
|---|---|
| Really another city | The log shows the second city's domain bounds, its station IDs and its data files. |
| Same gates | §5.1 and §5.3 gates are re-applied and the results reported, even if weaker. |

---

## 6. Config surface (what to edit vs never touch)

- **Edit freely:** `configs/pseudoseq/*.yaml`, `configs/model/*.yaml`, `configs/train/default.yaml`, `configs/experiment/*.yaml`, and the new `configs/data/station_wind.yaml` (QC thresholds, interpolation method, grid).
- **Never hand-edit:** units, CRS and normalization (`models/normalizer.py`); boundary direction (CCW from the SW corner, `data/domain.py`); split definitions (hash-pinned, `data/splits.py`); anything under `results/` or `paper/figures|tables/` (regenerated by scripts 90/91).
- **Registry:** names resolve via `utils/registry.py`, and `test_config.py` fails fast on typos.

---

## 7. Deliverables per step

| Step | Must exist before the step counts as done |
|---|---|
| E0 | `data/README.md` (what was downloaded, from where, when, any login/captcha seen) · `MANIFEST.csv` · `results/e0_data_quality/REPORT.md` with the §5.1 table filled |
| E1 | `results/e1_identifiability/` tidy table · identifiability surface figure · `REPORT.md` with the §5.2 gates |
| E2 | Per (season, pollutant): metrics + baselines + seeds + sensitivity · `REPORT.md` with the §5.3 gates |
| E3 | Accuracy / time / memory vs L per variant · `REPORT.md` |
| E4 | Pre-registered lag, then ρ/p/n table, placebo, NW-only · `REPORT.md` with the pasted leakage-test output |
| E5 | Same as E0 + E2 for the second city |
| Always | `paper/notes/claims_ledger.md`: every claim → results file → figure/table |

---

## 8. Review fixes applied (audit trail, as of 2026-09-12 — before the India-only data revision)

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

`hkv1` review (2026-10-03): ingestion, E0 build, characterization and real-data `20_fit_real` wiring are sound. Blocking issues are listed as F1–F9 in §3.1.

---

## 9. Known limitations (state these in the paper)

- **Wind from station anemometers**, not a reanalysis. Urban siting and calm periods add error. Mitigated by the wind QC, the leave-one-station-out check (§5.1) and the wind-sensitivity runs (§5.3). Coverage outside the station footprint relies on interpolation.
- **No boundary-layer height**, so `K` is a scalar. Winter inversions are where the 2-D vertically-integrated assumption is weakest. Report inversion-period performance separately.
- **CREAMS gives event counts, not fire radiative power**, so E4 correlates against counts. The proposal's FRP metric is replaced by counts; say so.
- **2022 rice-season bulletins** may be missing (§2, D3).
- **The CPCB network is uneven**: dense in Delhi, sparse in the outer NCR. E1 tells us what accuracy to expect at that density.
