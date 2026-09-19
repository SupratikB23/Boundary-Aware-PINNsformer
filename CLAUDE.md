# CLAUDE.md — Boundary-Aware PINNsformer

## What this repository is

A research project (paper + code) for:

> **Boundary-Aware PINNsformer: Observation-Driven Recovery of Transboundary Pollutant Inflow using Advection-Aligned Pseudo-Sequences**

A physics-informed spatiotemporal **inverse-problem** framework applied to CPCB continuous ambient air quality data over the **Delhi-NCR airshed**.

## The research statement in one paragraph

Every city-scale air quality model must say what enters the domain across its lateral boundaries. Chemical transport models get this from a coarser nested run driven by an emission inventory; ML models ignore it entirely. This project **inverts the problem**: the lateral boundary inflow is not an input, it is the unknown to be recovered. Given sparse interior CPCB station observations and an ERA5 wind field, we solve for the time-varying, direction-resolved boundary inflow concentration `C_b(s,t)` jointly with the interior source field `S(x,y,t)`, subject to the advection–diffusion–deposition PDE as a physics constraint. The solver is a **PINNsformer** whose pseudo-sequence generator is replaced by one that samples pseudo-tokens along **backward wind characteristics** instead of uniform forward time offsets.

## Governing equation

```
∂C/∂t + ∇·(uC) = ∇·(K∇C) + S(x,y,t) − λC
```

| Symbol | Meaning | Status |
|---|---|---|
| `C(x,y,t)` | Concentration field | Unknown — PINNsformer output, constrained at stations |
| `u = (u,v)` | Horizontal wind | **Known** — ERA5, cross-checked vs station met |
| `C_b(s,t)` | Boundary inflow | **PRIMARY UNKNOWN** — separate small net over (arc-length, time) |
| `S(x,y,t)` | Interior emissions | Unknown — net with sparsity prior; must be recovered jointly |
| `K` | Eddy diffusivity | Unknown scalar / small net of BLH + stability |
| `λ` | Deposition + removal | Unknown scalar per pollutant; must order PM1 < PM2.5 < PM10 |

Boundary is split per-timestep by wind sign: **Dirichlet** (learned `C_b`) on inflow segments, **zero-gradient** on outflow segments.

## Four research questions

1. **RQ1** — Can a physics-informed transformer jointly recover `C`, `S` and `C_b` from sparse interior stations + known wind?
2. **RQ2** — When is the `C_b` vs `S` decomposition **identifiable**? (station count/config, wind directional entropy, noise)
3. **RQ3** — Do advection-aligned pseudo-sequences beat uniform-offset ones in accuracy *and* in sequence length needed?
4. **RQ4** — Does recovered inflow reproduce the Oct–Nov north-westerly crop-burning signal **with no fire data supplied**?

## The gap being filled

- No work formulates atmospheric transboundary inflow as a physics-informed **inverse boundary problem**.
- No work applies **PINNsformer to real observational data** (every published result is on benchmark PDEs).
- No work gives an **emission-inventory-free** estimate of Delhi's transboundary share.
- No work adapts the pseudo-sequence to the **physics of the governing equation**.

## Experiments (short form)

- **E0** Data pipeline, missingness characterization, ERA5-vs-station wind agreement, interior/perimeter split.
- **E1** Synthetic identifiability sweep (FD solver ground truth; stations ∈ {7,15,25,40}, wind entropy, noise). *Publishable alone.*
- **E2** Real Delhi-NCR recovery; fit on interior Delhi stations, validate against **withheld NCR perimeter ring**.
- **E3** Ablations: 4 pseudo-sequence variants, sequence length vs accuracy/time/memory, Wavelet activation, physics term.
- **E4** Recovered NW inflow vs **withheld VIIRS fire counts / FRP**; compare to DSS & WRF-Chem published shares.
- **E5** Transfer to a second airshed (Kolkata or Chennai).

## Data

CPCB CCR portal (+ OpenCity mirror) hourly pollutants & met · ERA5 (wind, BLH, T, RH) · NASA FIRMS VIIRS/MODIS **held out entirely** · published DSS / WRF-Chem shares as external comparison only.
Study period **Oct 2019 – Feb 2024**, focus on post-monsoon (Oct–Nov) and winter (Dec–Feb).

## Constraints that shape every design decision

- Compute: **RTX 3050** (dev) + **Kaggle dual T4**, 30 h/week quota, 12 h session cap. Budget ≈ 150 GPU-hours total.
- Therefore: small models (<0.5 M params), **independent per-(city, season, pollutant) fits** rather than one big run; no checkpoint-resume engineering needed.
- Timeline: 16 weeks to manuscript.
- Masked loss over gaps — **never impute** CPCB missingness.
- FIRMS fire data must never enter training. Guard this in code.

## Fallback position (already accepted)

If the full inversion does not beat baselines on real data, **E1 alone is the paper**: the identifiability boundary is a contribution computed on known ground truth. A weak E2 does not sink the project.

## Working conventions for Claude in this repo

- **`PRD.md` is the source of truth** for architecture, file layout and task decomposition. Read it before writing any code or paper section.
- Prefer reproducibility over cleverness: every experiment must be re-runnable from a config file + seed.
- Physics symbols in code use the names in the table above (`C`, `C_b`, `S`, `K`, `lam`, `u`, `v`).
- Target venues: *Environmental Modelling & Software* (primary), *Atmospheric Environment*, *GMD*, SIGSPATIAL / AAAI AISI, *Sci. Reports / IEEE Access* (fallback).
