# Gap matrix — living related-work table (PRD §4.8; becomes paper §2)

Owned by the Literature Agent. Every row must cite a primary source verified via paper-search tools; re-check at weeks 1, 8, 14 (pre-emption watch).

| # | Approach / work | What it does | What it leaves open (our gap) | Source status |
|---|---|---|---|---|
| 1 | PINNsformer (transformer + uniform pseudo-sequences) | Benchmark-PDE accuracy via forward-offset tokens | No real observational data; offsets ignore governing-equation physics | seeded in refs.bib; verify |
| 2 | AirPhyNet | Physics-guided AQ prediction | No inverse boundary formulation; boundary inflow not recovered | seeded; verify |
| 3 | SPIN | Spatial PINN for AQ | Same boundary gap as above | seeded; verify |
| 4 | CoNOAir (+ OmniAir) | Neural-operator AQ modeling | Same boundary gap as above | seeded; verify |
| 5 | PINN inverse-boundary (acoustics / haemodynamics) | Boundary recovery in other domains | No atmospheric transboundary analogue | to be sourced |
| 6 | DSS bulletins | Inventory-based seasonal attribution for Delhi | Needs emission inventory; our reference point, not baseline | seeded; verify numbers |
| 7 | WRF-Chem attribution studies | CTM nested-run attribution | Same inventory dependence; coarse-boundary inputs | seeded; verify numbers |
| 8 | Back-trajectory + regression (HYSPLIT-class) | Statistical upwind attribution | Non-mechanistic; our trainable baseline #5 | to be sourced |

Rule: no related-work claim enters `sections/related.tex` without a row here reaching `Source status = verified`.
