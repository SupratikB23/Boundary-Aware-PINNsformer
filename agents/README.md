# agents/ — roster and operating rules (PRD §8)

Documented so the workflow is reproducible and the paper's methods section can state honestly how the code was produced. Per-session diary entries live in `agents/logs/` (via the `session-log` skill).

| Agent | Mandate | Outputs |
|---|---|---|
| **Orchestrator** (main session) | Owns `PRD.md`, decomposes work, enforces acceptance criteria, decides when a milestone is met. | Milestone status, task assignments |
| **Data Engineer** | E0: ingestion, QC, alignment, domain/split construction, data-quality report. | `src/bapinnsformer/data/*`, `scripts/0*` |
| **Physics/Numerics** | PDE residual, boundary switching, collocation, FD synthetic solver, manufactured-solution tests. | `physics/*`, `data/synthetic.py`, `tests/test_pde_residual.py` |
| **Model Engineer** | PINNsformer, pseudo-sequence variants, `Cb_net`, `S_net`, `K`/`lam`, baselines. | `models/*` |
| **Training Engineer** | Losses, gradient-norm balancing, curriculum, Adam -> L-BFGS, checkpointing, stability triage. | `train/*` |
| **Experiment Runner** | Executes E1–E5 sweeps, manages the Kaggle session cap, collects tidy results. | `results/*` |
| **Evaluator** | Metrics, attribution accounting, external comparison, profiling, identifiability analysis. | `eval/*`, results tables |
| **Literature Agent** | Maintains `gap_matrix.md` and `refs.bib`; verifies related-work claims; watches for pre-emption. | `paper/notes/gap_matrix.md`, `paper/refs.bib` |
| **Paper Writer** | Drafts/revises manuscript sections from `results/` only; maintains `claims_ledger.md`. | `paper/sections/*` |
| **Reviewer / Red Team** | Adversarial pass: identifiability, leakage, overclaiming, figure honesty, reproducibility. | Review reports, blocking issues |
| **Figure Agent** | Publication figures and tables regenerated from results. | `paper/figures`, `paper/tables` |

## Operating rules (PRD §8.3)

1. PRD-first: file responsibilities (§4) are not renegotiated ad hoc.
2. One concern per file.
3. Tests before claims (§4.5).
4. Leakage is a blocking bug (FIRMS / perimeter).
5. Results are generated, never typed (scripts 90/91).
6. Every claim gets a `claims_ledger.md` row.
7. Honest reporting; negative E1 is publishable.
8. Session logs under `agents/logs/`.
