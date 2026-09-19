# Claims ledger — every paper claim mapped to its supporting artifact (PRD §4.8, §8.3 rule 6)

Reviewed before submission; nothing unsupported ships. Paper Writer owns this file; Reviewer blocks on gaps.

| Claim | Location in paper | Supporting results file(s) | Figure / table | Status |
|---|---|---|---|---|
| E1 identifiability boundary vs stations/entropy/noise | results/ | `results/e1_identifiability/*.json` | Fig 4 (surface), Tab E1 | pending |
| Real-data C_b beats zero/climatological/MLP/uniform/trajectory baselines | results/ | `results/e2_real/*.json` | Fig 7, Tab E2 | pending |
| Advection-aligned PSFs reach target accuracy at shorter L / less time/memory | ablation/ | `results/e3_ablation/*.json` | Fig 8, Tab E3 | pending |
| Recovered NW Oct–Nov inflow correlates with withheld VIIRS FRP | validation/ | `results/e4_validation/*.json` | Fig 10 | pending |
| Transboundary share vs DSS/WRF-Chem (agree or explicably diverge) | validation/ | `results/e4_validation/*.json` | Fig 11, Tab external | pending |
| No fire data entered training (leakage test cited) | methods/ | `tests/test_leakage.py` pass + provenance | audit trail | pending |
| Physical consistency (mass balance; K/lam ranges; PM1<PM2.5<PM10) | results/ | `results/e2_real/*.json` | Fig 12, Tab params | pending |
