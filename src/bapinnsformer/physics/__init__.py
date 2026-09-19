"""Physics plane package.

PRD §4.3 (`physics/`): autodiff PDE residuals over network outputs,
wind-switched boundary conditions (Dirichlet inflow / Neumann outflow),
collocation sampling with temporal curriculum, and domain mass-balance
diagnostics. CPU-runnable; GPU via torch device of the inputs.
"""

from __future__ import annotations

__all__: list[str] = ["operators", "boundary", "collocation", "mass_balance"]
