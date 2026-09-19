"""Visualization plane package.

PRD §4.3 (`viz/`): publication figures. Every public function takes
an explicit ``save_path`` and writes the figure file; nothing calls
``plt.show()`` (headless-safe for Kaggle / CI).
"""

from __future__ import annotations

__all__: list[str] = [
    "style",
    "maps",
    "fields",
    "boundary_plots",
    "ablation_plots",
    "attention_plots",
]
