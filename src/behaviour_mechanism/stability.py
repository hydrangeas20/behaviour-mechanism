"""Stability rules.

* ``variance_stable_step`` -- the notebook's original rule (cross-seed std below a
  threshold for ``window`` consecutive checkpoints). Retained for provenance only: it
  returns step 0 for behaviour because every seed is at the chance floor at
  initialisation (the preprint, section 3.5, rejects it for this reason).
* ``sustained_ceiling_step`` -- the preprint's revised criterion: the first checkpoint
  ``t`` such that every seed's value is ``>= theta`` at ``t`` and the next
  ``window - 1`` recorded checkpoints.

Both operate on *recorded checkpoints*, so the returned step is quantised by the
checkpoint grid (50 steps up to 1,500, then 500). A step reported as 2,000 means
"somewhere in (1,500, 2,000]".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def variance_stable_step(
    steps: Sequence[int], stds: Sequence[float], threshold: float, window: int = 3
) -> int | None:
    for i in range(len(steps) - window + 1):
        if all(s < threshold for s in stds[i : i + window]):
            return int(steps[i])
    return None


def sustained_ceiling_step(wide: pd.DataFrame, theta: float, window: int = 3) -> int | None:
    """``wide``: index = step (sorted), one column per seed."""
    wide = wide.sort_index()
    ok = (wide >= theta).all(axis=1).to_numpy()
    steps = wide.index.to_numpy()
    for i in range(len(steps) - window + 1):
        if ok[i : i + window].all():
            return int(steps[i])
    return None


def threshold_sweep(
    long: pd.DataFrame,
    value_cols: dict[str, str],
    thetas: Sequence[float],
    window: int = 3,
) -> pd.DataFrame:
    """Apply ``sustained_ceiling_step`` to several measurements over a grid of thresholds.

    ``long`` has columns ``seed``, ``step`` and the value columns; ``value_cols`` maps an
    output name to a column. Returns one row per theta with one column per measurement.
    """
    rows = []
    for theta in thetas:
        row: dict[str, float | int | None] = {"theta": round(float(theta), 4)}
        for name, col in value_cols.items():
            wide = long.pivot(index="step", columns="seed", values=col)
            row[name] = sustained_ceiling_step(wide, theta, window)
        rows.append(row)
    return pd.DataFrame(rows)


def default_thetas() -> list[float]:
    return [round(x, 2) for x in np.arange(0.80, 1.0001, 0.01)]
