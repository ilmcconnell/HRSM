"""Depth-weighted 0-100 cm rootzone soil moisture.

Ported by copy from ``wiscosoil/ingest.py`` (``layer_thicknesses``, ``compute_rootzone``; Iain
McConnell, wiscosoil) following the paper's §2.2.1 rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _col(depth: float) -> str:
    return f"sm_{int(depth) if float(depth).is_integer() else depth}"


def layer_thicknesses(depths_cm, rootzone_cm: float = 100) -> list[float]:
    """Midpoint-rule layer thicknesses summing to ``rootzone_cm``.

    Each sensor depth represents a layer bounded by the midpoints to its neighbours, with edges
    anchored at 0 (top) and ``rootzone_cm`` (bottom). For 5/10/20/50/100 cm over a 100 cm
    rootzone this is ``[7.5, 7.5, 20, 40, 25]``.
    """
    depths = sorted(depths_cm)
    if depths[-1] > rootzone_cm:
        raise ValueError(f"Deepest depth {depths[-1]} > rootzone {rootzone_cm}")
    boundaries = [0.0]
    for i in range(len(depths) - 1):
        boundaries.append((depths[i] + depths[i + 1]) / 2.0)
    boundaries.append(float(rootzone_cm))
    return [boundaries[i + 1] - boundaries[i] for i in range(len(depths))]


def compute_rootzone(sm: pd.DataFrame, depths_cm, weights=None) -> pd.Series:
    """Depth-weighted mean of the ``sm_<depth>`` columns.

    A row is valid only when the shallowest depth, the deepest depth and at least one middle depth
    are present. Missing middle values are linearly interpolated along depth (interior only,
    ``limit_area="inside"``), then averaged with ``weights`` / sum(weights). ``weights`` defaults to
    the midpoint-rule thicknesses.
    """
    depths = list(depths_cm)
    w = np.asarray(layer_thicknesses(depths) if weights is None else weights, dtype=float)
    if len(w) != len(depths):
        raise ValueError("weights and depths differ in length")
    sm_d = sm.reindex(columns=[_col(d) for d in depths])

    valid = (
        sm_d.iloc[:, 0].notna() & sm_d.iloc[:, -1].notna() & sm_d.iloc[:, 1:-1].notna().any(axis=1)
    )

    sm_t = sm_d.T.copy()
    sm_t.index = depths
    filled = sm_t.interpolate(method="index", axis=0, limit_area="inside").T

    values = (filled.to_numpy() * w).sum(axis=1) / w.sum()
    rz = pd.Series(values, index=sm.index, name="sm_rootzone")
    rz[~valid.to_numpy()] = np.nan
    return rz
