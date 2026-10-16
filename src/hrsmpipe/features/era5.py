"""ERA5-Land lookback-weighted features (notebook cells 34 and 68)."""

from __future__ import annotations

import numpy as np

from hrsmpipe.gee.collections import ERA5_VARS

# lookback hours per ERA5 variable (short 24 h, mid 72 h, long 120 h)
LOOKBACK_HOURS = {
    "temperature_2m": 24,
    "dewpoint_temperature_2m": 24,
    "wind_10m": 24,
    "surface_pressure": 24,
    "surface_solar_radiation_downwards_hourly": 72,
    "surface_thermal_radiation_downwards_hourly": 72,
    "total_evaporation_hourly": 120,
    "total_precipitation_hourly": 120,
}
assert list(LOOKBACK_HOURS) == ERA5_VARS

# Feature order, table columns 16..23 of every sensor: d2m, t2m, ssr, str, e, wind, sp, tp.
FEATURE_VARS = [
    "dewpoint_temperature_2m",
    "temperature_2m",
    "surface_solar_radiation_downwards_hourly",
    "surface_thermal_radiation_downwards_hourly",
    "total_evaporation_hourly",
    "wind_10m",
    "surface_pressure",
    "total_precipitation_hourly",
]


def make_weights(n: int) -> np.ndarray:
    """``1 / [n ... 1]`` normalised to sum 1 (more weight on the most recent hour)."""
    w = 1.0 / np.arange(n, 0, -1)
    return w / w.sum()


def weighted_lookback(arr: np.ndarray, n: int, mode: str = "raw", lead: int = 0) -> np.ndarray:
    """Weighted mean of the last ``n`` hours (including the current one) per site.

    ``arr`` is ``(time, site)``. For row ``i >= n`` the result is
    ``np.sum(x[i-n+1 .. i] * w, axis=0)`` (NaN propagates), written exactly as in the notebook.
    Mode ``"raw"``: rows ``0..n-1`` keep the raw hourly value (the notebook looped over
    ``range(n, N)``). Mode ``"lead"``: ``arr`` already includes ``lead`` hours before the run start;
    rows are computed over the whole array and the first ``lead`` rows are dropped from the result.
    """
    if mode not in ("raw", "lead"):
        raise ValueError(f"unknown lookback mode {mode!r}")
    w = make_weights(n)
    out = arr.copy()
    for i in range(n, arr.shape[0]):
        out[i, :] = np.sum(arr[i - n + 1 : i + 1, :] * w[:, np.newaxis], axis=0)
    return out[lead:] if mode == "lead" else out
