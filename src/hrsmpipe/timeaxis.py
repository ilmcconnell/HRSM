"""One hourly UTC axis ``[start, end)`` everywhere."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def hourly_axis(start, end) -> pd.DatetimeIndex:
    """Hourly UTC axis over ``[start, end)``; its length is the LGAR ``num_hours``."""
    return pd.date_range(_utc(start), _utc(end), freq="h", inclusive="left")


def hour_index(times, start) -> np.ndarray:
    """``floor((t - start) / 1h)`` as int64; negative for times before ``start`` (ERA5 lead)."""
    t = pd.DatetimeIndex(pd.to_datetime(times))
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    delta = (t - _utc(start)) / pd.Timedelta(hours=1)
    return np.floor(np.asarray(delta, dtype="float64")).astype("int64")


def year_doy_hour(times) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calendar year, day of year (``tm_yday``) and hour of day, from UTC times."""
    t = pd.DatetimeIndex(pd.to_datetime(times))
    return (
        np.asarray(t.year, dtype="int64"),
        np.asarray(t.dayofyear, dtype="int64"),
        np.asarray(t.hour, dtype="int64"),
    )
