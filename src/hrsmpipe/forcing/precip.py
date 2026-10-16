"""Precipitation forcing: mesonet rain gauges with ERA5 filling gaps, with per-hour provenance."""

from __future__ import annotations

import numpy as np

from hrsmpipe.config import RunConfig

# precip_source codes (stored as int8 with attrs)
PRECIP_MISSING, PRECIP_MESONET, PRECIP_ERA5 = 0, 1, 2
PRECIP_SOURCE_ATTRS = {"flag_values": [0, 1, 2], "flag_meanings": "missing mesonet era5"}


def shift_hours(arr: np.ndarray, hours: int) -> np.ndarray:
    """Delay a ``(time, site)`` array by ``hours`` (``out[t] = arr[t - hours]``); gaps become NaN."""
    if hours == 0:
        return arr
    out = np.full(arr.shape, np.nan)
    if hours > 0:
        out[hours:] = arr[:-hours]
    else:
        out[:hours] = arr[-hours:]
    return out


def era5_precip_mm(tp_m: np.ndarray, zero_below_mm: float) -> np.ndarray:
    """ERA5 ``total_precipitation_hourly`` (m) -> mm, values below ``zero_below_mm`` set to 0 (NaN kept)."""
    mm = tp_m * 1000  # hourly precipitation, m to mm
    mm[mm < zero_below_mm] = 0
    return mm


def build_precip(
    cfg: RunConfig, era5_tp_m: np.ndarray, mesonet_mm: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """``(precip_mm, precip_source)`` as ``(time, site)`` arrays on the run axis.

    ``source = "mesonet"``: mesonet where finite, else ERA5 where finite (if ``fallback = "era5"``),
    else 0 flagged ``missing``. ``source = "era5"``: ERA5, NaN -> 0 flagged ``missing``.
    ``interval_shift_hours`` is applied to both series together (default 0).
    """
    c = cfg.forcing.precip
    shift = cfg.forcing.interval_shift_hours
    era5 = shift_hours(era5_precip_mm(era5_tp_m.copy(), c.era5_zero_below_mm), shift)
    prec = np.full(era5.shape, np.nan)
    src = np.full(era5.shape, PRECIP_MISSING, dtype=np.int8)
    if c.source == "mesonet" and mesonet_mm is not None:
        mes = shift_hours(mesonet_mm, shift)
        ok = np.isfinite(mes)
        prec[ok], src[ok] = mes[ok], PRECIP_MESONET
    if c.source == "era5" or c.fallback == "era5":
        fill = np.isnan(prec) & np.isfinite(era5)
        prec[fill], src[fill] = era5[fill], PRECIP_ERA5
    prec[np.isnan(prec)] = 0.0  # missing hours are dry, flagged in precip_source
    return prec, src
