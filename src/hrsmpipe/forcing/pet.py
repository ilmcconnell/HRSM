"""Potential evapotranspiration from the Priestley-Taylor equation (notebook cell 46)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from hrsmpipe.config import RunConfig

log = logging.getLogger(__name__)

RHO_W = 1000

# pet_source codes (stored as int8 with attrs)
PET_MISSING, PET_MESONET, PET_ERA5, PET_INTERPOLATED = 0, 1, 2, 3
PET_SOURCE_ATTRS = {
    "flag_values": [0, 1, 2, 3],
    "flag_meanings": "missing mesonet era5 interpolated",
}


def e_s(T):
    return 611 * np.exp((17.27 * T) / (273.3 + T))


def delta(T):
    return (4098 * e_s(T)) / ((273.3 + T) ** 2)


def l_v(T):
    return 2500 - 2.36 * T


def priestley_taylor(ssr_wm2, t2m_c, alpha: float = 1.3, gamma: float = 66.8):
    """PET in mm/h from net radiation ``R_n`` [W/m2] and air temperature [deg C].

    ``E_r = R_n / (l_v * 1000 * rho_w)``, ``E_PT = E_r * alpha * delta / (delta + gamma)``,
    ``PET_mm_h = 3.6e6 * E_PT`` (``gamma`` is the constant 66.8 Pa/K, as in the notebook).
    """
    e_r = ssr_wm2 / (l_v(t2m_c) * 1000 * RHO_W)
    e_pt = e_r * alpha * delta(t2m_c) / (delta(t2m_c) + gamma)
    return 3.6e6 * e_pt


def fill_pet_gaps(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation (both directions) per site column; fully-NaN sites are skipped (warned)."""
    out = arr.copy()
    for site in range(arr.shape[1]):
        col = arr[:, site]
        if np.all(np.isnan(col)):
            log.warning("PET: site column %d is entirely missing; skipped", site)
            continue
        out[:, site] = (
            pd.Series(col).interpolate(method="linear", limit_direction="both").to_numpy()
        )
    return out


def build_pet(
    cfg: RunConfig,
    era5_ssr_j: np.ndarray,
    era5_t2m_k: np.ndarray,
    mesonet_solar_w: np.ndarray | None = None,
    mesonet_air_temp_c: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(pet_mm, pet_source)`` as ``(time, site)`` arrays on the run axis.

    ERA5: ``R_n = ssr_J / 3600``, ``T = t2m_K - 273.15``. Mesonet: ``R_n = solar_w``,
    ``T = air_temp_c`` with an ERA5 hour-wise fallback (``forcing.pet.fallback``). Values below
    ``zero_below`` become 0, then gaps are interpolated per site and flagged ``interpolated``.
    """
    c = cfg.forcing.pet
    era5 = priestley_taylor(era5_ssr_j / 3600, era5_t2m_k - 273.15, c.alpha, c.gamma)
    pet = np.full(era5.shape, np.nan)
    src = np.full(era5.shape, PET_MISSING, dtype=np.int8)
    if c.source == "mesonet":
        if mesonet_solar_w is None or mesonet_air_temp_c is None:
            log.warning(
                "forcing.pet.source = mesonet but the observations have no solar/air temperature"
            )
            mesonet = np.full(era5.shape, np.nan)
        else:
            mesonet = priestley_taylor(mesonet_solar_w, mesonet_air_temp_c, c.alpha, c.gamma)
        ok = np.isfinite(mesonet)
        pet[ok], src[ok] = mesonet[ok], PET_MESONET
    if c.source == "era5" or c.fallback == "era5":
        fill = np.isnan(pet) & np.isfinite(era5)
        pet[fill], src[fill] = era5[fill], PET_ERA5
    pet[pet < c.zero_below] = 0
    filled = fill_pet_gaps(pet)
    src[np.isnan(pet) & np.isfinite(filled)] = PET_INTERPOLATED
    return filled, src
