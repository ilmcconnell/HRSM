"""Assemble the per-sensor ML feature tables and write ``features/<sensor>_<layer>.parquet``."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from hrsmpipe.config import RunConfig
from hrsmpipe.features import era5 as era5_features
from hrsmpipe.features import satellite
from hrsmpipe.gee.static import load_static
from hrsmpipe.paths import RunPaths, resolve_data_root
from hrsmpipe.sites import SiteSet, load_sites
from hrsmpipe.timeaxis import hourly_axis, year_doy_hour

log = logging.getLogger(__name__)

SENSORS = ("sentinel1", "sentinel2", "hls")
LAYERS = ("surface", "rootzone")
SENSOR_SOURCE = {"sentinel1": "s1", "sentinel2": "s2", "hls": "hls"}

_COMMON = [
    "site_id", "year", "doy", "hour", "latitude", "longitude", "elevation", "slope", "sinAspect",
    "cosAspect", "TWI", "bd_0_100", "sand_0_100", "clay_0_100", "ksat_0_100", "smap_constant",
    "d2m_lb", "t2m_lb", "ssr_lb", "str_lb", "e_lb", "wind_lb", "sp_lb", "tp_lb",
]  # fmt: skip
COLUMNS_S1 = [*_COMMON, "VV", "VH", "angle", "VV/VH", "(VV^2+VH^2)/sqrt2", "(VV+VH)/(VV-VH)"]
COLUMNS_S2 = [
    *_COMMON, "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
    "NDVI", "NDWI", "greenness", "brightness", "wetness",
]  # fmt: skip
COLUMNS_HLS = [
    *_COMMON, "B2", "B3", "B4", "B5", "B6", "B7", "B10", "B11", "SZA", "SAA", "VZA", "VAA",
    "NDVI", "NDWI", "greenness", "brightness", "wetness",
]  # fmt: skip
COLUMNS = {"sentinel1": COLUMNS_S1, "sentinel2": COLUMNS_S2, "hls": COLUMNS_HLS}
assert (len(COLUMNS_S1), len(COLUMNS_S2), len(COLUMNS_HLS)) == (30, 39, 41)


def feature_columns(sensor: str) -> list[str]:
    return [f"f{i:02d}" for i in range(len(COLUMNS[sensor]))]


def features_path(paths: RunPaths, sensor: str, layer: str):
    return paths.root / "features" / f"{sensor}_{layer}.parquet"


def era5_lookbacks(cfg: RunConfig, era5: xr.Dataset) -> list[np.ndarray]:
    """The 8 lookback features (d2m, t2m, ssr, str, e, wind, sp, tp) on ``[start, end)``, (time, site)."""
    mode = cfg.features.era5_lookback_warmup
    lead = cfg.gee.era5_lead_hours
    out = []
    for var in era5_features.FEATURE_VARS:
        arr = era5[var].to_numpy()
        if mode == "raw":
            arr = arr[lead:]  # the notebook's ERA5 array started at run.start
            out.append(
                era5_features.weighted_lookback(arr, era5_features.LOOKBACK_HOURS[var], "raw")
            )
        else:
            out.append(
                era5_features.weighted_lookback(
                    arr, era5_features.LOOKBACK_HOURS[var], "lead", lead
                )
            )
    return out


def build_table(
    cfg: RunConfig,
    sensor: str,
    sites: SiteSet,
    static: pd.DataFrame,
    lookbacks: list[np.ndarray],
    rs: xr.Dataset,
) -> pd.DataFrame:
    """Feature table for one sensor: rows with no NaN, ordered time-major then site (as the notebook)."""
    axis = hourly_axis(cfg.run.start, cfg.run.end)
    n_t, n_s = len(axis), len(sites)
    year, doy, hour = year_doy_hour(axis)

    def tile_time(v):  # (time,) -> (time, site)
        return np.repeat(np.asarray(v, dtype=float)[:, np.newaxis], n_s, axis=1)

    def tile_site(v):  # (site,) -> (time, site)
        return np.tile(np.asarray(v, dtype=float), (n_t, 1))

    st = static.loc[sites.ids]
    cols = [
        tile_site(np.asarray(sites.ids)),
        tile_time(year),
        tile_time(doy),
        tile_time(hour),
        tile_site(st["latitude"]),
        tile_site(st["longitude"]),
        tile_site(st["elevation"]),
        tile_site(st["slope"]),
        tile_site(st["sin_asp"]),
        tile_site(st["cos_asp"]),
        tile_site(st["twi"]),
        tile_site(st["bd_0_100"]),
        tile_site(st["sand_0_100"]),
        tile_site(st["clay_0_100"]),
        tile_site(st["ksat_0_100"]),
        np.full((n_t, n_s), cfg.features.smap_constant),
        *lookbacks,
    ]
    bands = {b: rs[b].to_numpy() for b in rs.data_vars}
    if sensor == "sentinel1":
        cols += satellite.s1_features(
            bands["VV"], bands["VH"], bands["angle"], cfg.features.s1_invalid_above
        )
    elif sensor == "sentinel2":
        cols += satellite.s2_features(bands)
    else:
        cols += satellite.hls_features(bands, cfg.features.hls_angle_constant)
    X = np.stack(cols).reshape(len(cols), n_t * n_s)
    mask = ~np.any(np.isnan(X), axis=0)
    df = pd.DataFrame(X[:, mask].T, columns=feature_columns(sensor))
    df["site_id"] = df["f00"].astype("int64")
    df["time"] = axis[np.nonzero(mask)[0] // n_s]
    return df


def build_features(cfg: RunConfig, sites: SiteSet | None = None) -> dict[tuple[str, str], int]:
    """Write ``features/<sensor>_<layer>.parquet``; returns the row count per (sensor, layer)."""
    sites = sites or load_sites(cfg)
    paths = RunPaths(resolve_data_root(cfg), cfg.run.name)
    static = load_static(cfg, sites)
    with xr.open_dataset(paths.rs("era5")) as era5:
        lookbacks = era5_lookbacks(cfg, era5.load())
    counts = {}
    for sensor in SENSORS:
        with xr.open_dataset(paths.rs(SENSOR_SOURCE[sensor])) as rs:
            table = build_table(cfg, sensor, sites, static, lookbacks, rs.load())
        # Surface and rootzone use identical feature tables (SMAP is a constant in both).
        for layer in LAYERS:
            p = features_path(paths, sensor, layer)
            p.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(p, index=False)
            counts[(sensor, layer)] = len(table)
        log.info("features %s: %d rows with no NaN", sensor, len(table))
    return counts


def run(cfg: RunConfig) -> None:
    build_features(cfg)
