"""LGAR forcing: precipitation, PET and soil classes (``forcing/prec.nc``, ``pet.nc``, ``soil_classes.csv``)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from hrsmpipe.config import RunConfig
from hrsmpipe.forcing import pet, precip, soils
from hrsmpipe.gee.static import load_static
from hrsmpipe.obs import obs_grid
from hrsmpipe.paths import RunPaths, resolve_data_root
from hrsmpipe.sites import load_sites
from hrsmpipe.timeaxis import hourly_axis

log = logging.getLogger(__name__)


def forcing_paths(paths: RunPaths) -> dict[str, object]:
    d = paths.root / "forcing"
    return {"prec": d / "prec.nc", "pet": d / "pet.nc", "soils": d / "soil_classes.csv"}


def build_forcing(cfg: RunConfig, sites=None) -> None:
    sites = sites or load_sites(cfg)
    paths = RunPaths(resolve_data_root(cfg), cfg.run.name)
    out = forcing_paths(paths)
    out["prec"].parent.mkdir(parents=True, exist_ok=True)
    axis = hourly_axis(cfg.run.start, cfg.run.end)
    naive = axis.tz_localize(None)
    lead = cfg.gee.era5_lead_hours
    with xr.open_dataset(paths.rs("era5")) as ds:
        era5 = ds.isel(time=slice(lead, None)).load()
    if era5.sizes["time"] != len(axis):
        raise ValueError(
            f"rs/era5.nc has {era5.sizes['time'] - 0} rows after the lead; expected {len(axis)}"
        )
    obs = pd.read_parquet(paths.obs) if paths.obs.exists() else None
    mesonet_needed = cfg.forcing.precip.source == "mesonet" or cfg.forcing.pet.source == "mesonet"
    if mesonet_needed and obs is None:
        log.warning("mesonet forcing requested but obs.parquet is missing: ERA5 only")

    coords = {"time": naive, "site": np.asarray(sites.ids, dtype="int64")}

    mes_p = obs_grid(obs, "precip_mm", axis, sites.ids)
    prec, prec_src = precip.build_precip(cfg, era5["total_precipitation_hourly"].to_numpy(), mes_p)
    xr.Dataset(
        {
            "precip_mm": (("time", "site"), prec),
            "precip_source": (("time", "site"), prec_src, precip.PRECIP_SOURCE_ATTRS),
        },
        coords=coords,
    ).to_netcdf(out["prec"])

    solar = obs_grid(obs, "solar_w", axis, sites.ids) if obs is not None else None
    airt = obs_grid(obs, "air_temp_c", axis, sites.ids) if obs is not None else None
    pet_mm, pet_src = pet.build_pet(
        cfg,
        era5["surface_solar_radiation_downwards_hourly"].to_numpy(),
        era5["temperature_2m"].to_numpy(),
        solar,
        airt,
    )
    xr.Dataset(
        {
            "pet_mm": (("time", "site"), pet_mm),
            "pet_source": (("time", "site"), pet_src, pet.PET_SOURCE_ATTRS),
        },
        coords=coords,
    ).to_netcdf(out["pet"])

    table = soils.soil_classes(load_static(cfg, sites))
    table.to_csv(out["soils"], index=False)


def run(cfg: RunConfig) -> None:
    build_forcing(cfg)
