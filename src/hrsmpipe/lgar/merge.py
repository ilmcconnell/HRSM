"""Merge the LGAR / EnKF csv outputs, ML, observations and forcing into ``lgar/site_<id>.parquet``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from hrsmpipe.config import RunConfig
from hrsmpipe.forcing import forcing_paths, pet, precip
from hrsmpipe.obs import obs_grid
from hrsmpipe.paths import RunPaths, resolve_data_root
from hrsmpipe.postproc.cdf import cdf_matching
from hrsmpipe.sites import SiteSet
from hrsmpipe.timeaxis import hourly_axis

COLUMNS = [
    "forcast_sd_sly", "forcast_sd_rly", "analysis_sd_sly", "analysis_sd_rly",
    "forcast_sm_sly", "forcast_sm_rly", "analysis_sm_sly", "analysis_sm_rly",
    "ml_sd_sly", "ml_sd_rly", "ml_sm_sly", "ml_sm_rly",
    "sly_sm_ml_arr", "rly_sm_ml_arr", "observation_sly", "observation_rly",
    "forcast_sm_sly_cdf", "forcast_sm_rly_cdf", "analysis_sm_sly_cdf", "analysis_sm_rly_cdf",
    "original_sm_sly", "original_sm_rly", "original_sm_sly_cdf", "original_sm_rly_cdf",
    "analysis_sd_sly_cdf", "analysis_sd_rly_cdf",
    "precip_mm", "precip_source", "pet_mm", "pet_source",
]  # fmt: skip

_PRECIP_NAMES = dict(
    zip(
        precip.PRECIP_SOURCE_ATTRS["flag_values"],
        precip.PRECIP_SOURCE_ATTRS["flag_meanings"].split(),
        strict=True,
    )
)
_PET_NAMES = dict(
    zip(
        pet.PET_SOURCE_ATTRS["flag_values"],
        pet.PET_SOURCE_ATTRS["flag_meanings"].split(),
        strict=True,
    )
)


def cdf_to(series: np.ndarray, ml_target: np.ndarray) -> np.ndarray:
    """``cdf_matching(series, ml_target)`` on the finite values of ``series`` (NaN elsewhere)."""
    out = np.full(series.shape, np.nan)
    ok = np.isfinite(series)
    if ok.any() and len(ml_target):
        out[ok] = cdf_matching(series[ok], ml_target)
    return out


def sd_band_cdf(
    analysis: np.ndarray, analysis_sd: np.ndarray, analysis_cdf: np.ndarray
) -> np.ndarray:
    """Half width of the +/- 1 sigma band mapped through the monotone analysis -> ML mapping (cell 125)."""
    out = np.full(analysis.shape, np.nan)
    ok = np.isfinite(analysis) & np.isfinite(analysis_cdf) & np.isfinite(analysis_sd)
    if not ok.any():
        return out
    a, c, s = analysis[ok], analysis_cdf[ok], analysis_sd[ok]
    order = np.argsort(a)

    def to_ml(v):
        return np.interp(v, a[order], c[order])

    out[ok] = (to_ml(a + s) - to_ml(a - s)) / 2
    return out


def _on_axis(df: pd.DataFrame, index_col: str, n: int) -> pd.DataFrame:
    """Place rows on the hour index 0..n-1 (missing rows -> NaN)."""
    return df.set_index(index_col).reindex(range(n))


def merge_site(
    cfg: RunConfig, sites: SiteSet, site_id: int, obs: pd.DataFrame | None = None
) -> pd.DataFrame:
    paths = RunPaths(resolve_data_root(cfg), cfg.run.name)
    axis = hourly_axis(cfg.run.start, cfg.run.end)
    n = len(axis)
    j = sites.index_of(site_id)
    base = paths.root / "lgar"

    enkf = _on_axis(pd.read_csv(base / "enkf" / f"EnKF_LGAR_{site_id}.csv"), "time", n)
    orig = _on_axis(
        pd.read_csv(
            base / "original" / f"Original_LGAR_{site_id}.csv",
            header=None,
            names=["hour", "sm_sly", "sm_rly"],
        ),
        "hour",
        n,
    )
    out = pd.DataFrame(index=pd.Index(axis, name="time"))
    for col in COLUMNS[:12]:
        out[col] = enkf[col].to_numpy()

    ml = {}
    for layer in ("sly", "rly"):
        with xr.open_dataset(paths.root / "postproc" / f"ssm_{layer}.nc") as ds:
            ml[layer] = ds[f"ssm_{layer}"].to_numpy()[:, j]
        out[f"{layer}_sm_ml_arr"] = ml[layer]
    for layer, column in (("sly", "sm_surface"), ("rly", "sm_rootzone")):
        out[f"observation_{layer}"] = (
            obs_grid(obs, column, axis, [site_id])[:, 0] if obs is not None else np.nan
        )
    for layer in ("sly", "rly"):
        target = ml[layer][~np.isnan(ml[layer])]
        out[f"forcast_sm_{layer}_cdf"] = cdf_to(out[f"forcast_sm_{layer}"].to_numpy(), target)
        analysis = out[f"analysis_sm_{layer}"].to_numpy()
        analysis_cdf = cdf_to(analysis, target)
        out[f"analysis_sm_{layer}_cdf"] = analysis_cdf
        out[f"original_sm_{layer}"] = orig[f"sm_{layer}"].to_numpy()
        out[f"original_sm_{layer}_cdf"] = cdf_to(out[f"original_sm_{layer}"].to_numpy(), target)
        out[f"analysis_sd_{layer}_cdf"] = sd_band_cdf(
            analysis, out[f"analysis_sd_{layer}"].to_numpy(), analysis_cdf
        )

    fp = forcing_paths(paths)
    with xr.open_dataset(fp["prec"]) as ds:
        out["precip_mm"] = ds["precip_mm"].to_numpy()[:, j]
        out["precip_source"] = (
            pd.Series(ds["precip_source"].to_numpy()[:, j]).map(_PRECIP_NAMES).to_numpy()
        )
    with xr.open_dataset(fp["pet"]) as ds:
        out["pet_mm"] = ds["pet_mm"].to_numpy()[:, j]
        out["pet_source"] = pd.Series(ds["pet_source"].to_numpy()[:, j]).map(_PET_NAMES).to_numpy()

    out = out[COLUMNS]
    out.to_parquet(base / f"site_{site_id}.parquet")
    return out
