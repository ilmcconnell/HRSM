"""Combine the three sensors' inferences into one field per layer (notebook cell 114).

Steps: CDF-match the non-reference sensors onto the reference, derive the standard deviation from
the (unmatched) interval bounds, spread each value to the neighbouring hours, then average over
sensors on the hourly axis. The semantics (including which duplicate wins) follow the notebook.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from hrsmpipe.config import RunConfig
from hrsmpipe.features.assemble import LAYERS, SENSORS
from hrsmpipe.ml.infer import inference_path
from hrsmpipe.paths import RunPaths, resolve_data_root
from hrsmpipe.postproc.cdf import cdf_matching
from hrsmpipe.sites import SiteSet, load_sites
from hrsmpipe.timeaxis import hour_index, hourly_axis

log = logging.getLogger(__name__)

# Only used for grouping; the notebook coded sentinel1=1, sentinel2=2, hls=3.
SENSOR_CODE = {"sentinel1": 1, "sentinel2": 2, "hls": 3}
LAYER_CODE = {"surface": "sly", "rootzone": "rly"}


def reference_match(inf: pd.DataFrame, reference: str, column: str = "y_mean") -> pd.Series:
    """``column`` with the two non-reference sensors CDF-matched onto the reference, per site.

    Only sites that have rows in the reference sensor are matched; others are left unmatched and a
    warning is logged. Only ``y_mean`` is matched (bounds stay as predicted).
    """
    out = inf[column].copy()
    ref_sites = set(inf.loc[inf["sensor"] == reference, "site_id"])
    others = [s for s in SENSORS if s != reference]
    unmatched = set(inf["site_id"]) - ref_sites
    if unmatched:
        log.warning(
            "no %s inference for site(s) %s: left unmatched",
            reference,
            sorted(int(s) for s in unmatched),
        )
    for site in sorted(ref_sites):
        target = inf.loc[(inf["site_id"] == site) & (inf["sensor"] == reference), column].to_numpy()
        for sensor in others:
            mask = (inf["site_id"] == site) & (inf["sensor"] == sensor)
            if mask.any():
                out[mask] = cdf_matching(inf.loc[mask, column].to_numpy(), target)
    return out


def sd_from_bounds(inf: pd.DataFrame, divisor: float, floor: float) -> pd.Series:
    """``(y_high - y_low) / divisor`` with values <= 0 replaced by ``floor``."""
    sd = (inf["y_high"] - inf["y_low"]) / divisor
    return sd.mask(sd <= 0, floor)


def spread_rows(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """``concat(rows shifted -h .. -1, original rows, rows shifted +1 .. +h)`` in that order.

    A row shifted by ``off`` keeps its date and moves to hour ``hour + off``; ``_hour`` records
    that hour (it may be outside 0..23 and is filtered later, as in the notebook).
    """
    offsets = [*range(-hours, 0), 0, *range(1, hours + 1)]
    blocks = []
    for off in offsets:
        b = df.copy()
        b["_hour"] = b["time"].dt.hour + off
        b["time"] = b["time"] + pd.Timedelta(hours=off)
        blocks.append(b)
    return pd.concat(blocks, ignore_index=True)


def spread_and_grid(
    values: pd.DataFrame, hours: int, axis: pd.DatetimeIndex, site_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """``values`` has columns site_id, time, sensor, value. Returns ``(grid (time, site), miu (site,))``.

    ``miu`` is the mean over every spread row (before the hour filter), as ``calculate_sd_and_miu``.
    """
    spread = spread_rows(values, hours)
    miu_by_site = spread.groupby("site_id")["value"].mean()
    miu = np.array([miu_by_site.get(s, np.nan) for s in site_ids], dtype=float)

    spread = spread[(spread["_hour"] >= 0) & (spread["_hour"] <= 23)]
    spread = spread.drop_duplicates(["sensor", "site_id", "time"], keep="first")
    t_idx = hour_index(spread["time"], axis[0])
    col = {s: i for i, s in enumerate(site_ids)}
    s_idx = spread["site_id"].map(col)
    keep = ((t_idx >= 0) & (t_idx < len(axis)) & s_idx.notna()).to_numpy()
    grid3 = np.full((len(SENSORS), len(axis), len(site_ids)), np.nan)
    sensor_idx = spread["sensor"].map({s: i for i, s in enumerate(SENSORS)}).to_numpy()
    grid3[sensor_idx[keep], t_idx[keep], s_idx.to_numpy()[keep].astype(int)] = spread[
        "value"
    ].to_numpy()[keep]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices -> NaN, as np.nanmean did
        return np.nanmean(grid3, axis=0), miu


def postprocess_layer(
    inf: pd.DataFrame, cfg: RunConfig, axis: pd.DatetimeIndex, site_ids: list[int]
) -> dict[str, np.ndarray]:
    """``ssm``, ``sd`` grids (time, site) and ``miu`` (site,) for one layer's inference table."""
    if inf.empty:
        nan = np.full((len(axis), len(site_ids)), np.nan)
        return {"ssm": nan, "sd": nan.copy(), "miu": np.full(len(site_ids), np.nan)}
    inf = inf.copy()
    inf["time"] = pd.to_datetime(inf["time"], utc=True)
    value = pd.DataFrame(
        {
            "site_id": inf["site_id"],
            "time": inf["time"],
            "sensor": inf["sensor"],
            "value": reference_match(inf, cfg.postproc.reference_sensor),
        }
    )
    sd = value.assign(value=sd_from_bounds(inf, cfg.postproc.sd_divisor, cfg.postproc.sd_floor))
    ssm, miu = spread_and_grid(value, cfg.postproc.hour_spread, axis, site_ids)
    sd_grid, _ = spread_and_grid(sd, cfg.postproc.hour_spread, axis, site_ids)
    return {"ssm": ssm, "sd": sd_grid, "miu": miu}


def postprocess(cfg: RunConfig, sites: SiteSet | None = None) -> None:
    sites = sites or load_sites(cfg)
    paths = RunPaths(resolve_data_root(cfg), cfg.run.name)
    axis = hourly_axis(cfg.run.start, cfg.run.end)
    naive = axis.tz_localize(None)
    out_dir = paths.root / "postproc"
    out_dir.mkdir(parents=True, exist_ok=True)
    miu = pd.DataFrame({"site_id": sites.ids})
    for layer in LAYERS:
        code = LAYER_CODE[layer]
        res = postprocess_layer(pd.read_parquet(inference_path(paths, layer)), cfg, axis, sites.ids)
        for name in ("ssm", "sd"):
            ds = xr.Dataset(
                {f"{name}_{code}": (("time", "site"), res[name])},
                coords={"time": naive, "site": np.asarray(sites.ids, dtype="int64")},
                attrs={"reference_sensor": cfg.postproc.reference_sensor, "layer": layer},
            )
            ds.to_netcdf(out_dir / f"{name}_{code}.nc")
        miu[f"miu_{code}"] = res["miu"]
    miu.to_csv(out_dir / "miu.csv", index=False)


def run(cfg: RunConfig) -> None:
    postprocess(cfg)
