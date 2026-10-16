"""Observations: ``load_observations(cfg, sites)`` returns one tidy UTC frame for all sites.

Columns: ``time`` (tz-aware UTC, floored to the hour), ``site_id``, ``sm_surface``,
``sm_rootzone``, ``precip_mm``; optional ``sm_5, sm_10, sm_20, sm_50, sm_100, air_temp_c, solar_w``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from hrsmpipe.config import RunConfig
from hrsmpipe.obs.rootzone import compute_rootzone
from hrsmpipe.sites import SiteSet
from hrsmpipe.timeaxis import hourly_axis

log = logging.getLogger(__name__)

DEPTH_COLUMNS = ["sm_5", "sm_10", "sm_20", "sm_50", "sm_100"]
OPTIONAL = [*DEPTH_COLUMNS, "air_temp_c", "solar_w"]
BASE = ["time", "site_id", "sm_surface", "sm_rootzone", "precip_mm"]


def _add_rootzone(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    depths = cfg.obs.rootzone_depths_cm
    depth_cols = [f"sm_{int(d)}" for d in depths]
    if not all(c in df.columns for c in depth_cols):
        return df
    df = df.copy()
    df[depth_cols] = df[depth_cols].astype(
        "float64"
    )  # all-empty chunks may arrive as object columns
    df["sm_surface"] = df[depth_cols[0]]
    df["sm_rootzone"] = compute_rootzone(df, depths, cfg.obs.rootzone_weights)
    return df


def _site_frame(cfg: RunConfig, sites: SiteSet, site_id: int) -> pd.DataFrame | None:
    from hrsmpipe.obs import table, wisconet_api, wisconet_csv

    src = cfg.obs.source
    if src == "wisconet_api":
        df = wisconet_api.load_site(cfg, site_id, sites.code(site_id))
    elif src == "wisconet_csv_dir":
        path = wisconet_csv.find_station_file(cfg.obs.path, sites.code(site_id))
        if path is None:
            log.warning(
                "site %s (%s): no archive file in %s", site_id, sites.code(site_id), cfg.obs.path
            )
            return None
        df = wisconet_csv.ingest_station(path)
    else:
        try:
            df = table.read_table(cfg.obs.path, site_id, cfg.obs.columns, cfg.obs.timezone)
        except FileNotFoundError:
            log.warning("site %s: observation table not found (%s)", site_id, cfg.obs.path)
            return None
    return _add_rootzone(df, cfg)


def load_observations(cfg: RunConfig, sites: SiteSet) -> pd.DataFrame:
    """Tidy observations on whole UTC hours inside ``[run.start, run.end)`` (exact-hour floor)."""
    axis = hourly_axis(cfg.run.start, cfg.run.end)
    parts = []
    for sid in sites.ids:
        df = _site_frame(cfg, sites, sid)
        if df is None or df.empty:
            continue
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index).floor("h")
        df = df[~df.index.duplicated(keep="first")]
        df = df[(df.index >= axis[0]) & (df.index <= axis[-1])]
        for col in ("sm_surface", "sm_rootzone", "precip_mm"):
            if col not in df:
                df[col] = np.nan
        df.insert(0, "site_id", sid)
        parts.append(df.rename_axis("time").reset_index())
    if not parts:
        return pd.DataFrame(columns=BASE)
    out = pd.concat(parts, ignore_index=True)
    cols = BASE + [c for c in OPTIONAL if c in out.columns]
    return out[cols]


def obs_grid(obs: pd.DataFrame | None, column: str, axis: pd.DatetimeIndex, site_ids) -> np.ndarray:
    """One observed column placed on the hourly UTC ``axis`` as a ``(time, site)`` array (NaN = none)."""
    out = np.full((len(axis), len(site_ids)), np.nan)
    if obs is None or obs.empty or column not in obs:
        return out
    t_idx = axis.get_indexer(
        pd.DatetimeIndex(pd.to_datetime(obs["time"], utc=True)).as_unit(axis.unit)
    )
    col = {int(s): j for j, s in enumerate(site_ids)}
    j_idx = obs["site_id"].map(col)
    keep = (t_idx >= 0) & j_idx.notna().to_numpy()
    out[t_idx[keep], j_idx.to_numpy()[keep].astype(int)] = obs[column].to_numpy(dtype=float)[keep]
    return out


def run(cfg: RunConfig) -> None:
    from hrsmpipe.paths import RunPaths, resolve_data_root
    from hrsmpipe.sites import load_sites

    obs = load_observations(cfg, load_sites(cfg))
    obs.to_parquet(RunPaths(resolve_data_root(cfg), cfg.run.name).obs, index=False)
