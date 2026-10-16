"""Wisconet observations through ``wiscopy``, cached per site-month (UTC).

Bounds are passed to ``wiscopy.data.bulk_measures`` in UTC and the result is already UTC-indexed
(``WisconetStation.fetch_data`` would treat naive bounds as station-local, so it is not used).

Units (checked against the live API, ALTN, June 2024):

* ``60min_soil_moisture_pct_avg@*`` is named ``pct`` but the API returns a 0-1 fraction
  (0.335-0.457), so it is **not** divided by 100. A month whose maximum exceeds 1 is treated as
  percent and divided by 100 (and one above 100 is rejected).
* ``60min_rain_in_tot`` is inches -> x 25.4 = mm.
* ``60min_air_temp_f_avg`` is Fahrenheit -> (F - 32) * 5/9 = deg C.
* ``60min_solar_radiation_w_avg`` is already W/m2. ``60min_solar_radiation_kwm2_tot`` x 1000 is
  only a documented fallback when the W field is absent.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd

from hrsmpipe.config import RunConfig
from hrsmpipe.paths import CachePaths, resolve_data_root
from hrsmpipe.sites import SiteSet
from hrsmpipe.timeaxis import hourly_axis

log = logging.getLogger(__name__)

SM_FIELDS = {
    "60min_soil_moisture_pct_avg@2in": "sm_5",
    "60min_soil_moisture_pct_avg@4in": "sm_10",
    "60min_soil_moisture_pct_avg@8in": "sm_20",
    "60min_soil_moisture_pct_avg@20in": "sm_50",
    "60min_soil_moisture_pct_avg@40in": "sm_100",
}
RAIN_FIELD = "60min_rain_in_tot"
AIR_TEMP_FIELD = "60min_air_temp_f_avg"
SOLAR_FIELD = "60min_solar_radiation_w_avg"
SOLAR_FALLBACK_FIELD = "60min_solar_radiation_kwm2_tot"
FIELDS = [*SM_FIELDS, RAIN_FIELD, AIR_TEMP_FIELD, SOLAR_FIELD, SOLAR_FALLBACK_FIELD]
OUT_COLUMNS = ["sm_5", "sm_10", "sm_20", "sm_50", "sm_100", "precip_mm", "air_temp_c", "solar_w"]

_MM_PER_INCH = 25.4

_throttle_lock = threading.Lock()
_last_call = [0.0]


def convert_fields(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide frame (columns = API ``standard_name``) to ``OUT_COLUMNS`` in SI-ish units."""
    out = pd.DataFrame(index=wide.index, columns=OUT_COLUMNS, dtype="float64")
    sm_src = [c for c in SM_FIELDS if c in wide.columns]
    if sm_src:
        peak = (
            np.nanmax(wide[sm_src].to_numpy(dtype=float))
            if wide[sm_src].notna().any().any()
            else 0.0
        )
        if peak > 100:
            raise ValueError(f"soil moisture maximum {peak} is outside 0-100")
        scale = 100.0 if peak > 1.0 else 1.0  # API normally already returns 0-1
        for src in sm_src:
            out[SM_FIELDS[src]] = wide[src].astype(float) / scale
    if RAIN_FIELD in wide:
        out["precip_mm"] = wide[RAIN_FIELD].astype(float) * _MM_PER_INCH
    if AIR_TEMP_FIELD in wide:
        out["air_temp_c"] = (wide[AIR_TEMP_FIELD].astype(float) - 32.0) * 5.0 / 9.0
    if SOLAR_FIELD in wide:
        out["solar_w"] = wide[SOLAR_FIELD].astype(float)
    elif SOLAR_FALLBACK_FIELD in wide:
        out["solar_w"] = wide[SOLAR_FALLBACK_FIELD].astype(float) * 1000.0
    return out


def _empty_frame() -> pd.DataFrame:
    """No observations: float64 columns (object columns would break the later concat/interpolation)."""
    return pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in OUT_COLUMNS},
        index=pd.DatetimeIndex([], tz="UTC", name="time"),
    )


def _throttle() -> None:
    """Keep to wiscopy's configured rate (calls per period); the sync client only retries on 429."""
    from wiscopy import data as wdata

    min_interval = float(wdata.RATE_LIMIT_PERIOD) / float(wdata.RATE_LIMIT_CALLS)
    with _throttle_lock:
        wait = _last_call[0] + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def fetch_range(station_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """One API request for ``[start, end)`` (UTC); a converted frame indexed by UTC time."""
    from wiscopy.data import bulk_measures
    from wiscopy.process import bulk_measures_to_df

    _throttle()
    # NOTE(hrsmpipe): the API's start bound is exclusive (a request from 00:00 returns 01:00 first),
    # so ask one hour early to keep the 00:00 observation.
    first = (start - pd.Timedelta(hours=1)).to_pydatetime()
    bms = bulk_measures(station_code, first, end.to_pydatetime(), fields=FIELDS)
    long = bulk_measures_to_df(bms)  # UTC-aware collection_time index
    if long is None:
        return _empty_frame()
    wide = long.pivot_table(
        index=long.index, columns="standard_name", values="value", aggfunc="last"
    )
    out = convert_fields(wide)
    out.index = pd.DatetimeIndex(out.index).tz_convert("UTC").floor("h")
    out.index.name = "time"
    return out[~out.index.duplicated(keep="first")].sort_index()


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize(UTC) if t.tzinfo is None else t.tz_convert(UTC)


def month_ranges(start, end) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    s, e = _utc(start), _utc(end)
    out, cur = [], s.normalize().replace(day=1)
    while cur < e:
        nxt = cur + pd.offsets.MonthBegin(1)
        lo, hi = max(cur, s), min(nxt, e)
        if lo < hi:
            out.append((cur.strftime("%Y-%m"), lo, hi))
        cur = nxt
    return out


def _read_chunk(path: Path, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame | None:
    import pyarrow.parquet as pq

    if not path.exists():
        return None
    try:
        meta = pq.read_schema(path).metadata or {}
        if _utc(meta[b"covered_from"].decode()) > lo or _utc(meta[b"covered_to"].decode()) < hi:
            return None
        df = pd.read_parquet(path)
    except Exception:
        return None
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df


def _write_chunk(path: Path, df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    meta = {
        **(table.schema.metadata or {}),
        b"covered_from": lo.isoformat().encode(),
        b"covered_to": hi.isoformat().encode(),
    }
    pq.write_table(table.replace_schema_metadata(meta), path)


def load_site(cfg: RunConfig, site_id: int, station_code: str) -> pd.DataFrame:
    """All months of one site, from the cache or the API; indexed by UTC time."""
    cache = CachePaths(resolve_data_root(cfg))
    start, end = hourly_axis(cfg.run.start, cfg.run.end)[[0, -1]]
    end = end + pd.Timedelta(hours=1)
    frames = []
    for label, lo, hi in month_ranges(start, end):
        path = cache.obs_chunk(site_id, label)
        df = _read_chunk(path, lo, hi)
        if df is None:
            try:
                df = fetch_range(station_code, lo, hi)
            except Exception as exc:  # HTTP errors, validation: do not cache a failed month
                log.warning("site %s (%s) %s: fetch failed (%s)", site_id, station_code, label, exc)
                continue
            _write_chunk(path, df, lo, hi)
        frames.append(df)
    if not frames:
        return _empty_frame()
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="first")].sort_index()


def load_all(cfg: RunConfig, sites: SiteSet) -> dict[int, pd.DataFrame]:
    from tqdm.auto import tqdm

    return {
        sid: load_site(cfg, sid, sites.code(sid)) for sid in tqdm(sites.ids, desc="Wisconet API")
    }
