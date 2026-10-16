"""Generic observation table reader (CSV or parquet) with a column map and timezone."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hrsmpipe.config import ObsColumns

_OPTIONAL = ("sm_surface", "sm_rootzone", "precip_mm", "air_temp_c", "solar_w")


def localize_to_utc(times: pd.Series, timezone: str) -> pd.DatetimeIndex:
    """Naive timestamps are read in ``timezone``; tz-aware ones are converted to UTC."""
    t = pd.DatetimeIndex(pd.to_datetime(times))
    if t.tz is None:
        t = t.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    return t.tz_convert("UTC")


def read_table(path: str, site_id: int, columns: ObsColumns, timezone: str = "UTC") -> pd.DataFrame:
    """Read one site's table. ``{site_id}`` in ``path`` selects one file per site; otherwise the
    file must hold a ``site_id`` column. Returns rows indexed by UTC time with the standard names."""
    p = Path(path.format(site_id=site_id)).expanduser()
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if "{site_id}" not in path:
        if not columns.site_id or columns.site_id not in df.columns:
            raise ValueError(f"{p}: no '{columns.site_id}' column and path has no {{site_id}}")
        df = df[df[columns.site_id].astype(int) == int(site_id)]
    out = pd.DataFrame(index=localize_to_utc(df[columns.time], timezone))
    for std in _OPTIONAL:
        src = getattr(columns, std)
        if src:
            if src not in df.columns:
                raise ValueError(f"{p}: column {src!r} (obs.columns.{std}) not found")
            out[std] = pd.to_numeric(df[src], errors="coerce").to_numpy()
    out.index.name = "time"
    out = out[out.index.notna()].sort_index()
    return out[~out.index.duplicated(keep="first")]
