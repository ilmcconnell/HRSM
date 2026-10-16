"""Reader for a Wisconet CSV archive (``<CODE>_all_hourly_*.csv`` with ``gmt_timestamp``).

Ported by copy from ``wiscosoil/ingest.py`` (``ingest_station`` / ``pivot_sm``; Iain McConnell,
wiscosoil), without its QC filtering: the notebook applied none.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SM_DEPTHS_CM = (5, 10, 20, 50, 100)
_MET = {
    "rain_mm_tot": "precip_mm",
    "air_temp_c_avg": "air_temp_c",
    "solar_radiation_w_avg": "solar_w",
}


def pivot_sm(hourly: pd.DataFrame) -> pd.DataFrame:
    rename = {f"soil_moisture_{d}cm_m3m-3_avg": f"sm_{d}" for d in SM_DEPTHS_CM}
    present = {k: v for k, v in rename.items() if k in hourly.columns}
    out = hourly[list(present)].rename(columns=present)
    return out.reindex(columns=[f"sm_{d}" for d in SM_DEPTHS_CM])


def find_station_file(directory: Path, station_code: str) -> Path | None:
    for path in sorted(Path(directory).glob("*_all_hourly_*.csv")):
        if path.name.split("_")[0] == station_code:
            return path
    return None


def ingest_station(path: Path) -> pd.DataFrame:
    """Hourly frame indexed by UTC time with ``sm_<depth>``, ``precip_mm``, ``air_temp_c``, ``solar_w``."""
    raw = pd.read_csv(path, na_values=["", "NA", "NaN"], low_memory=False)
    idx = pd.DatetimeIndex(pd.to_datetime(raw["gmt_timestamp"], unit="s", utc=True), name="time")
    sm = pivot_sm(raw)
    sm.index = idx
    met = raw[[c for c in _MET if c in raw.columns]].rename(columns=_MET)
    met.index = idx
    out = pd.concat([sm, met], axis=1).sort_index()
    return out[~out.index.duplicated(keep="first")]
