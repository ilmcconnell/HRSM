#!/usr/bin/env python
"""Convert the legacy ``wisconet_sites_soil_moisture_with_rootzone.npy`` into a table that
``obs.source = "table"`` can read.

The legacy array holds *Central local time* (naive) timestamps, which is why the package never reads
it directly. Row layout (object array): 0 station id (int), 1 station code, 2 local datetime,
3-7 soil moisture at 5/10/20/50/100 cm, 8 rootzone soil moisture, 9 precipitation (mm).

Usage::

    pixi run python scripts/convert_legacy_obs_npy.py legacy.npy out.parquet [--tz America/Chicago]

and in the run config::

    [obs]
    source = "table"
    path = "out.parquet"
    timezone = "UTC"          # the output is already UTC

Naive local times are localised with ``--tz``; ambiguous (DST fall-back) and non-existent (spring
forward) local times cannot be placed on the UTC axis and are dropped (their number is reported).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "site_id",
    "station_code",
    "local_time",
    "sm_5",
    "sm_10",
    "sm_20",
    "sm_50",
    "sm_100",
    "sm_rootzone",
    "precip_mm",
]


def convert(npy: Path, tz: str = "America/Chicago") -> tuple[pd.DataFrame, int]:
    """Return ``(tidy frame with UTC `time`, number of rows dropped for DST ambiguity)``."""
    raw = np.load(npy, allow_pickle=True)
    if raw.ndim != 2 or raw.shape[1] < len(COLUMNS):
        raise ValueError(f"unexpected legacy array shape {raw.shape}")
    df = pd.DataFrame(raw[:, : len(COLUMNS)], columns=COLUMNS)
    df["site_id"] = df["site_id"].astype(int)
    for col in COLUMNS[3:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    local = pd.DatetimeIndex(pd.to_datetime(df.pop("local_time")))
    utc = local.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    dropped = int(utc.isna().sum())
    df.insert(0, "time", utc)
    df["sm_surface"] = df["sm_5"]
    df = (
        df[df["time"].notna()].sort_values(["site_id", "time"]).drop_duplicates(["site_id", "time"])
    )
    return df.reset_index(drop=True), dropped


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("npy", type=Path, help="legacy wisconet_sites_soil_moisture_with_rootzone.npy")
    ap.add_argument("out", type=Path, help="output .parquet or .csv")
    ap.add_argument("--tz", default="America/Chicago", help="timezone of the legacy local times")
    args = ap.parse_args()
    df, dropped = convert(args.npy, args.tz)
    if args.out.suffix == ".csv":
        df.to_csv(args.out, index=False)
    else:
        df.to_parquet(args.out, index=False)
    print(
        f"wrote {len(df)} rows for {df['site_id'].nunique()} sites to {args.out} ({dropped} DST-ambiguous rows dropped)"
    )


if __name__ == "__main__":
    main()
