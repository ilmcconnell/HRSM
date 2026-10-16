"""Hourly GEE time series (ERA5-Land, Sentinel-1/2, HLSL30), cached per (source, site, month).

A request is made per ``(source, month, all sites missing from that month's cache)`` and split into
``gee.window_hours`` windows aligned to the start of the month job. Every sample carries
``system:time_start``; values per ``(hour, site)`` are the mean of all samples (NaN where none).
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from hrsmpipe import __version__
from hrsmpipe.config import RunConfig
from hrsmpipe.gee import auth, collections
from hrsmpipe.paths import CachePaths, RunPaths, resolve_data_root
from hrsmpipe.sites import SiteSet, load_sites
from hrsmpipe.timeaxis import hourly_axis

log = logging.getLogger(__name__)

SOURCES = ("era5", "s1", "s2", "hls")
_ASSET_IDS = {
    "era5": collections.ERA5_ID,
    "s1": collections.S1_ID,
    "s2": collections.S2_ID,
    "hls": collections.HLS_ID,
}
_ERA5_SLEEP_S = 2.0  # the notebook paused 2 s per ERA5 request to stay inside rate limits


def _naive(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def source_span(cfg: RunConfig, source: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """ERA5 covers ``[start - era5_lead_hours, end)``; satellites ``[start, end)`` (naive UTC)."""
    start, end = pd.Timestamp(cfg.run.start), pd.Timestamp(cfg.run.end)
    if source == "era5":
        start = start - pd.Timedelta(hours=cfg.gee.era5_lead_hours)
    return start, end


@dataclass(frozen=True)
class MonthJob:
    label: str  # YYYY-MM
    start: pd.Timestamp  # clipped to the span, naive UTC
    end: pd.Timestamp


def month_jobs(span_start: pd.Timestamp, span_end: pd.Timestamp) -> list[MonthJob]:
    jobs = []
    cur = span_start.normalize().replace(day=1)
    while cur < span_end:
        nxt = cur + pd.offsets.MonthBegin(1)
        s, e = max(cur, span_start), min(nxt, span_end)
        if s < e:
            jobs.append(MonthJob(cur.strftime("%Y-%m"), s, e))
        cur = nxt
    return jobs


def windows(job: MonthJob, hours: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Consecutive ``hours``-long windows aligned to the job start; the last is clipped."""
    out, t = [], job.start
    step = pd.Timedelta(hours=hours)
    while t < job.end:
        out.append((t, min(t + step, job.end)))
        t += step
    return out


def _chunk_attrs(cfg: RunConfig, source: str, job: MonthJob) -> dict:
    return {
        "time_min": job.start.isoformat(),
        "time_max": job.end.isoformat(),
        "scale_m": float(cfg.gee.sample_scale_m),
        "window_hours": int(cfg.gee.window_hours),
        "asset_ids": _ASSET_IDS[source],
        "hrsmpipe_version": __version__,
    }


def chunk_is_valid(path, cfg: RunConfig, job: MonthJob) -> bool:
    """A chunk is reused only if it covers the requested month range at the same scale."""
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path) as ds:
            a = ds.attrs
            return (
                pd.Timestamp(a["time_min"]) <= job.start
                and pd.Timestamp(a["time_max"]) >= job.end
                and float(a["scale_m"]) == float(cfg.gee.sample_scale_m)
            )
    except Exception:
        return False


def _sample_window(source: str, fc, roi, t0, t1, scale) -> pd.DataFrame:
    """Samples in ``[t0, t1)`` -> hourly mean per (hour, id_num), columns = the source's bands."""
    bands = collections.SOURCE_BANDS[source]
    parts = []
    for make in collections.COLLECTIONS[source]:
        coll = make(roi, t0, t1)

        def sample_with_time(image):
            samples = image.sampleRegions(collection=fc, scale=scale)
            return samples.map(
                lambda feat: feat.set("system:time_start", image.get("system:time_start"))
            )

        feats = collections.get_info(coll.map(sample_with_time).flatten())["features"]
        if source == "era5":
            time.sleep(_ERA5_SLEEP_S)
        df = pd.DataFrame([f["properties"] for f in feats])
        if "id_num" in df.columns:
            df["hour"] = pd.to_datetime(df["system:time_start"], unit="ms").dt.floor("h")
            cols = [b for b in bands if b in df.columns]
            parts.append(df.groupby(["hour", "id_num"])[cols].mean())
    if not parts:
        return pd.DataFrame(
            columns=bands, index=pd.MultiIndex.from_arrays([[], []], names=["hour", "id_num"])
        )
    out = pd.concat(parts, axis=1)
    return out.reindex(columns=bands)


def _write_chunks(cfg, cache: CachePaths, source: str, job: MonthJob, site_ids, frames) -> None:
    bands = collections.SOURCE_BANDS[source]
    axis = pd.date_range(job.start, job.end, freq="h", inclusive="left")
    data = pd.concat(frames) if frames else pd.DataFrame(columns=bands)
    if len(data):
        data = data.groupby(level=["hour", "id_num"]).mean()  # (windows never overlap; defensive)
    attrs = _chunk_attrs(cfg, source, job)
    for sid in site_ids:
        if len(data) and sid in data.index.get_level_values("id_num"):
            site = data.xs(sid, level="id_num").reindex(axis)
        else:
            site = pd.DataFrame(np.nan, index=axis, columns=bands)
        ds = xr.Dataset(
            {b: ("time", site[b].to_numpy(dtype="float64")) for b in bands}, coords={"time": axis}
        )
        ds.attrs.update(attrs)
        path = cache.gee_chunk(source, sid, job.label)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)


def fetch_source(cfg: RunConfig, sites: SiteSet, source: str) -> int:
    """Fill the chunk cache for one source; returns the number of windows requested."""
    cache = CachePaths(resolve_data_root(cfg))
    span = source_span(cfg, source)
    plan: list[tuple[MonthJob, list[int]]] = []
    for job in month_jobs(*span):
        missing = [
            s
            for s in sites.ids
            if not chunk_is_valid(cache.gee_chunk(source, s, job.label), cfg, job)
        ]
        if missing:
            plan.append((job, missing))
    if not plan:
        log.info("%s: all site-months cached", source)
        return 0

    auth.initialize(cfg.gee.project)
    workers = cfg.gee.workers_era5 if source == "era5" else cfg.gee.workers_satellite
    tasks = []  # (plan index, window)
    for i, (job, _) in enumerate(plan):
        tasks += [(i, w) for w in windows(job, cfg.gee.window_hours)]
    fcs, rois = {}, {}
    for i, (_, missing) in enumerate(plan):
        fcs[i] = collections.feature_collection(sites.subset(missing))
        rois[i] = fcs[i].geometry().bounds()
    frames: dict[int, list] = {i: [] for i in range(len(plan))}
    remaining = {i: sum(1 for t in tasks if t[0] == i) for i in frames}
    failures = []

    def work(task):
        i, (t0, t1) = task
        return i, _sample_window(source, fcs[i], rois[i], t0, t1, cfg.gee.sample_scale_m)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(work, t): t for t in tasks}
        for fut in tqdm(cf.as_completed(futs), total=len(futs), desc=f"GEE {source}"):
            i, (t0, _) = futs[fut]
            try:
                _, df = fut.result()
                frames[i].append(df)
            except Exception as exc:
                log.error("%s window %s failed: %s", source, t0, exc)
                failures.append((source, plan[i][0].label, str(t0)))
            remaining[i] -= 1
            if remaining[i] == 0:
                job, missing = plan[i]
                if not any(f[1] == job.label for f in failures):
                    _write_chunks(cfg, cache, source, job, missing, frames[i])
                frames[i] = []  # free memory
    if failures:
        raise RuntimeError(f"{len(failures)} GEE window(s) failed for {source}: {failures[:5]} ...")
    return len(tasks)


def assemble_rs(cfg: RunConfig, sites: SiteSet, source: str) -> xr.Dataset:
    """Read the cached chunks into one ``(time, site)`` Dataset on the source's hourly axis."""
    cache = CachePaths(resolve_data_root(cfg))
    start, end = source_span(cfg, source)
    axis = hourly_axis(start, end).tz_localize(None)
    bands = collections.SOURCE_BANDS[source]
    arrays = {b: np.full((len(axis), len(sites)), np.nan) for b in bands}
    for job in month_jobs(start, end):
        lo = axis.get_loc(job.start)
        hi = lo + int((job.end - job.start) / pd.Timedelta(hours=1))
        for j, sid in enumerate(sites.ids):
            path = cache.gee_chunk(source, sid, job.label)
            with xr.open_dataset(path) as ds:
                for b in bands:
                    arrays[b][lo:hi, j] = ds[b].to_numpy()
    out = xr.Dataset(
        {b: (("time", "site"), arrays[b]) for b in bands},
        coords={"time": axis, "site": np.asarray(sites.ids, dtype="int64")},
        attrs={
            "source": source,
            "asset_ids": _ASSET_IDS[source],
            "scale_m": float(cfg.gee.sample_scale_m),
            "time_axis": "hourly UTC, naive; [time[0], time[-1] + 1 h)",
            "hrsmpipe_version": __version__,
        },
    )
    return out


def fetch_rs(cfg: RunConfig, sites: SiteSet, sources=SOURCES) -> dict[str, xr.Dataset]:
    """Fetch (cache-aware) and assemble ``rs/<source>.nc`` in the run directory."""
    paths = RunPaths(resolve_data_root(cfg), cfg.run.name)
    outs = {}
    for source in sources:
        fetch_source(cfg, sites, source)
        ds = assemble_rs(cfg, sites, source)
        target = paths.rs(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(target)
        outs[source] = ds
    return outs


def run(cfg: RunConfig) -> None:
    fetch_rs(cfg, load_sites(cfg))
