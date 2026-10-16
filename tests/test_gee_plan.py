import pandas as pd
import xarray as xr

from hrsmpipe.config import load_config
from hrsmpipe.gee import collections, timeseries
from hrsmpipe.manifest import fingerprint, is_done, mark_done, stages_to_run
from hrsmpipe.paths import CachePaths


def test_spans_and_months():
    cfg = load_config(None, {"run.start": "2024-01-20", "run.end": "2024-03-05"})
    assert timeseries.source_span(cfg, "s2") == (
        pd.Timestamp("2024-01-20"),
        pd.Timestamp("2024-03-05"),
    )
    assert timeseries.source_span(cfg, "era5")[0] == pd.Timestamp("2024-01-14")  # 144 h lead
    jobs = timeseries.month_jobs(*timeseries.source_span(cfg, "s2"))
    assert [j.label for j in jobs] == ["2024-01", "2024-02", "2024-03"]
    assert jobs[0].start == pd.Timestamp("2024-01-20") and jobs[0].end == pd.Timestamp("2024-02-01")
    assert jobs[-1].end == pd.Timestamp("2024-03-05")


def test_windows_are_15h_aligned_and_cover_each_hour_once():
    job = timeseries.MonthJob("2024-02", pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01"))
    ws = timeseries.windows(job, 15)
    assert ws[0] == (pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-01 15:00"))
    assert ws[-1][1] == job.end
    assert all(a[1] == b[0] for a, b in zip(ws, ws[1:], strict=False))  # contiguous, no overlap


def test_canonical_orders():
    assert collections.ERA5_VARS == [
        "temperature_2m", "dewpoint_temperature_2m", "wind_10m", "surface_pressure",
        "surface_solar_radiation_downwards_hourly", "surface_thermal_radiation_downwards_hourly",
        "total_evaporation_hourly", "total_precipitation_hourly",
    ]  # fmt: skip
    assert len(collections.HLS_BANDS) == 9 and len(collections.S2_BANDS) == 10
    assert collections.S1_BANDS == ["VV", "VH", "angle"]


def test_chunk_validity_needs_range_and_scale(cfg, data_root):
    job = timeseries.MonthJob("2024-01", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03"))
    path = CachePaths(data_root).gee_chunk("s2", 7, "2024-01")
    assert not timeseries.chunk_is_valid(path, cfg, job)
    timeseries._write_chunks(cfg, CachePaths(data_root), "s2", job, [7], [])
    assert timeseries.chunk_is_valid(path, cfg, job)
    longer = timeseries.MonthJob("2024-01", job.start, pd.Timestamp("2024-01-10"))
    assert not timeseries.chunk_is_valid(path, cfg, longer)  # coverage too short -> refetch
    cfg2 = load_config(None, {"gee.sample_scale_m": 30})
    assert not timeseries.chunk_is_valid(path, cfg2, job)
    with xr.open_dataset(path) as ds:
        assert (
            ds.attrs["scale_m"] == 100.0 and ds.attrs["window_hours"] == 15 and len(ds.time) == 48
        )


def test_chunks_assemble_with_nan_gaps(cfg, data_root):
    from hrsmpipe.sites import load_sites

    cfg = load_config(
        None, {"run.start": "2024-01-01", "run.end": "2024-01-02", "sites.include": [7, 47]}
    )
    sites = load_sites(cfg)
    job = timeseries.month_jobs(*timeseries.source_span(cfg, "s1"))[0]
    row = pd.DataFrame(
        {"VV": [-10.0], "VH": [-15.0], "angle": [38.0]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-01-01 05:00"), 7)], names=["hour", "id_num"]
        ),
    )
    timeseries._write_chunks(cfg, CachePaths(data_root), "s1", job, sites.ids, [row])
    ds = timeseries.assemble_rs(cfg, sites, "s1")
    assert ds.VV.shape == (24, 2) and ds.site.values.tolist() == [
        47,
        7,
    ]  # CSV order is column order
    assert ds.VV.values[5, 1] == -10.0 and ds.VV.notnull().sum() == 1


def test_manifest_fingerprint_and_force(cfg, tmp_path):
    fp = fingerprint("fetch-rs", cfg)
    assert fp == fingerprint("fetch-rs", cfg.model_copy(update={}))  # deterministic
    renamed = load_config(
        None, {"run.name": "other", "run.start": "2024-01-01", "run.end": "2024-01-03"}
    )
    assert fingerprint("fetch-rs", renamed) == fp  # run.name is excluded
    longer = load_config(
        None, {"run.name": "t", "run.start": "2024-01-01", "run.end": "2024-01-04"}
    )
    assert fingerprint("fetch-rs", longer) != fp
    assert not is_done(tmp_path, fp)
    mark_done(tmp_path, fp, "a", "b", {"git_tag": None})
    assert is_done(tmp_path, fp) and not is_done(tmp_path, "x")
    assert (
        stages_to_run(("a", "b", "c"), "b") == {"b", "c"} and stages_to_run(("a",), None) == set()
    )
