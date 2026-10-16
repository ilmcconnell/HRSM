import pandas as pd
import pytest

from hrsmpipe.obs import wisconet_api as w


def _wide(**cols):
    return pd.DataFrame(
        cols,
        index=pd.date_range(
            "2024-06-01", periods=len(next(iter(cols.values()))), freq="h", tz="UTC"
        ),
    )


def test_unit_table():
    wide = _wide(
        **{
            "60min_soil_moisture_pct_avg@2in": [0.34, 0.40],
            "60min_soil_moisture_pct_avg@40in": [0.52, 0.53],
            "60min_rain_in_tot": [0.0, 0.1],
            "60min_air_temp_f_avg": [32.0, 212.0],
            "60min_solar_radiation_w_avg": [0.0, 700.0],
        }
    )
    out = w.convert_fields(wide)
    assert list(out.columns) == w.OUT_COLUMNS
    assert out["precip_mm"].tolist() == pytest.approx([0.0, 2.54])
    assert out["air_temp_c"].tolist() == pytest.approx([0.0, 100.0])
    assert out["solar_w"].tolist() == [0.0, 700.0]
    assert out["sm_5"].tolist() == [0.34, 0.40]  # API already returns 0-1: not divided
    assert out["sm_100"].tolist() == [0.52, 0.53]
    assert out["sm_10"].isna().all()  # absent fields are NaN columns


def test_percent_sm_is_divided_and_out_of_range_rejected():
    out = w.convert_fields(_wide(**{"60min_soil_moisture_pct_avg@2in": [34.0, 40.0]}))
    assert out["sm_5"].tolist() == pytest.approx([0.34, 0.40])
    with pytest.raises(ValueError):
        w.convert_fields(_wide(**{"60min_soil_moisture_pct_avg@2in": [34.0, 140.0]}))


def test_solar_kwm2_only_as_fallback():
    only_kw = w.convert_fields(_wide(**{"60min_solar_radiation_kwm2_tot": [0.5]}))
    assert only_kw["solar_w"].iloc[0] == 500.0
    both = w.convert_fields(
        _wide(**{"60min_solar_radiation_w_avg": [10.0], "60min_solar_radiation_kwm2_tot": [0.5]})
    )
    assert both["solar_w"].iloc[0] == 10.0


def test_depth_map_covers_rootzone_depths():
    assert list(w.SM_FIELDS.values()) == ["sm_5", "sm_10", "sm_20", "sm_50", "sm_100"]


def test_month_ranges_clip_to_span():
    r = w.month_ranges("2024-01-30", "2024-03-02")
    assert [x[0] for x in r] == ["2024-01", "2024-02", "2024-03"]
    assert r[0][1] == pd.Timestamp("2024-01-30", tz="UTC") and r[0][2] == pd.Timestamp(
        "2024-02-01", tz="UTC"
    )
    assert r[-1][2] == pd.Timestamp("2024-03-02", tz="UTC")


def test_chunk_cache_reuse_requires_coverage(tmp_path):
    lo, hi = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-02-01", tz="UTC")
    df = pd.DataFrame({"sm_5": [0.1]}, index=pd.DatetimeIndex([lo], name="time"))
    p = tmp_path / "c.parquet"
    w._write_chunk(p, df, lo, pd.Timestamp("2024-01-15", tz="UTC"))
    assert w._read_chunk(p, lo, pd.Timestamp("2024-01-10", tz="UTC")) is not None
    assert w._read_chunk(p, lo, hi) is None  # run extended past what was fetched -> refetch


def test_empty_month_does_not_poison_the_site_frame(monkeypatch, cfg, data_root):
    """A month without data (empty API response) must not turn the columns into object dtype."""
    import pandas as pd

    from hrsmpipe.obs import _add_rootzone

    full = w.convert_fields(
        pd.DataFrame(
            {f: [0.2, 0.3] for f in w.SM_FIELDS},
            index=pd.date_range("2024-06-01", periods=2, freq="h", tz="UTC"),
        )
    )
    both = pd.concat([w._empty_frame(), full])
    assert (both.dtypes == "float64").all()
    out = _add_rootzone(both, cfg)
    assert out["sm_rootzone"].notna().sum() == 2
    # object columns (e.g. from an older cache file) are tolerated as well
    odd = both.astype({"sm_5": "object"})
    assert _add_rootzone(odd, cfg)["sm_rootzone"].notna().sum() == 2


def test_api_empty_response_returns_float_frame(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(w, "_throttle", lambda: None)
    import wiscopy.data as wdata
    import wiscopy.process as wprocess

    monkeypatch.setattr(wdata, "bulk_measures", lambda *a, **k: object())
    monkeypatch.setattr(wprocess, "bulk_measures_to_df", lambda bms: None)
    df = w.fetch_range(
        "ALTN", pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC")
    )
    assert df.empty and (df.dtypes == "float64").all() and str(df.index.tz) == "UTC"
