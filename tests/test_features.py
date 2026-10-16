import numpy as np
import pandas as pd
import pytest

from hrsmpipe.config import load_config
from hrsmpipe.features import assemble, satellite
from hrsmpipe.features.era5 import FEATURE_VARS, LOOKBACK_HOURS, make_weights, weighted_lookback
from hrsmpipe.ml.models import DROP_COLUMNS

# ---- ERA5 lookbacks --------------------------------------------------------------------------


def test_weights_favour_recent_hour_and_sum_to_one():
    w = make_weights(3)
    assert w.sum() == pytest.approx(1.0)
    raw = np.array([1 / 3, 1 / 2, 1.0])
    assert w == pytest.approx(raw / raw.sum())
    assert w[-1] > w[0]


def test_lookback_boundary_rows_raw_then_computed():
    x = np.arange(1.0, 9.0)[:, None] * np.array([[1.0, 10.0]])  # (8 hours, 2 sites)
    out = weighted_lookback(x, 3)
    w = make_weights(3)
    np.testing.assert_array_equal(out[:3], x[:3])  # rows 0..n-1 raw (row n-1 is the last raw row)
    np.testing.assert_allclose(
        out[3], (x[1:4] * w[:, None]).sum(axis=0)
    )  # first computed row uses x[1..3]
    np.testing.assert_allclose(out[7], (x[5:8] * w[:, None]).sum(axis=0))


def test_lookback_nan_propagates_through_the_window_only():
    x = np.ones((8, 1))
    x[4, 0] = np.nan
    out = weighted_lookback(x, 3)
    assert np.isnan(out[4:7, 0]).all()  # windows 2..4, 3..5, 4..6 contain the NaN
    assert not np.isnan(out[7, 0]) and not np.isnan(out[:4, 0]).any()


def test_lookback_lead_mode_slices_extended_axis():
    x = np.arange(20.0)[:, None]
    lead = 5
    out = weighted_lookback(x, 3, "lead", lead)
    assert out.shape == (15, 1)
    w = make_weights(3)
    assert out[0, 0] == pytest.approx((x[lead - 2 : lead + 1, 0] * w).sum())  # no raw warm-up rows
    with pytest.raises(ValueError):
        weighted_lookback(x, 3, "bogus")


def test_lookback_hours_and_feature_order():
    assert [LOOKBACK_HOURS[v] for v in FEATURE_VARS] == [24, 24, 72, 72, 120, 24, 24, 120]
    assert FEATURE_VARS[:2] == ["dewpoint_temperature_2m", "temperature_2m"]  # d2m, t2m
    assert FEATURE_VARS[-1] == "total_precipitation_hourly"  # tp stays in metres


# ---- satellite formulas ----------------------------------------------------------------------


def _bands(names, value):
    return {n: np.full((1, 1), float(value)) for n in names}


def test_s1_formulas_and_invalid_masking():
    f = satellite.s1_features(np.array([[-10.0]]), np.array([[-20.0]]), np.array([[38.0]]))
    vv, vh, ang, ratio, dpsvim, pol = (a[0, 0] for a in f)
    assert (vv, vh, ang) == (-10.0, -20.0, 38.0)
    assert ratio == 0.5 and dpsvim == pytest.approx(500 / np.sqrt(2)) and pol == pytest.approx(-3.0)
    bad = satellite.s1_features(np.array([[1e5]]), np.array([[-20.0]]), np.array([[38.0]]))
    assert all(
        np.isnan(a[0, 0]) for a in (bad[0], bad[3], bad[4], bad[5])
    )  # >1000 -> NaN, then features
    assert bad[1][0, 0] == -20.0


def test_s2_scaling_indices_and_tasseled_cap():
    b = _bands(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"], 1000)
    b["B8"] = np.array([[3000.0]])
    f = [a[0, 0] for a in satellite.s2_features(b)]
    assert len(f) == 15
    assert f[0] == pytest.approx(0.1) and f[6] == pytest.approx(0.3)  # x1e-4
    assert f[10] == pytest.approx((0.3 - 0.1) / (0.3 + 0.1))  # NDVI = (B8-B4)/(B8+B4)
    assert f[11] == pytest.approx((0.1 - 0.3) / (0.1 + 0.3))  # NDWI = (B3-B8)/(B3+B8)
    b2, b3, b4, b8, b11, b12 = 0.1, 0.1, 0.1, 0.3, 0.1, 0.1
    assert f[12] == pytest.approx(
        -0.3599 * b2 - 0.3533 * b3 - 0.4734 * b4 + 0.6633 * b8 - 0.0087 * b11 - 0.2856 * b12
    )
    assert f[13] == pytest.approx(
        0.3510 * b2 + 0.3813 * b3 + 0.3437 * b4 + 0.7196 * b8 + 0.2396 * b11 + 0.1949 * b12
    )
    assert f[14] == pytest.approx(
        0.2578 * b2 + 0.2305 * b3 + 0.0883 * b4 + 0.1071 * b8 - 0.7611 * b11 - 0.5308 * b12
    )


def test_hls_no_scaling_constant_angles_and_indices():
    b = _bands(["B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"], 0.1)
    b["B5"] = np.array([[0.3]])
    f = [a[0, 0] for a in satellite.hls_features(b, 90.0)]
    assert len(f) == 17
    assert f[:6] == pytest.approx([0.1, 0.1, 0.1, 0.3, 0.1, 0.1])  # raw, no 1e-4
    assert (
        f[6:8] == [0.1, 0.1] and f[8:12] == [90.0] * 4
    )  # B10, B11, then SZA/SAA/VZA/VAA constants
    assert f[12] == pytest.approx((0.3 - 0.1) / (0.3 + 0.1))  # NDVI = (B5-B4)/(B5+B4)
    assert f[13] == pytest.approx((0.1 - 0.3) / (0.1 + 0.3))  # NDWI = (B3-B5)/(B3+B5)
    assert f[14] == pytest.approx(
        -0.2941 * 0.1 - 0.243 * 0.1 - 0.5424 * 0.1 + 0.7276 * 0.3 + 0.0713 * 0.1 - 0.1608 * 0.1
    )
    assert satellite.hls_features(b, 45.0)[8][0, 0] == 45.0


# ---- column tables and the assembled table ---------------------------------------------------


def test_column_tables():
    assert [len(assemble.COLUMNS[s]) for s in ("sentinel1", "sentinel2", "hls")] == [30, 39, 41]
    s1, s2, hls = assemble.COLUMNS_S1, assemble.COLUMNS_S2, assemble.COLUMNS_HLS
    assert s1[:24] == s2[:24] == hls[:24] and s1[15] == "smap_constant" and s1[0] == "site_id"
    assert s1[24:27] == ["VV", "VH", "angle"] and s2[24] == "B2" and s2[33] == "B12"
    assert hls[30:32] == ["B10", "B11"] and hls[32:36] == ["SZA", "SAA", "VZA", "VAA"]
    assert assemble.feature_columns("hls")[-1] == "f40"


def test_drop_columns_remove_the_documented_features():
    def names(sensor, fam):
        return [c for i, c in enumerate(assemble.COLUMNS[sensor]) if i in DROP_COLUMNS[fam]]

    assert names("sentinel1", "ML1") == ["site_id", "year", "hour", "smap_constant"]
    assert names("sentinel2", "ML2")[4:] == ["B2", "B3", "B4"]
    assert names("hls", "ML3")[4:] == ["B2", "B3", "B4", "SZA", "SAA", "VZA", "VAA"]
    assert [30 - 4, 39 - 7, 41 - 11] == [26, 32, 30]


def _tiny_inputs():
    cfg = load_config(None, {"run.start": "2024-02-28", "run.end": "2024-02-29"})  # 24 h, day 59
    from hrsmpipe.sites import SiteSet

    sites = SiteSet(
        pd.DataFrame(
            {
                "site_id": [7, 47],
                "station_code": ["a", "b"],
                "latitude": [43.0, 45.0],
                "longitude": [-89.0, -90.0],
            }
        )
    )
    static = pd.DataFrame(
        {"latitude": [43.0, 45.0], "longitude": [-89.0, -90.0], "elevation": [200.0, 300.0], "slope": [1.0, 2.0],
         "sin_asp": [0.1, 0.2], "cos_asp": [0.9, 0.8], "twi": [5.0, 6.0], "bd_0_100": [1.4, 1.5],
         "sand_0_100": [20.0, 30.0], "clay_0_100": [10.0, 15.0], "ksat_0_100": [2.0, 3.0]},
        index=pd.Index([7, 47], name="site_id"),
    )  # fmt: skip
    return cfg, sites, static


def test_build_table_layout_nan_rows_and_order():
    import xarray as xr

    cfg, sites, static = _tiny_inputs()
    T = 24
    lookbacks = [np.full((T, 2), float(i)) for i in range(8)]
    vv = np.full((T, 2), np.nan)
    vh = np.full((T, 2), np.nan)
    ang = np.full((T, 2), np.nan)
    vv[3, 1], vh[3, 1], ang[3, 1] = -10.0, -20.0, 38.0  # one complete (hour 3, site 47) sample
    vv[5, 0], vh[5, 0], ang[5, 0] = -11.0, -21.0, 39.0  # and one at (hour 5, site 7)
    rs = xr.Dataset(
        {
            "VV": (("time", "site"), vv),
            "VH": (("time", "site"), vh),
            "angle": (("time", "site"), ang),
        }
    )
    df = assemble.build_table(cfg, "sentinel1", sites, static, lookbacks, rs)
    assert (
        len(df) == 2
        and list(df.columns[:2]) == ["f00", "f01"]
        and list(df.columns[-2:]) == ["site_id", "time"]
    )
    assert df["site_id"].tolist() == [47, 7]  # time-major: hour 3 first, then hour 5
    r = df.iloc[0]
    assert (r.f00, r.f01, r.f02, r.f03) == (47, 2024, 59, 3)  # year, doy (Feb 28 = 59), hour
    assert (r.f04, r.f05, r.f06, r.f10) == (45.0, -90.0, 300.0, 6.0)  # lat, lon, elevation, TWI
    assert r.f15 == 0.3 and [r[f"f{i}"] for i in range(16, 24)] == list(map(float, range(8)))
    assert (r.f24, r.f27, r.f29) == (-10.0, 0.5, pytest.approx(-3.0))
    assert df["time"].iloc[0] == pd.Timestamp("2024-02-28 03:00", tz="UTC")
    assert not df.drop(columns="time").isna().any().any()  # rows with any NaN are dropped
