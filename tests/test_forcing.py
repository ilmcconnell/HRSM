import numpy as np
import pandas as pd
import pytest

from hrsmpipe.config import load_config
from hrsmpipe.forcing import pet, precip, soils


# ---- Priestley-Taylor ------------------------------------------------------------------------
def _pt_scalar(r_n, t, alpha=1.3, gamma=66.8):
    """Independent scalar implementation of notebook cell 46."""
    import math

    e_s = 611 * math.exp(17.27 * t / (273.3 + t))
    delta = 4098 * e_s / (273.3 + t) ** 2
    l_v = 2500 - 2.36 * t
    e_r = r_n / (l_v * 1000 * 1000)
    return 3.6e6 * e_r * alpha * delta / (delta + gamma)


@pytest.mark.parametrize("r_n, t", [(500.0, 20.0), (100.0, 0.0), (800.0, 30.0), (50.0, -5.0)])
def test_priestley_taylor_known_values(r_n, t):
    assert pet.priestley_taylor(np.array(r_n), np.array(t)) == pytest.approx(_pt_scalar(r_n, t))


def test_priestley_taylor_reference_value_and_properties():
    # R_n = 500 W/m2 at 20 C is a bit over 0.4 mm/h: hand check of the notebook formula
    v = float(pet.priestley_taylor(np.array(500.0), np.array(20.0)))
    assert 0.5 < v < 0.8
    assert pet.priestley_taylor(np.array(0.0), np.array(15.0)) == 0
    a = pet.priestley_taylor(np.array(300.0), np.array(20.0), alpha=1.3)
    b = pet.priestley_taylor(np.array(300.0), np.array(20.0), alpha=2.6)
    assert b == pytest.approx(2 * a)
    assert pet.priestley_taylor(np.array(300.0), np.array(25.0)) > pet.priestley_taylor(
        np.array(300.0), np.array(5.0)
    )


def test_pet_threshold_gap_fill_and_sources():
    cfg = load_config(None)
    ssr = np.array(
        [[0.0, 3.6e6], [1.8e6, np.nan], [np.nan, np.nan], [3.6e6, np.nan]]
    )  # J/m2 -> /3600 W/m2
    t2m = np.full(ssr.shape, 293.15)
    out, src = pet.build_pet(cfg, ssr, t2m)
    assert out[0, 0] == 0  # below zero_below -> 0
    assert not np.isnan(out[:, 0]).any() and src[:, 0].tolist() == [
        2,
        2,
        3,
        2,
    ]  # hour 2 interpolated
    assert out[2, 0] == pytest.approx((out[1, 0] + out[3, 0]) / 2)
    # 2nd site: a single valid hour; both-direction interpolation spreads it over the other hours
    assert np.allclose(out[:, 1], out[0, 1]) and src[:, 1].tolist() == [2, 3, 3, 3]


def test_pet_fully_missing_site_is_skipped(caplog):
    cfg = load_config(None)
    out, src = pet.build_pet(cfg, np.full((3, 1), np.nan), np.full((3, 1), 290.0))
    assert np.isnan(out).all() and (src == pet.PET_MISSING).all()


def test_pet_mesonet_source_with_era5_fallback():
    cfg = load_config(None, {"forcing.pet.source": "mesonet"})
    ssr = np.full((3, 1), 3.6e6)
    t2m = np.full((3, 1), 293.15)
    solar = np.array([[700.0], [np.nan], [700.0]])
    airt = np.array([[20.0], [20.0], [np.nan]])
    out, src = pet.build_pet(cfg, ssr, t2m, solar, airt)
    assert src[:, 0].tolist() == [pet.PET_MESONET, pet.PET_ERA5, pet.PET_ERA5]
    assert out[0, 0] == pytest.approx(_pt_scalar(700.0, 20.0))
    assert out[1, 0] == pytest.approx(_pt_scalar(1000.0, 20.0))  # ERA5: 3.6e6 J / 3600 = 1000 W/m2
    nofb = load_config(None, {"forcing.pet.source": "mesonet", "forcing.pet.fallback": "none"})
    _, src2 = pet.build_pet(nofb, ssr, t2m, solar, airt)
    assert src2[:, 0].tolist() == [pet.PET_MESONET, pet.PET_INTERPOLATED, pet.PET_INTERPOLATED]


# ---- precipitation ---------------------------------------------------------------------------
def test_era5_precip_units_and_zero_threshold():
    mm = precip.era5_precip_mm(np.array([0.0, 0.000005, 0.0001, np.nan]), 0.01)
    assert mm[:3].tolist() == [0.0, 0.0, pytest.approx(0.1)] and np.isnan(
        mm[3]
    )  # m -> mm, <0.01 -> 0


def test_precip_blend_and_provenance_codes():
    cfg = load_config(None)  # mesonet with ERA5 fallback
    era5_m = np.array([[0.001], [0.002], [np.nan], [np.nan]])  # 1, 2 mm, NaN, NaN
    mesonet = np.array([[5.0], [np.nan], [np.nan], [0.0]])
    prec, src = precip.build_precip(cfg, era5_m, mesonet)
    assert prec[:, 0].tolist() == [5.0, 2.0, 0.0, 0.0]
    assert src[:, 0].tolist() == [
        precip.PRECIP_MESONET,
        precip.PRECIP_ERA5,
        precip.PRECIP_MISSING,
        precip.PRECIP_MESONET,
    ]
    assert src.dtype == np.int8
    assert precip.PRECIP_SOURCE_ATTRS["flag_meanings"].split() == ["missing", "mesonet", "era5"]


def test_precip_era5_only_and_no_fallback():
    era5_m = np.array([[0.001], [np.nan]])
    prec, src = precip.build_precip(
        load_config(None, {"forcing.precip.source": "era5"}), era5_m, np.array([[9.0], [9.0]])
    )
    assert prec[:, 0].tolist() == [1.0, 0.0] and src[:, 0].tolist() == [
        precip.PRECIP_ERA5,
        precip.PRECIP_MISSING,
    ]
    nofb = load_config(None, {"forcing.precip.fallback": "none"})
    prec, src = precip.build_precip(nofb, era5_m, np.array([[np.nan], [3.0]]))
    assert prec[:, 0].tolist() == [0.0, 3.0] and src[:, 0].tolist() == [
        precip.PRECIP_MISSING,
        precip.PRECIP_MESONET,
    ]
    prec, src = precip.build_precip(load_config(None), era5_m, None)  # no observations at all
    assert src[:, 0].tolist() == [precip.PRECIP_ERA5, precip.PRECIP_MISSING]


def test_interval_shift_applies_to_both_sources_together():
    arr = np.arange(1.0, 5.0)[:, None]
    assert precip.shift_hours(arr, 0) is arr
    assert precip.shift_hours(arr, 1)[:, 0].tolist()[1:] == [1.0, 2.0, 3.0] and np.isnan(
        precip.shift_hours(arr, 1)[0, 0]
    )
    assert precip.shift_hours(arr, -1)[:, 0].tolist()[:3] == [2.0, 3.0, 4.0]
    cfg = load_config(None, {"forcing.interval_shift_hours": 1})
    era5_m = np.array([[0.001], [0.002], [0.003]])
    mes = np.array([[np.nan], [4.0], [np.nan]])
    prec, src = precip.build_precip(cfg, era5_m, mes)
    # mesonet (4.0) moves to hour 2; ERA5 moves too (1 mm -> hour 1, 2 mm -> hour 2); hour 0 is missing
    assert prec[:, 0].tolist() == [0.0, 1.0, 4.0]
    assert src[:, 0].tolist() == [precip.PRECIP_MISSING, precip.PRECIP_ERA5, precip.PRECIP_MESONET]


# ---- soil classes ----------------------------------------------------------------------------
TEXTURES = {  # soil_id: (sand %, clay %) picked inside the notebook's masks
    1: (90, 7), 2: (85, 8), 3: (65, 10), 4: (40, 18), 5: (5, 8), 6: (20, 15),
    7: (60, 27), 8: (33, 33), 9: (10, 34), 10: (52, 40), 11: (6, 47), 12: (20, 60),
}  # fmt: skip


def test_soil_masks_on_the_twelve_textures():
    sand = np.array([float(s) for s, _ in TEXTURES.values()])
    clay = np.array([float(c) for _, c in TEXTURES.values()])
    soilid, theta_r, theta_e, alpha, n, ks = soils.extract_soil_properties(
        sand, 100 - sand - clay, clay
    )
    assert soilid.tolist() == list(map(float, TEXTURES))
    assert (
        ks[0] == 29.7
        and ks[11] == 0.2
        and theta_r[4] == 0.034
        and alpha[2] == 0.075
        and n[3] == 1.56
    )
    assert theta_e.tolist()[:3] == [0.43, 0.41, 0.41]
    assert list(soils.SOIL_NAMES).index("silt_loam") == 5  # soilMap order: ... silt=5, silt_loam=6


def test_sand_loamy_sand_are_overridden_by_sandy_loam_when_clay_below_7():
    """Notebook quirk kept verbatim: `sandy_loam` is assigned after `sand`/`loamy_sand`."""
    sand, clay = np.array([92.0]), np.array([3.0])
    assert soils.extract_soil_properties(sand, 100 - sand - clay, clay)[0][0] == 3


def _static_for(textures):
    cols = {}
    for depth in ("0_5", "5_15", "15_30", "30_60", "60_100"):
        cols[f"sand_{depth}"] = [float(t[0]) for t in textures]
        cols[f"clay_{depth}"] = [float(t[1]) for t in textures]
    return pd.DataFrame(cols, index=pd.Index(range(len(textures)), name="site_id"))


def test_soil_classes_table_and_nan_rejection():
    static = _static_for([TEXTURES[4], TEXTURES[12], (np.nan, np.nan)])
    table = soils.soil_classes(static)
    assert list(table.columns) == soils.SOIL_COLUMNS
    assert len(table) == 15 and table["layer"].tolist()[:5] == [1, 2, 3, 4, 5]
    assert table["depth"].tolist()[:5] == ["0_5", "5_15", "15_30", "30_60", "60_100"]
    check = soils.valid_soil_ids(table)
    assert check[0] is None and check[1] is None
    assert "invalid soil_id" in check[2]  # NaN would be cast to int by the C++ parser
    bad = table.copy()
    bad.loc[bad["site_id"] == 0, "soil_id"] = [4, 4, 13, 4, 4]
    assert "layers [3]" in soils.valid_soil_ids(bad)[0]
    bad.loc[bad["site_id"] == 0, "soil_id"] = [4, 4, 2.5, 4, 4]
    assert soils.valid_soil_ids(bad)[0] is not None
