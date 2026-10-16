import numpy as np
import pytest
from pydantic import ValidationError

from hrsmpipe.config import load_config
from hrsmpipe.lgar import build, merge, settings


# ---- settings writer -------------------------------------------------------------------------
def test_settings_format_lists_bools_and_order():
    cfg = load_config(None)
    s = settings.site_settings(cfg, 296, 78912, [6, 6, 6, 9, 9])
    text = settings.format_settings(s)
    lines = text.splitlines()
    assert (
        lines[0] == "verbosity: none"
        and lines[1] == "site_ID: 296"
        and lines[2] == "num_hours: 78912"
    )
    assert "layer_thickness_cm: [5, 10, 15, 30, 40]" in lines
    assert "giuh_ordinates: [0.06, 0.51, 0.28, 0.12, 0.03]" in lines
    assert "layer_soil_type: [6, 6, 6, 9, 9]" in lines
    assert "endtime_s: 284083200" in lines  # 3600 * N
    assert (
        "TO_enabled: false" in lines and "use_closed_form_G: false" in lines
    )  # lowercase booleans
    assert "field_capacity_psi: 340.9" in lines and "max_valid_soil_types: 12" in lines
    assert not any(line.startswith("soil_params_file") for line in lines)  # empty -> not written
    assert all(": " in line for line in lines)  # `key: value`, as the C++ parser splits on ':'
    assert [line.split(":")[0] for line in lines] == list(s)


def test_settings_cover_every_key_the_cpp_parser_reads():
    cfg = load_config(None, {"lgar.soil_params_file": "x.dat"})
    keys = set(settings.site_settings(cfg, 1, 24, [1] * 5))
    assert keys == set(settings.KEY_ORDER)
    for derived in ("site_ID", "num_hours", "endtime_s", "layer_soil_type"):
        with pytest.raises(ValidationError):  # derived at run time, never a config key
            load_config(None, {f"lgar.{derived}": 1})


def test_site_override_replaces_soil_classes_and_other_keys():
    cfg = load_config(
        None,
        {"lgar.site_overrides": {"7": {"layer_soil_type": [3, 3, 4, 4, 4], "initial_psi": 1500.0}}},
    )
    s7 = settings.site_settings(cfg, 7, 24, [1, 1, 1, 1, 1])
    assert s7["layer_soil_type"] == [3, 3, 4, 4, 4] and s7["initial_psi"] == 1500.0
    s8 = settings.site_settings(cfg, 8, 24, [1, 1, 1, 1, 1])
    assert s8["layer_soil_type"] == [1] * 5 and s8["initial_psi"] == 2000.0


def test_write_settings_file(tmp_path):
    p = tmp_path / "s.txt"
    settings.write_settings(p, {"a": [1, 2.5], "b": True, "c": 3})
    assert p.read_text() == "a: [1, 2.5]\nb: true\nc: 3\n"


# ---- binary names ----------------------------------------------------------------------------
def test_binary_name_formula():
    assert build.enkf_binary_name(load_config(None)) == "EnKF_LGAR_20_003002"
    c = load_config(
        None, {"lgar.ensemble_size": 50, "lgar.model_sd_sly": 0.05, "lgar.model_sd_rly": 0.03}
    )
    assert build.enkf_binary_name(c) == "EnKF_LGAR_50_005003"
    assert (
        build.enkf_binary_name(load_config(None, {"lgar.seed": 7})) == "EnKF_LGAR_20_003002_seed7"
    )
    assert build.sd_tag(0.03, 0.02) == "003002"
    assert build.sd_tag(0.1, 0.02) == "010002"


def test_binary_paths_use_bin_dir(data_root):
    cfg = load_config(None)
    paths = build.binary_paths(cfg)
    assert paths["original"].name == "LGAR" and paths["original"].parent.parent == data_root / "bin"
    custom = load_config(None, {"lgar.bin_dir": "/tmp/xbin"})
    assert str(build.binary_paths(custom)["enkf"]).startswith("/tmp/xbin/EnKF_LGAR_20_")


# ---- source patching -------------------------------------------------------------------------
@pytest.fixture
def enkf_source():
    return (build.bundled_source("enkf") / "bmi_main_lgar.cxx").read_text()


def test_bundled_sources_are_complete():
    for kind in ("original", "enkf"):
        root = build.bundled_source(kind)
        for rel in build.SOURCES:
            assert (root / rel).exists(), f"{kind}/{rel}"
        assert not list(root.rglob("*.bin")) and not list(root.rglob(".DS_Store"))


def test_patches_apply_exactly_once(enkf_source):
    assert "int N_ensemble = 100;" in enkf_source  # the zip ships the 100-member variant
    text, patches = build.patch_enkf_source(enkf_source, 20, 0.03, 0.02, 0)
    assert "int N_ensemble = 20;" in text and "N_ensemble = 100" not in text
    assert "model_error_sly(0.0, 0.03)" in text and "model_error_rly(0.0, 0.02)" in text
    assert "default_random_engine generator(rd());" in text  # unseeded keeps random_device
    assert len(patches) == 3
    text2, patches2 = build.patch_enkf_source(enkf_source, 50, 0.05, 0.03, 42)
    assert "int N_ensemble = 50;" in text2
    assert "model_error_sly(0.0, 0.05)" in text2 and "model_error_rly(0.0, 0.03)" in text2
    assert "default_random_engine generator(42);" in text2 and "generator(rd())" not in text2
    assert len(patches2) == 4


def test_patch_fails_when_pattern_missing_or_ambiguous(enkf_source):
    with pytest.raises(build.BuildError, match="ensemble size"):
        build.patch_enkf_source(
            enkf_source.replace("int N_ensemble = 100;", "int N = 100;"), 20, 0.03, 0.02, 0
        )
    with pytest.raises(build.BuildError, match="matched 2"):
        build.patch_enkf_source(enkf_source + "\nint N_ensemble = 5;\n", 20, 0.03, 0.02, 0)
    with pytest.raises(build.BuildError, match="random seed"):
        build.patch_enkf_source(
            enkf_source.replace("generator(rd())", "generator(7)"), 20, 0.03, 0.02, 5
        )
    # unseeded builds do not need the generator line at all
    build.patch_enkf_source(
        enkf_source.replace("generator(rd())", "generator(7)"), 20, 0.03, 0.02, 0
    )


# ---- merge helpers ---------------------------------------------------------------------------
def test_merge_columns_match_the_specified_order():
    assert merge.COLUMNS[:4] == [
        "forcast_sd_sly",
        "forcast_sd_rly",
        "analysis_sd_sly",
        "analysis_sd_rly",
    ]
    assert merge.COLUMNS[12:16] == [
        "sly_sm_ml_arr",
        "rly_sm_ml_arr",
        "observation_sly",
        "observation_rly",
    ]
    assert merge.COLUMNS[-4:] == ["precip_mm", "precip_source", "pet_mm", "pet_source"][-4:]
    assert len(merge.COLUMNS) == 30 and len(set(merge.COLUMNS)) == 30


def test_cdf_to_ignores_nan_and_empty_target():
    s = np.array([3.0, np.nan, 1.0, 2.0])
    out = merge.cdf_to(s, np.array([10.0, 20.0, 30.0]))
    assert np.isnan(out[1]) and out[0] == 30.0 and out[2] == pytest.approx(10.0)
    assert np.isnan(merge.cdf_to(s, np.array([]))).all()


def test_sd_band_maps_through_the_monotone_cdf_mapping():
    a = np.linspace(0.1, 0.4, 301)
    sd = np.full(a.shape, 0.01)
    identity = merge.sd_band_cdf(a, sd, a.copy())  # analysis already in the ML climatology
    assert identity[100] == pytest.approx(0.01, abs=1e-6)
    doubled = merge.sd_band_cdf(a, sd, 2 * a)  # mapping that stretches by 2 doubles the half width
    assert doubled[100] == pytest.approx(0.02, abs=1e-6)
    assert np.isnan(merge.sd_band_cdf(np.array([np.nan]), np.array([0.01]), np.array([0.2]))).all()
