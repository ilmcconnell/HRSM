import tomllib

import pytest
from pydantic import ValidationError

from hrsmpipe.config import RunConfig, load_config, resolved_toml


def test_defaults_match_notebook():
    c = RunConfig()
    assert (c.run.start, c.run.end) == ("2021-07-16", "2026-02-10")
    assert c.gee.window_hours == 15 and c.gee.era5_lead_hours == 144
    assert c.gee.workers_era5 == 2 and c.gee.workers_satellite == 4
    assert c.obs.rootzone_weights == [7.5, 7.5, 20, 40, 25]
    assert c.postproc.reference_sensor == "sentinel2"
    assert c.forcing.interval_shift_hours == 0
    assert c.lgar.ensemble_size == 20 and c.lgar.layer_thickness_cm == [5, 10, 15, 30, 40]
    assert c.lgar.max_valid_soil_types == 12


def test_file_and_dotted_overrides(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[run]\nname = "a"\nstart = 2020-01-01\nend = "2020-02-01"\n[gee]\nwindow_hours = 6\n'
    )
    c = load_config(p, {"gee.window_hours": 3, "obs.columns.time": "ts"})
    assert c.run.name == "a" and c.run.start == "2020-01-01"  # native TOML date accepted
    assert c.gee.window_hours == 3  # override beats the file
    assert c.obs.columns.time == "ts"


def test_unknown_key_and_bad_dates_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_config(None, {"gee.nope": 1})
    with pytest.raises(ValidationError):
        load_config(None, {"run.start": "2025-01-02", "run.end": "2025-01-01"})


def test_resolved_dump_roundtrips_and_provenance_ignored(tmp_path):
    c = load_config(
        None, {"run.name": "x", "lgar.site_overrides": {"7": {"layer_soil_type": [3, 3, 4, 4, 4]}}}
    )
    text = resolved_toml(c, {"git_tag": None, "git_dirty": True, "python": "3.13"})
    assert "[provenance]" in text and "git_tag" not in text  # nulls cannot be written to TOML
    p = tmp_path / "r.toml"
    p.write_text(text)
    assert "provenance" in tomllib.loads(text)
    again = load_config(p)  # [provenance] ignored, rest identical
    assert again == c
    assert again.lgar.site_overrides[7]["layer_soil_type"] == [3, 3, 4, 4, 4]


def test_resolved_toml_documents_legacy_weights():
    assert "[2.5, 7.5, 20, 40, 25]" in resolved_toml(RunConfig())


def test_override_keys_validated():
    with pytest.raises(ValidationError):
        load_config(None, {"lgar.site_overrides": {"7": {"ensemble_size": 5}}})  # build parameter
    with pytest.raises(ValidationError):
        load_config(None, {"obs.rootzone_weights": [1, 2]})  # length mismatch
