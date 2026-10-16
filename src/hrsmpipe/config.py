"""Run configuration: one TOML file completely defines a run.

Every key has a default equal to the value used in the original notebook
(``HRSM_MAPPING_ML_pipeline_multiplePoints_lgar_inputs.ipynb``).
"""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path
from typing import Any, Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunSection(_Model):
    name: str = "wisconet_2021_2026"  # run directory name under <data_root>/runs/ (excluded from stage fingerprints)
    data_root: str = ""  # "" -> $HRSM_DATA_DIR
    start: str = "2021-07-16"  # UTC date; axis is [start, end)
    end: str = "2026-02-10"  # UTC date, exclusive

    @field_validator("start", "end", mode="before")
    @classmethod
    def _date_to_str(cls, v: Any) -> str:
        # TOML has native dates; accept them as well as quoted strings.
        if isinstance(v, dt.datetime):
            v = v.date()
        if isinstance(v, dt.date):
            return v.isoformat()
        dt.date.fromisoformat(str(v))
        return str(v)

    @model_validator(mode="after")
    def _ordered(self) -> RunSection:
        if dt.date.fromisoformat(self.end) <= dt.date.fromisoformat(self.start):
            raise ValueError("run.end must be after run.start")
        return self


class SitesSection(_Model):
    csv: str = "pkg:wisconet_stations.csv"  # sites CSV (site_id,station_code,latitude,longitude); `pkg:` = packaged file; row order = array column order
    include: list[int] = []  # optional subset of site_id (CSV order is kept)


class GeeSection(_Model):
    project: str = "run-hrsm"  # Earth Engine cloud project
    workers_era5: int = 2  # ERA5 request threads
    workers_satellite: int = 4  # Sentinel-1/2 and HLSL30 request threads
    window_hours: int = 15  # hours per Earth Engine request window (aligned to the month start)
    sample_scale_m: float = 100  # NLCD and all time series
    polaris_scale: float | Literal["native"] = "native"  # bd_0_5.projection().nominalScale() (30 m)
    terrain_scale: float | Literal["native"] = "native"  # each asset's nominalScale()
    era5_lead_hours: int = 144  # 120 + 24, as the notebook's start_date_era5


class ObsColumns(_Model):
    time: str = "time"  # timestamp column
    site_id: str = "site_id"  # site column (not needed when `obs.path` contains `{site_id}`)
    sm_surface: str = "sm_surface"  # surface soil moisture column
    sm_rootzone: str = "sm_rootzone"  # rootzone soil moisture column
    precip_mm: str = "precip_mm"  # hourly precipitation column (mm); "" = none
    air_temp_c: str = (
        ""  # air temperature column (deg C); needed for `forcing.pet.source = "mesonet"`
    )
    solar_w: str = ""  # solar radiation column (W/m2); needed for `forcing.pet.source = "mesonet"`


class ObsSection(_Model):
    source: Literal["wisconet_api", "wisconet_csv_dir", "table"] = (
        "wisconet_api"  # observation source
    )
    path: str = ""  # csv dir / table path (may contain {site_id})
    timezone: str = "UTC"  # naive table timestamps only
    rootzone_depths_cm: list[float] = [
        5,
        10,
        20,
        50,
        100,
    ]  # sensor depths used for the rootzone average
    rootzone_weights: list[float] = [7.5, 7.5, 20, 40, 25]  # midpoint-rule layer thicknesses (cm)
    columns: ObsColumns = ObsColumns()

    @model_validator(mode="after")
    def _same_length(self) -> ObsSection:
        if len(self.rootzone_depths_cm) != len(self.rootzone_weights):
            raise ValueError("obs.rootzone_depths_cm and obs.rootzone_weights differ in length")
        return self


class FeaturesSection(_Model):
    smap_constant: float = 0.3  # value of the constant SMAP feature (table column 15)
    hls_angle_constant: float = (
        90.0  # value of the four constant HLS angle features (columns 32-35)
    )
    s1_invalid_above: float = 1000.0  # Sentinel-1 values above this become NaN before the features
    era5_lookback_warmup: Literal["raw", "lead"] = (
        "raw"  # "raw": first N rows keep raw hourly values (notebook); "lead": use era5_lead_hours
    )


class ModelsSection(_Model):
    dir: str = ""  # directory of the 18 joblib models; "" -> <repo>/trainedML


class PostprocSection(_Model):
    reference_sensor: Literal["sentinel1", "sentinel2", "hls"] = (
        "sentinel2"  # the other two sensors are CDF-matched onto this one (the notebook matched onto Sentinel-2)
    )
    sd_divisor: float = 3.29  # sd = (y_high - y_low) / sd_divisor
    sd_floor: float = 0.02  # sd values <= 0 are replaced by this
    hour_spread: int = (
        1  # each value also fills the +/- this many neighbouring hours of the same day
    )


class PrecipSection(_Model):
    source: Literal["mesonet", "era5"] = (
        "mesonet"  # precipitation source (the second `source` key below is for PET)
    )
    fallback: Literal["era5", "none"] = "era5"  # fill hours the primary source lacks from ERA5
    era5_zero_below_mm: float = 0.01  # ERA5 hourly values below this (mm) become 0


class PetSection(_Model):
    source: Literal["era5", "mesonet"] = (
        "era5"  # PET source: ERA5 radiation/temperature, or mesonet solar/air temperature
    )
    fallback: Literal["era5", "none"] = "era5"  # fill hours the primary source lacks from ERA5
    alpha: float = 1.3  # Priestley-Taylor alpha
    gamma: float = 66.8  # psychrometric constant [Pa/K]
    zero_below: float = 1e-6  # PET values below this (mm/h) become 0


class ForcingSection(_Model):
    interval_shift_hours: int = (
        0  # delay precipitation by this many hours (both sources together); 0 = none
    )
    precip: PrecipSection = PrecipSection()
    pet: PetSection = PetSection()


class LgarSection(_Model):
    bin_dir: str = ""  # "" -> <data_root>/bin/<platform>
    ensemble_size: int = 20  # EnKF members (build parameter)
    model_sd_sly: float = 0.03  # surface model-error sd (build parameter)
    model_sd_rly: float = 0.02  # rootzone model-error sd (build parameter)
    seed: int = 0  # 0 keeps random_device (original behaviour)
    workers: int = 0  # 0 -> os.cpu_count()
    verbosity: str = "none"  # LGAR log level
    layer_thickness_cm: list[float] = [5, 10, 15, 30, 40]  # LGAR soil layer thicknesses
    giuh_ordinates: list[float] = [0.06, 0.51, 0.28, 0.12, 0.03]  # GIUH ordinates
    timestep_h: float = 1  # LGAR time step
    field_capacity_psi: float = 340.9  # field capacity suction head [cm]
    wilting_point_psi: float = 15495.0  # wilting point suction head [cm]
    initial_psi: float = 2000.0  # initial suction head [cm]
    initial_sm_perturb: float = 0  # initial soil-moisture perturbation
    soil_params_file: str = ""  # optional van Genuchten parameter file; "" = built-in Hydrus table
    TO_enabled: bool = False  # LGAR `TO_enabled` switch
    adaptive_timestep: bool = False  # LGAR adaptive time step
    free_drainage_enabled: bool = False  # LGAR free drainage at the bottom
    use_closed_form_G: bool = False  # LGAR closed-form capillary drive
    root_zone_depth_cm: float = 60  # LGAR root-zone depth for AET
    mbal_tol: float = 80.0  # LGAR mass-balance tolerance
    ponded_depth_max_cm: float = 0  # LGAR maximum ponded depth
    forcing_resolution_h: float = 1  # forcing time resolution
    max_valid_soil_types: int = 12  # number of soil classes in the Hydrus table
    site_overrides: dict[int, dict[str, Any]] = {}  # [lgar.site_overrides.<id>]

    @field_validator("site_overrides")
    @classmethod
    def _override_keys(cls, v: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
        allowed = set(OVERRIDABLE_LGAR_KEYS)
        for site, d in v.items():
            bad = set(d) - allowed
            if bad:
                raise ValueError(f"lgar.site_overrides.{site}: unknown keys {sorted(bad)}")
        return v


# Keys of [lgar] that are written to the C++ settings file and may be overridden per site.
# `layer_soil_type` is derived at run time but may be overridden (§4.8).
OVERRIDABLE_LGAR_KEYS = tuple(
    k
    for k in LgarSection.model_fields
    if k
    not in {
        "bin_dir",
        "ensemble_size",
        "model_sd_sly",
        "model_sd_rly",
        "seed",
        "workers",
        "site_overrides",
    }
) + ("layer_soil_type",)


class EvaluateSection(_Model):
    study_dir: str = ""  # e.g. <data_root>/shuohao_ground_truth_three_sites
    plots: bool = True  # write the PNG figures
    time_window: list[str] = []  # optional ["YYYY-MM-DD", "YYYY-MM-DD"] for plot x-limits


class RunConfig(_Model):
    run: RunSection = RunSection()
    sites: SitesSection = SitesSection()
    gee: GeeSection = GeeSection()
    obs: ObsSection = ObsSection()
    features: FeaturesSection = FeaturesSection()
    models: ModelsSection = ModelsSection()
    postproc: PostprocSection = PostprocSection()
    forcing: ForcingSection = ForcingSection()
    lgar: LgarSection = LgarSection()
    evaluate: EvaluateSection = EvaluateSection()


def _set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
        if not isinstance(d, dict):
            raise ValueError(f"cannot override {key!r}: {p!r} is not a table")
    d[parts[-1]] = value


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> RunConfig:
    """Load a TOML config (or defaults when ``path`` is None) and apply dotted-key overrides.

    A ``[provenance]`` table in the input is ignored; it is regenerated for every run.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    raw.pop("provenance", None)
    for key, value in (overrides or {}).items():
        _set_dotted(raw, key, value)
    return RunConfig.model_validate(raw)


def _dump(cfg: RunConfig) -> dict[str, Any]:
    # mode="json" turns the int keys of lgar.site_overrides into strings, as TOML requires.
    return cfg.model_dump(mode="json")


# Explanatory TOML comments, injected before the (uniquely named) key in every emitted file.
_COMMENTS = {
    "data_root": '# "" -> $HRSM_DATA_DIR',
    "rootzone_weights": ("# midpoint rule; the validated legacy array used [2.5, 7.5, 20, 40, 25]"),
    "polaris_scale": '# "native" -> bd_0_5.projection().nominalScale() (30 m)',
    "terrain_scale": '# "native" -> each asset\'s nominalScale()',
    "era5_lookback_warmup": (
        '# "raw": rows 0..N-1 keep raw hourly values (notebook) | "lead": compute from era5_lead_hours'
    ),
    "bin_dir": '# "" -> <data_root>/bin/<platform>',
    "seed": "# 0 keeps random_device (original behaviour)",
    "workers": "# 0 -> os.cpu_count()",
}


def _with_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        key = line.split(" = ", 1)[0] if " = " in line else None
        if key in _COMMENTS:
            out.append(_COMMENTS[key])
        out.append(line)
    return "\n".join(out) + "\n"


def resolved_toml(cfg: RunConfig, provenance: dict[str, Any] | None = None) -> str:
    """Every key expanded (defaults + file + overrides) plus an optional ``[provenance]`` table."""
    d = _dump(cfg)
    if provenance is not None:
        d["provenance"] = {k: v for k, v in provenance.items() if v is not None}
    return _with_comments(tomli_w.dumps(d))


def config_for_fingerprint(cfg: RunConfig, sections: list[str]) -> dict[str, Any]:
    """Subset of the resolved config by dotted section/key names (never provenance or run.name)."""
    d = _dump(cfg)
    out: dict[str, Any] = {}
    for s in sections:
        cur: Any = d
        for p in s.split("."):
            cur = cur[p]
        out[s] = cur
    return out
