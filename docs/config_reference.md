# Config reference

Generated from the pydantic model by `scripts/gen_config_reference.py`; do not edit by hand.
Every key has a default equal to the value used in the original notebook. `hrsmpipe config-schema`
prints the same keys as TOML, and `hrsmpipe init <path>` writes them to a file.

`site_ID`, `num_hours`, `endtime_s` and `layer_soil_type` are derived at run time and are not config keys.
A `[provenance]` table in an input TOML is ignored and regenerated.

## `[run]`

| key | type | default | meaning |
|---|---|---|---|
| `run.name` | `str` | `'wisconet_2021_2026'` | run directory name under <data_root>/runs/ (excluded from stage fingerprints) |
| `run.data_root` | `str` | `""` | "" -> $HRSM_DATA_DIR |
| `run.start` | `str` | `'2021-07-16'` | UTC date; axis is [start, end) |
| `run.end` | `str` | `'2026-02-10'` | UTC date, exclusive |

## `[sites]`

| key | type | default | meaning |
|---|---|---|---|
| `sites.csv` | `str` | `'pkg:wisconet_stations.csv'` | sites CSV (site_id,station_code,latitude,longitude); `pkg:` = packaged file; row order = array column order |
| `sites.include` | `list[int]` | `[]` | optional subset of site_id (CSV order is kept) |

## `[gee]`

| key | type | default | meaning |
|---|---|---|---|
| `gee.project` | `str` | `'run-hrsm'` | Earth Engine cloud project |
| `gee.workers_era5` | `int` | `2` | ERA5 request threads |
| `gee.workers_satellite` | `int` | `4` | Sentinel-1/2 and HLSL30 request threads |
| `gee.window_hours` | `int` | `15` | hours per Earth Engine request window (aligned to the month start) |
| `gee.sample_scale_m` | `float` | `100` | NLCD and all time series |
| `gee.polaris_scale` | `Union[float, Literal['native']]` | `'native'` | bd_0_5.projection().nominalScale() (30 m) |
| `gee.terrain_scale` | `Union[float, Literal['native']]` | `'native'` | each asset's nominalScale() |
| `gee.era5_lead_hours` | `int` | `144` | 120 + 24, as the notebook's start_date_era5 |

## `[obs]`

| key | type | default | meaning |
|---|---|---|---|
| `obs.source` | `Literal['wisconet_api', 'wisconet_csv_dir', 'table']` | `'wisconet_api'` | observation source |
| `obs.path` | `str` | `""` | csv dir / table path (may contain {site_id}) |
| `obs.timezone` | `str` | `'UTC'` | naive table timestamps only |
| `obs.rootzone_depths_cm` | `list[float]` | `[5, 10, 20, 50, 100]` | sensor depths used for the rootzone average |
| `obs.rootzone_weights` | `list[float]` | `[7.5, 7.5, 20, 40, 25]` | midpoint-rule layer thicknesses (cm) |
| `obs.columns.time` | `str` | `'time'` | timestamp column |
| `obs.columns.site_id` | `str` | `'site_id'` | site column (not needed when `obs.path` contains `{site_id}`) |
| `obs.columns.sm_surface` | `str` | `'sm_surface'` | surface soil moisture column |
| `obs.columns.sm_rootzone` | `str` | `'sm_rootzone'` | rootzone soil moisture column |
| `obs.columns.precip_mm` | `str` | `'precip_mm'` | hourly precipitation column (mm); "" = none |
| `obs.columns.air_temp_c` | `str` | `""` | air temperature column (deg C); needed for `forcing.pet.source = "mesonet"` |
| `obs.columns.solar_w` | `str` | `""` | solar radiation column (W/m2); needed for `forcing.pet.source = "mesonet"` |

## `[features]`

| key | type | default | meaning |
|---|---|---|---|
| `features.smap_constant` | `float` | `0.3` | value of the constant SMAP feature (table column 15) |
| `features.hls_angle_constant` | `float` | `90.0` | value of the four constant HLS angle features (columns 32-35) |
| `features.s1_invalid_above` | `float` | `1000.0` | Sentinel-1 values above this become NaN before the features |
| `features.era5_lookback_warmup` | `Literal['raw', 'lead']` | `'raw'` | "raw": first N rows keep raw hourly values (notebook); "lead": use era5_lead_hours |

## `[models]`

| key | type | default | meaning |
|---|---|---|---|
| `models.dir` | `str` | `""` | directory of the 18 joblib models; "" -> <repo>/trainedML |

## `[postproc]`

| key | type | default | meaning |
|---|---|---|---|
| `postproc.reference_sensor` | `Literal['sentinel1', 'sentinel2', 'hls']` | `'sentinel2'` | the other two sensors are CDF-matched onto this one (the notebook matched onto Sentinel-2) |
| `postproc.sd_divisor` | `float` | `3.29` | sd = (y_high - y_low) / sd_divisor |
| `postproc.sd_floor` | `float` | `0.02` | sd values <= 0 are replaced by this |
| `postproc.hour_spread` | `int` | `1` | each value also fills the +/- this many neighbouring hours of the same day |

## `[forcing]`

| key | type | default | meaning |
|---|---|---|---|
| `forcing.interval_shift_hours` | `int` | `0` | delay precipitation by this many hours (both sources together); 0 = none |
| `forcing.precip.source` | `Literal['mesonet', 'era5']` | `'mesonet'` | precipitation source (the second `source` key below is for PET) |
| `forcing.precip.fallback` | `Literal['era5', 'none']` | `'era5'` | fill hours the primary source lacks from ERA5 |
| `forcing.precip.era5_zero_below_mm` | `float` | `0.01` | ERA5 hourly values below this (mm) become 0 |
| `forcing.pet.source` | `Literal['era5', 'mesonet']` | `'era5'` | PET source: ERA5 radiation/temperature, or mesonet solar/air temperature |
| `forcing.pet.fallback` | `Literal['era5', 'none']` | `'era5'` | fill hours the primary source lacks from ERA5 |
| `forcing.pet.alpha` | `float` | `1.3` | Priestley-Taylor alpha |
| `forcing.pet.gamma` | `float` | `66.8` | psychrometric constant [Pa/K] |
| `forcing.pet.zero_below` | `float` | `1e-06` | PET values below this (mm/h) become 0 |

## `[lgar]`

| key | type | default | meaning |
|---|---|---|---|
| `lgar.bin_dir` | `str` | `""` | "" -> <data_root>/bin/<platform> |
| `lgar.ensemble_size` | `int` | `20` | EnKF members (build parameter) |
| `lgar.model_sd_sly` | `float` | `0.03` | surface model-error sd (build parameter) |
| `lgar.model_sd_rly` | `float` | `0.02` | rootzone model-error sd (build parameter) |
| `lgar.seed` | `int` | `0` | 0 keeps random_device (original behaviour) |
| `lgar.workers` | `int` | `0` | 0 -> os.cpu_count() |
| `lgar.verbosity` | `str` | `'none'` | LGAR log level |
| `lgar.layer_thickness_cm` | `list[float]` | `[5, 10, 15, 30, 40]` | LGAR soil layer thicknesses |
| `lgar.giuh_ordinates` | `list[float]` | `[0.06, 0.51, 0.28, 0.12, 0.03]` | GIUH ordinates |
| `lgar.timestep_h` | `float` | `1` | LGAR time step |
| `lgar.field_capacity_psi` | `float` | `340.9` | field capacity suction head [cm] |
| `lgar.wilting_point_psi` | `float` | `15495.0` | wilting point suction head [cm] |
| `lgar.initial_psi` | `float` | `2000.0` | initial suction head [cm] |
| `lgar.initial_sm_perturb` | `float` | `0` | initial soil-moisture perturbation |
| `lgar.soil_params_file` | `str` | `""` | optional van Genuchten parameter file; "" = built-in Hydrus table |
| `lgar.TO_enabled` | `bool` | `False` | LGAR `TO_enabled` switch |
| `lgar.adaptive_timestep` | `bool` | `False` | LGAR adaptive time step |
| `lgar.free_drainage_enabled` | `bool` | `False` | LGAR free drainage at the bottom |
| `lgar.use_closed_form_G` | `bool` | `False` | LGAR closed-form capillary drive |
| `lgar.root_zone_depth_cm` | `float` | `60` | LGAR root-zone depth for AET |
| `lgar.mbal_tol` | `float` | `80.0` | LGAR mass-balance tolerance |
| `lgar.ponded_depth_max_cm` | `float` | `0` | LGAR maximum ponded depth |
| `lgar.forcing_resolution_h` | `float` | `1` | forcing time resolution |
| `lgar.max_valid_soil_types` | `int` | `12` | number of soil classes in the Hydrus table |
| `lgar.site_overrides` | `dict[int, dict[str, Any]]` | `{}` | [lgar.site_overrides.<id>] |

## `[evaluate]`

| key | type | default | meaning |
|---|---|---|---|
| `evaluate.study_dir` | `str` | `""` | e.g. <data_root>/shuohao_ground_truth_three_sites |
| `evaluate.plots` | `bool` | `True` | write the PNG figures |
| `evaluate.time_window` | `list[str]` | `[]` | optional ["YYYY-MM-DD", "YYYY-MM-DD"] for plot x-limits |

## Per-site LGAR overrides

`[lgar.site_overrides.<site_id>]` may set any of: `verbosity`, `layer_thickness_cm`, `giuh_ordinates`, `timestep_h`, `field_capacity_psi`, `wilting_point_psi`, `initial_psi`, `initial_sm_perturb`, `soil_params_file`, `TO_enabled`, `adaptive_timestep`, `free_drainage_enabled`, `use_closed_form_G`, `root_zone_depth_cm`, `mbal_tol`, `ponded_depth_max_cm`, `forcing_resolution_h`, `max_valid_soil_types`, `layer_soil_type`.
